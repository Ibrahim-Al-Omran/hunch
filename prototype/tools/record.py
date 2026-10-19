"""Guided recording session: prompts each posture condition and logs landmarks.

Only landmark coordinates are written (JSONL). Camera frames stay in memory
and are discarded unless --save-frames is given.

    python tools/record.py --camera 1 --session s1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv.capture import LatestFrameReader, open_camera, to_rgb  # noqa: E402
from hunch_cv.features import choose_side, compute_features  # noqa: E402
from hunch_cv.overlay import draw_pose, put_lines  # noqa: E402
from hunch_cv.pose import make_backend  # noqa: E402
from hunch_cv.protocol import PROTOCOL  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
RESOLUTIONS = {"320": (320, 240), "640": (640, 480)}
WINDOW = "hunch - recording (Esc to abort)"


def parse_variants(spec: str) -> list[tuple[str, str, tuple[int, int]]]:
    out = []
    for item in spec.split(","):
        backend, res = item.strip().split("@")
        out.append((item.strip(), backend, RESOLUTIONS[res]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--session", required=True)
    ap.add_argument("--variants", default="mp-lite@320,mp-full@320,mp-lite@640,onnx-lite@320",
                    help="first one drives the preview overlay")
    ap.add_argument("--rate", type=float, default=5.0, help="samples per second while recording")
    ap.add_argument("--quick", action="store_true", help="one third of the normal step durations")
    ap.add_argument("--transition", type=float, default=3.0, help="unlabelled seconds before each step")
    ap.add_argument("--notes", default="", help="free text, e.g. camera position and lighting")
    ap.add_argument("--save-frames", action="store_true", help="also store low-res JPEG frames locally")
    args = ap.parse_args()

    variants = parse_variants(args.variants)
    backends = {}
    for _, b, _ in variants:
        if b not in backends:
            backends[b] = make_backend(b)

    cap = open_camera(args.camera, 640, 480)
    reader = LatestFrameReader(cap)
    first = reader.latest(5.0)
    if first is None:
        sys.exit("camera opened but delivered no frames")
    cam_h, cam_w = first[0].shape[:2]

    out_dir = DATA / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.session}.jsonl"
    if path.exists():
        sys.exit(f"{path} already exists; choose another --session name")
    frames_dir = DATA / "frames" / args.session if args.save_frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    scale = 1 / 3 if args.quick else 1.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    primary_key, primary_backend, primary_res = variants[0]

    # Positioning: live preview until the user confirms framing.
    while True:
        got = reader.latest()
        if got is None:
            sys.exit("camera stopped delivering frames")
        frame = got[0]
        lm = backends[primary_backend].detect(to_rgb(frame, primary_res))
        view = frame.copy()
        draw_pose(view, lm)
        lines = ["Position yourself as you normally sit.", "SPACE = start session, Esc = quit"]
        if lm is not None:
            side = choose_side([lm])
            lines.append(f"near side: {side}   key landmark visibility: {compute_features(lm, side).visibility:.2f}")
        else:
            lines.append("no person detected")
        put_lines(view, lines)
        cv2.imshow(WINDOW, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            reader.close()
            return
        if k == 32:
            break

    period = 1.0 / args.rate
    completed = False
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "header", "session": args.session, "camera": args.camera,
            "camera_resolution": [cam_w, cam_h], "variants": [v[0] for v in variants],
            "rate": args.rate, "quick": args.quick, "notes": args.notes,
            "started": datetime.now(timezone.utc).isoformat(),
            "protocol": [{"code": s.code, "name": s.name, "seconds": s.seconds * scale} for s in PROTOCOL],
        }) + "\n")
        t_start = time.monotonic()
        try:
            for i, step in enumerate(PROTOCOL):
                for phase, dur in (("transition", args.transition), ("step", step.seconds * scale)):
                    t_phase = time.monotonic()
                    next_tick = t_phase
                    while (elapsed := time.monotonic() - t_phase) < dur:
                        got = reader.latest()
                        if got is None:
                            raise RuntimeError("camera stopped delivering frames")
                        frame, stamp = got
                        results = {}
                        primary_lm = None
                        for key, b, res in variants:
                            t0 = time.perf_counter()
                            lm = backends[b].detect(to_rgb(frame, res))
                            ms = (time.perf_counter() - t0) * 1e3
                            if key == primary_key:
                                primary_lm = lm
                            results[key] = None if lm is None else {**lm.to_json(), "w": res[0], "h": res[1],
                                                                    "ms": round(ms, 2)}
                        sample = {"type": "sample", "t": round(stamp - t_start, 3), "cond": step.code,
                                  "phase": phase, "step_t": round(elapsed, 3), "results": results}
                        f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                        if frames_dir:
                            cv2.imwrite(str(frames_dir / f"{stamp - t_start:09.3f}_{step.code}.jpg"),
                                        cv2.resize(frame, (320, 240)))

                        view = frame.copy()
                        draw_pose(view, primary_lm)
                        head = "GET READY" if phase == "transition" else f"[{step.code}] {step.name}"
                        remaining = dur - elapsed
                        put_lines(view, [f"{head}  ({remaining:4.1f}s)  step {i + 1}/{len(PROTOCOL)}", step.prompt])
                        cv2.imshow(WINDOW, view)
                        if (cv2.waitKey(1) & 0xFF) == 27:
                            raise KeyboardInterrupt
                        next_tick += period
                        time.sleep(max(0.0, next_tick - time.monotonic()))
            completed = True
        except KeyboardInterrupt:
            print("aborted; partial session kept")
        finally:
            f.write(json.dumps({"type": "end", "completed": completed}) + "\n")
            reader.close()
            cv2.destroyAllWindows()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
