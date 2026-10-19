"""Live preview: skeleton overlay, calibration, and per-sample posture state.

Keys: c = calibrate upright (10 s), s = record slouch reference (6 s),
      m = toggle rule/learned scoring, [ / ] = threshold, q/Esc = quit.
Frames are never written to disk.

    python tools/live.py --camera 1 --interval 2
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv.calibration import Baseline, CalibrationError, calibrate, learn_slouch_direction  # noqa: E402
from hunch_cv.capture import LatestFrameReader, open_camera, to_rgb  # noqa: E402
from hunch_cv.features import compute_features  # noqa: E402
from hunch_cv.overlay import draw_pose, put_lines  # noqa: E402
from hunch_cv.pose import make_backend  # noqa: E402
from hunch_cv.scoring import POSSIBLE, GOOD, ScoringConfig, WindowMedian, assess, RULE_FEATURES  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "data" / "baseline.json"
COLORS = {GOOD: (80, 200, 80), POSSIBLE: (40, 140, 255)}


def collect(reader, backend, size, seconds, label):
    lms, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        frame, _ = reader.latest()
        lm = backend.detect(to_rgb(frame, size))
        lms.append(lm)
        view = frame.copy()
        draw_pose(view, lm)
        put_lines(view, [f"{label}: hold still ({t_end - time.monotonic():.1f}s)"])
        cv2.imshow("hunch - live", view)
        cv2.waitKey(1)
        time.sleep(0.2)
    return lms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="mp-lite")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between analysed frames")
    args = ap.parse_args()

    backend = make_backend(args.backend)
    size = (args.width, args.height)
    reader = LatestFrameReader(open_camera(args.camera, 640, 480))
    cfg = ScoringConfig()
    baseline = Baseline.load(BASELINE) if BASELINE.exists() else None
    window = WindowMedian(3)
    last = 0.0
    result_lines: list[str] = ["press c to calibrate"]
    color = (200, 200, 200)
    lm = None
    while True:
        got = reader.latest()
        if got is None:
            sys.exit("camera stopped delivering frames")
        frame = got[0]
        now = time.monotonic()
        if now - last >= args.interval:
            last = now
            t0 = time.perf_counter()
            lm = backend.detect(to_rgb(frame, size))
            ms = (time.perf_counter() - t0) * 1e3
            if baseline is not None:
                ff = compute_features(lm, baseline.side, baseline.scale_px) if lm is not None else None
                a = assess(ff, baseline, cfg)
                win = window.push(a)
                top = sorted(((k, v) for k, v in a.deviations.items() if k in RULE_FEATURES),
                             key=lambda kv: -abs(kv[1]))[:3]
                result_lines = [
                    f"{a.state}  score {a.score:.2f}  window {win:.2f}  thr {cfg.threshold:.1f} ({cfg.method})",
                    f"visibility {a.visibility:.2f}  {' '.join(a.reasons)}",
                    "  ".join(f"{k} {v:+.1f}" for k, v in top),
                    f"inference {ms:.0f} ms",
                ]
                color = COLORS.get(a.state, (160, 160, 160))
                if math.isfinite(win) and win >= cfg.threshold:
                    color = (40, 40, 255)
        view = frame.copy()
        draw_pose(view, lm)
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), color, 6)
        put_lines(view, result_lines)
        cv2.imshow("hunch - live", view)
        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("c"):
            try:
                baseline = calibrate(collect(reader, backend, size, 10, "Sit upright"),
                                     meta={"backend": args.backend, "size": list(size)})
                baseline.save(BASELINE)
                result_lines = [f"calibrated ({baseline.samples} samples, near side {baseline.side})"]
            except CalibrationError as e:
                result_lines = [f"calibration failed: {e}"]
        if k == ord("s") and baseline is not None:
            try:
                baseline.slouch_direction = learn_slouch_direction(
                    baseline, collect(reader, backend, size, 6, "Slouch"), tuple(RULE_FEATURES))
                baseline.save(BASELINE)
                result_lines = ["slouch reference stored"]
            except CalibrationError as e:
                result_lines = [f"slouch reference failed: {e}"]
        if k == ord("m"):
            cfg.method = "learned" if cfg.method == "rule" and baseline and baseline.slouch_direction else "rule"
        if k == ord("["):
            cfg.threshold = max(0.5, cfg.threshold - 0.5)
        if k == ord("]"):
            cfg.threshold += 0.5
    reader.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
