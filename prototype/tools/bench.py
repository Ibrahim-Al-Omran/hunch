"""Benchmark inference cost and camera capture strategies.

    python tools/bench.py --camera 1

Frames are held in memory only. Process CPU is reported as % of one core;
on Windows the camera FrameServer service is measured too, because camera
decoding partly happens outside our process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv.capture import LatestFrameReader, open_camera, to_rgb  # noqa: E402
from hunch_cv.pose import make_backend  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
PROC = psutil.Process()


def cpu_s(p: psutil.Process) -> float:
    t = p.cpu_times()
    return t.user + t.system


def frameserver_procs() -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if p.info["name"] and p.info["name"].lower() == "svchost.exe" and "FrameServer" in " ".join(p.info["cmdline"] or []):
                out.append(p)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return out


class CpuMeter:
    def __init__(self):
        self.extra = frameserver_procs()

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c0 = cpu_s(PROC)
        self.e0 = [self._safe(p) for p in self.extra]
        return self

    @staticmethod
    def _safe(p):
        try:
            return cpu_s(p)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self.proc_pct = 100 * (cpu_s(PROC) - self.c0) / self.wall
        deltas = [(self._safe(p) - e) for p, e in zip(self.extra, self.e0) if e is not None and self._safe(p) is not None]
        self.frameserver_pct = 100 * sum(deltas) / self.wall if deltas else None


def bench_inference(frames: list[np.ndarray], specs: list[str], repeats: int) -> list[dict]:
    results = []
    for spec in specs:
        backend_name, res = spec.split("@")
        size = (320, 240) if res == "320" else (640, 480)
        rss0 = PROC.memory_info().rss
        t0 = time.perf_counter()
        backend = make_backend(backend_name)
        load_ms = (time.perf_counter() - t0) * 1e3
        rgbs = [to_rgb(f, size) for f in frames]
        backend.detect(rgbs[0])  # warm-up
        walls, found, cpu_total = [], 0, 0.0
        for i in range(repeats):
            rgb = rgbs[i % len(rgbs)]
            # OS CPU counters tick at ~15.6 ms on Windows, so only the total is meaningful
            c0, w0 = cpu_s(PROC), time.perf_counter()
            found += backend.detect(rgb) is not None
            walls.append((time.perf_counter() - w0) * 1e3)
            cpu_total += cpu_s(PROC) - c0
            time.sleep(0.05)  # avoid measuring a hot loop
        rss1 = PROC.memory_info().rss
        results.append({
            "variant": spec, "load_ms": load_ms, "rss_after_mb": rss1 / 2**20, "rss_delta_mb": (rss1 - rss0) / 2**20,
            "wall_ms_median": float(np.median(walls)), "wall_ms_p95": float(np.quantile(walls, 0.95)),
            "cpu_ms_mean": 1e3 * cpu_total / repeats, "detected": f"{found}/{repeats}",
        })
        if hasattr(backend, "close"):
            backend.close()
        print(results[-1])
    return results


def bench_capture_open(camera: int, size: tuple[int, int], fps: float | None, seconds: float) -> dict:
    cap = open_camera(camera, *size, fps=fps)
    reader = LatestFrameReader(cap)
    reader.latest(5)
    time.sleep(1.0)
    n0 = reader.frames_read
    with CpuMeter() as m:
        time.sleep(seconds)
    n1 = reader.frames_read
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    reader.close()
    return {"mode": f"open {size[0]}x{size[1]} fps={fps or 'default'}", "actual_size": actual,
            "delivered_fps": (n1 - n0) / m.wall, "process_cpu_pct_one_core": m.proc_pct,
            "frameserver_cpu_pct_one_core": m.frameserver_pct}


def bench_capture_reopen(camera: int, size: tuple[int, int], cycles: int, interval: float) -> dict:
    opens, firsts, settle = [], [], []
    with CpuMeter() as m:
        for _ in range(cycles):
            t0 = time.perf_counter()
            cap = open_camera(camera, *size)
            t1 = time.perf_counter()
            ok, frame = cap.read()
            t2 = time.perf_counter()
            # auto-exposure settles over the first frames; count until brightness stabilises
            prev, n = float(frame.mean()) if ok else 0.0, 1
            while n < 30:
                ok, frame = cap.read()
                if not ok:
                    break
                cur = float(frame.mean())
                n += 1
                if abs(cur - prev) < 1.0:
                    break
                prev = cur
            t3 = time.perf_counter()
            cap.release()
            opens.append((t1 - t0) * 1e3)
            firsts.append((t2 - t0) * 1e3)
            settle.append((t3 - t0) * 1e3)
            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
    return {"mode": f"reopen every {interval:g}s {size[0]}x{size[1]}",
            "open_ms_median": float(np.median(opens)), "first_frame_ms_median": float(np.median(firsts)),
            "stable_frame_ms_median": float(np.median(settle)),
            "process_cpu_pct_one_core": m.proc_pct, "frameserver_cpu_pct_one_core": m.frameserver_pct}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--variants", default="mp-lite@320,mp-full@320,mp-lite@640,onnx-lite@320,onnx-full@320,onnx-lite@640")
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--capture-seconds", type=float, default=15)
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--image", type=Path, help="benchmark inference on this image instead of camera frames")
    args = ap.parse_args()

    if args.image:
        frames = [cv2.imread(str(args.image))]
    else:
        cap = open_camera(args.camera, 640, 480)
        reader = LatestFrameReader(cap)
        reader.latest(5)
        time.sleep(1.5)
        frames = []
        for _ in range(10):
            frames.append(reader.latest()[0].copy())
            time.sleep(0.3)
        reader.close()

    report = {"machine": {"cpu_count": os.cpu_count(), "platform": sys.platform},
              "source": str(args.image or f"camera {args.camera}"),
              "inference": bench_inference(frames, args.variants.split(","), args.repeats)}
    if not args.skip_capture:
        report["capture"] = []
        for size, fps in (((640, 480), None), ((320, 240), None), ((320, 240), 5.0)):
            report["capture"].append(bench_capture_open(args.camera, size, fps, args.capture_seconds))
            print(report["capture"][-1])
        report["capture"].append(bench_capture_reopen(args.camera, (320, 240), 6, 2.0))
        print(report["capture"][-1])

    out = DATA / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
