from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


def _api() -> int:
    if sys.platform == "win32":
        return cv2.CAP_MSMF
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def open_camera(index: int, width: int, height: int, fps: float | None = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, _api())
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} could not be opened (missing, in use, or permission denied)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def probe_cameras(max_index: int = 5) -> list[dict]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _api())
        if cap.isOpened():
            ok, frame = cap.read()
            found.append({
                "index": i,
                "readable": bool(ok),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
            })
        cap.release()
    return found


class LatestFrameReader:
    """Drains the camera on a thread so `latest()` never returns a stale frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self.frames_read = 0
        self.failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        misses = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                misses += 1
                if misses > 50:
                    self.failed = True  # disconnected or taken by another app
                    return
                time.sleep(0.02)
                continue
            misses = 0
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()
                self.frames_read += 1

    def latest(self, timeout: float = 2.0) -> tuple[np.ndarray, float] | None:
        """BGR frame and its monotonic timestamp."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame, self._stamp
            if self.failed:
                return None
            time.sleep(0.01)
        return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


def to_rgb(bgr: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    if size and (bgr.shape[1], bgr.shape[0]) != size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
