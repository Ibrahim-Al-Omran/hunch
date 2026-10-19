from __future__ import annotations

import cv2
import numpy as np

from .landmarks import CONNECTIONS, Landmarks


def draw_pose(bgr: np.ndarray, lm: Landmarks | None, min_vis: float = 0.5) -> None:
    if lm is None:
        return
    sx = bgr.shape[1] / lm.width
    sy = bgr.shape[0] / lm.height
    pts = (lm.xy * [sx, sy]).astype(int)
    vis = lm.visibility
    for a, b in CONNECTIONS:
        good = min(vis[a], vis[b]) >= min_vis
        cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), (80, 220, 80) if good else (60, 60, 160), 2, cv2.LINE_AA)
    for i in range(min(25, len(pts))):
        color = (255, 255, 255) if vis[i] >= min_vis else (90, 90, 200)
        cv2.circle(bgr, tuple(pts[i]), 3, color, -1, cv2.LINE_AA)


def put_lines(bgr: np.ndarray, lines: list[str], origin=(10, 24), scale: float = 0.55) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(26 * scale / 0.55)
