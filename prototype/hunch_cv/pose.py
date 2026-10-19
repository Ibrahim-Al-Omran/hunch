"""Pose backends: MediaPipe Tasks (reference) and a plain ONNX Runtime port.

The ONNX backend reimplements the BlazePose two-stage pipeline (person
detector -> rotated ROI -> landmark model) so the same logic can be ported to
Rust without the MediaPipe framework. Both backends run every frame from
scratch (no tracking), matching sparse sampling every few seconds.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .landmarks import Landmarks, NUM_LANDMARKS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class PoseBackend(Protocol):
    name: str

    def detect(self, rgb: np.ndarray) -> Landmarks | None: ...


class MediaPipeBackend:
    def __init__(self, variant: str = "lite", num_threads: int | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.name = f"mp-{variant}"
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS_DIR / f"pose_landmarker_{variant}.task")),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> Landmarks | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        norm = np.array(
            [[p.x, p.y, p.z, p.visibility or 0.0, p.presence or 0.0] for p in lms], dtype=np.float64
        )
        world = None
        if result.pose_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]], dtype=np.float64)
        h, w = rgb.shape[:2]
        return Landmarks(norm, world, w, h)

    def close(self) -> None:
        self._landmarker.close()


def ssd_anchors() -> np.ndarray:
    """Anchor centres for the BlazePose detector (fixed anchor size of 1)."""
    strides = [8, 16, 32, 32, 32]
    size = 224
    anchors = []
    layer = 0
    while layer < len(strides):
        same = 0
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            same += 2  # one aspect ratio plus the interpolated scale
            last += 1
        fm = math.ceil(size / strides[layer])
        for y in range(fm):
            for x in range(fm):
                for _ in range(same):
                    anchors.append(((x + 0.5) / fm, (y + 0.5) / fm))
        layer = last
    return np.asarray(anchors, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def refine_from_heatmap(xy: np.ndarray, heatmap: np.ndarray, kernel: int = 7,
                        min_confidence: float = 0.5) -> np.ndarray:
    """Confidence-weighted centroid in a small heatmap window (MediaPipe's refinement).

    `xy` is normalized to the crop; `heatmap` is (rows, cols, landmarks) logits.
    """
    rows, cols = heatmap.shape[:2]
    off = (kernel - 1) // 2
    out = xy.copy()
    for i in range(len(xy)):
        c, r = int(xy[i, 0] * cols), int(xy[i, 1] * rows)
        if not (0 <= c < cols and 0 <= r < rows):
            continue
        r0, r1, c0, c1 = max(0, r - off), min(rows, r + off + 1), max(0, c - off), min(cols, c + off + 1)
        conf = _sigmoid(heatmap[r0:r1, c0:c1, i].astype(np.float64))
        total = conf.sum()
        if conf.max() < min_confidence or total <= 0:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        out[i, 0] = (cc * conf).sum() / total / cols
        out[i, 1] = (rr * conf).sum() / total / rows
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between box a and boxes b, boxes as (xmin, ymin, xmax, ymax)."""
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


class OnnxBackend:
    DET_SIZE = 224
    LM_SIZE = 256

    def __init__(self, variant: str = "lite", num_threads: int = 1,
                 det_threshold: float = 0.5, presence_threshold: float = 0.5,
                 refine_heatmap: bool = True, border: int = cv2.BORDER_REPLICATE):
        import onnxruntime as ort

        self.name = f"onnx-{variant}"
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        d = MODELS_DIR / variant
        self._det = ort.InferenceSession(str(d / "pose_detector.onnx"), so, providers=["CPUExecutionProvider"])
        self._lm = ort.InferenceSession(str(d / "pose_landmarks_detector.onnx"), so, providers=["CPUExecutionProvider"])
        self._det_in = self._det.get_inputs()[0].name
        self._lm_in = self._lm.get_inputs()[0].name
        self._anchors = ssd_anchors()
        self.det_threshold = det_threshold
        self.presence_threshold = presence_threshold
        self.refine_heatmap = refine_heatmap
        self.border = border
        self.last_timing: dict[str, float] = {}

    def _detect_person(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        s = self.DET_SIZE
        scale = s / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        pad_x, pad_y = (s - nw) // 2, (s - nh) // 2
        canvas = np.zeros((s, s, 3), np.float32)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (canvas / 127.5 - 1.0)[None]
        raw_boxes, raw_scores = self._det.run(None, {self._det_in: x})
        scores = _sigmoid(np.clip(raw_scores[0, :, 0], -100, 100))
        keep = scores >= self.det_threshold
        if not keep.any():
            return None
        raw = raw_boxes[0][keep] / s
        anc = self._anchors[keep]
        sc = scores[keep]
        cx = raw[:, 0] + anc[:, 0]
        cy = raw[:, 1] + anc[:, 1]
        bw, bh = raw[:, 2], raw[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        kps = raw[:, 4:12].reshape(-1, 4, 2) + anc[:, None, :]
        # Weighted NMS, keeping only the strongest cluster (single user).
        best = int(np.argmax(sc))
        cluster = _iou(boxes[best], boxes) > 0.3
        wts = sc[cluster][:, None, None]
        kp = (kps[cluster] * wts).sum(0) / wts.sum()
        # Undo letterboxing; keypoints back to source pixels.
        kp_px = (kp * s - np.array([pad_x, pad_y])) / scale
        return kp_px, float(sc[best])

    @staticmethod
    def _roi_from_keypoints(kp_px: np.ndarray):
        """Square ROI centred on the hip keypoint, sized and rotated by the hip->body-top vector."""
        (x0, y0), (x1, y1) = kp_px[0], kp_px[1]
        rotation = math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0)
        rotation = (rotation + math.pi) % (2 * math.pi) - math.pi
        radius = math.hypot(x1 - x0, y1 - y0)
        return (x0, y0), 2.0 * radius * 1.25, rotation

    def detect(self, rgb: np.ndarray) -> Landmarks | None:
        t0 = time.perf_counter()
        det = self._detect_person(rgb)
        t1 = time.perf_counter()
        if det is None:
            self.last_timing = {"detector_ms": (t1 - t0) * 1e3}
            return None
        kp_px, _ = det
        (cx, cy), side, rot = self._roi_from_keypoints(kp_px)
        n = self.LM_SIZE
        c, s_ = math.cos(rot), math.sin(rot)
        k = side / n
        # Crop pixel (u, v) -> image: centre + R(rot) * ((u, v) - n/2) * side/n
        fwd = np.array([[c * k, -s_ * k, cx - (c * k * n / 2) + (s_ * k * n / 2)],
                        [s_ * k, c * k, cy - (s_ * k * n / 2) - (c * k * n / 2)]])
        crop = cv2.warpAffine(rgb, fwd, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=self.border)
        x = (crop.astype(np.float32) / 255.0)[None]
        outs = self._lm.run(None, {self._lm_in: x})
        t2 = time.perf_counter()
        self.last_timing = {"detector_ms": (t1 - t0) * 1e3, "landmark_ms": (t2 - t1) * 1e3}
        lm_raw, flag, _, heatmap, world_raw = outs
        if float(np.asarray(flag).ravel()[0]) < self.presence_threshold:
            return None
        lm = lm_raw.reshape(-1, 5)[:NUM_LANDMARKS].astype(np.float64)
        if self.refine_heatmap:
            lm[:, :2] = refine_from_heatmap(lm[:, :2] / n, heatmap[0]) * n
        uv1 = np.concatenate([lm[:, :2], np.ones((NUM_LANDMARKS, 1))], axis=1)
        img_xy = uv1 @ fwd.T
        h, w = rgb.shape[:2]
        norm = np.empty((NUM_LANDMARKS, 5))
        norm[:, 0] = img_xy[:, 0] / w
        norm[:, 1] = img_xy[:, 1] / h
        norm[:, 2] = lm[:, 2] / n * side / w
        norm[:, 3] = _sigmoid(lm[:, 3])
        norm[:, 4] = _sigmoid(lm[:, 4])
        wl = world_raw.reshape(-1, 3)[:NUM_LANDMARKS].astype(np.float64)
        world = wl.copy()
        world[:, 0] = c * wl[:, 0] - s_ * wl[:, 1]
        world[:, 1] = s_ * wl[:, 0] + c * wl[:, 1]
        return Landmarks(norm, world, w, h)

    def close(self) -> None:
        pass


def make_backend(spec: str) -> PoseBackend:
    """`mp-lite`, `mp-full`, `onnx-lite`, `onnx-full`."""
    kind, variant = spec.split("-", 1)
    if kind == "mp":
        return MediaPipeBackend(variant)
    if kind == "onnx":
        return OnnxBackend(variant)
    raise ValueError(f"unknown backend {spec!r}")
