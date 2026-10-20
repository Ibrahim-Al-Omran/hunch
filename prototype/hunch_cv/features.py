"""Pure geometry: one frame's landmarks -> posture features.

Features use the side of the body nearest the camera (chosen at calibration)
and are expressed in units of the user's calibrated face size (`scale`, nose
to near ear in pixels). Dividing by a fixed calibrated scale instead of the
per-frame face size avoids injecting face-size jitter into every feature.
None of them is meaningful on its own; scoring compares them to the baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import landmarks as L
from .landmarks import Landmarks

SIDES = {
    "left": {"ear": L.LEFT_EAR, "eye": L.LEFT_EYE, "shoulder": L.LEFT_SHOULDER,
             "mouth": L.MOUTH_LEFT, "hip": L.LEFT_HIP},
    "right": {"ear": L.RIGHT_EAR, "eye": L.RIGHT_EYE, "shoulder": L.RIGHT_SHOULDER,
              "mouth": L.MOUTH_RIGHT, "hip": L.RIGHT_HIP},
}

FEATURES = (
    "head_y",          # mean of near ear and nose height (down = larger)
    "shoulder_y",      # mean shoulder height (sinking into the chair = larger)
    "ear_forward",     # near ear ahead of near shoulder, towards the facing direction
    "neck_ratio",      # vertical near ear-above-shoulder distance
    "face_scale",      # log(face size / calibrated face size): head closer to camera -> larger
    "head_pitch",      # nose below ear / current face size: looking down -> larger
    "head_turn",       # nose horizontal offset from near eye: turning changes it
    "yaw",             # nose-to-ear asymmetry; relies on the occluded far ear, diagnostic only
    "shoulder_width",  # shoulder span
    "world_neck_deg",  # 3D ear-shoulder angle from vertical (model's metric estimate)
    "torso_deg",       # 2D shoulder-hip angle from vertical, only when hips are visible
)


@dataclass
class FrameFeatures:
    values: dict[str, float]
    visibility: float  # weakest required landmark
    side: str
    flags: list[str] = field(default_factory=list)

    def get(self, name: str) -> float:
        return self.values.get(name, math.nan)


def choose_side(lms: list[Landmarks]) -> str:
    """Side nearest the camera.

    MediaPipe reports near-1.0 visibility even for the hidden ear, so the
    model's relative depth (smaller z = nearer) decides; visibility only breaks ties.
    """
    if not lms:
        return "left"

    def mean(side: str, col: int) -> float:
        idx = [SIDES[side]["ear"], SIDES[side]["shoulder"]]
        return float(np.mean([lm.norm[idx, col].mean() for lm in lms]))

    dz = mean("left", 2) - mean("right", 2)
    if abs(dz) > 0.02:
        return "left" if dz < 0 else "right"
    return "left" if mean("left", 3) >= mean("right", 3) else "right"


def face_size(lm: Landmarks, side: str) -> float:
    xy = lm.xy
    return float(np.linalg.norm(xy[L.NOSE] - xy[SIDES[side]["ear"]]))


def _angle_from_vertical(v: np.ndarray) -> float:
    """Degrees between an upward-pointing vector and straight up (y is down)."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return math.nan
    return math.degrees(math.acos(max(-1.0, min(1.0, -v[1] / n))))


def compute_features(lm: Landmarks, side: str, scale: float | None = None,
                     hip_visibility: float = 0.6) -> FrameFeatures:
    """`scale` is the calibrated face size in pixels; defaults to this frame's face size."""
    s = SIDES[side]
    xy = lm.xy
    vis = lm.visibility
    ear, sh = xy[s["ear"]], xy[s["shoulder"]]
    nose = xy[L.NOSE]
    required = [L.NOSE, s["ear"], s["eye"], s["shoulder"]]
    visibility = float(vis[required].min())

    face = face_size(lm, side)
    flags: list[str] = []
    if face < 1.0:
        flags.append("degenerate_face")
        return FrameFeatures({k: math.nan for k in FEATURES}, visibility, side, flags)
    unit = scale if scale else face
    shoulders = xy[[L.LEFT_SHOULDER, L.RIGHT_SHOULDER]]

    v: dict[str, float] = {}
    facing = 1.0 if nose[0] >= ear[0] else -1.0
    v["head_y"] = (ear[1] + nose[1]) / 2 / unit
    v["shoulder_y"] = float(shoulders[:, 1].mean()) / unit
    v["ear_forward"] = facing * (ear[0] - sh[0]) / unit
    v["neck_ratio"] = (sh[1] - ear[1]) / unit
    v["face_scale"] = math.log(face / unit)
    v["head_pitch"] = (nose[1] - ear[1]) / face
    v["head_turn"] = (nose[0] - xy[s["eye"]][0]) / unit

    d_l = float(np.linalg.norm(nose - xy[L.LEFT_EAR]))
    d_r = float(np.linalg.norm(nose - xy[L.RIGHT_EAR]))
    v["yaw"] = (d_l - d_r) / max(d_l + d_r, 1e-9)
    v["shoulder_width"] = float(np.linalg.norm(shoulders[0] - shoulders[1])) / unit

    if lm.world is not None:
        v["world_neck_deg"] = _angle_from_vertical(lm.world[s["ear"]] - lm.world[s["shoulder"]])
    else:
        v["world_neck_deg"] = math.nan

    hips = [L.LEFT_HIP, L.RIGHT_HIP]
    if vis[hips].min() >= hip_visibility:
        v["torso_deg"] = _angle_from_vertical(shoulders.mean(0) - xy[hips].mean(0))
    else:
        v["torso_deg"] = math.nan

    return FrameFeatures(v, visibility, side, flags)
