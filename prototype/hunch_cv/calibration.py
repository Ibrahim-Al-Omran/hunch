from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import FEATURES, choose_side, compute_features, face_size
from .landmarks import Landmarks

# Minimum per-feature spread, so a very still calibration is not hypersensitive.
# Position features are in calibrated face-size units.
SPREAD_FLOOR = {
    "head_y": 0.05,
    "shoulder_y": 0.05,
    "ear_forward": 0.05,
    "neck_ratio": 0.05,
    "face_scale": 0.03,
    "head_pitch": 0.05,
    "head_turn": 0.03,
    "yaw": 0.03,
    "shoulder_width": 0.05,
    "world_neck_deg": 2.0,
    "torso_deg": 2.0,
}


class CalibrationError(ValueError):
    pass


@dataclass
class Baseline:
    side: str
    median: dict[str, float]
    spread: dict[str, float]
    samples: int
    scale_px: float | None = None  # calibrated face size, nose to near ear
    slouch_direction: dict[str, float] | None = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Baseline:
        return cls(**json.loads(path.read_text()))


def _usable(lms: list[Landmarks | None], side: str, min_visibility: float, scale: float | None):
    out = []
    for lm in lms:
        if lm is None:
            continue
        ff = compute_features(lm, side, scale)
        if ff.visibility >= min_visibility and not ff.flags:
            out.append(ff)
    return out


def calibrate(lms: list[Landmarks | None], min_visibility: float = 0.5, min_samples: int = 5,
              min_fraction: float = 0.6, side: str | None = None, meta: dict | None = None) -> Baseline:
    present = [lm for lm in lms if lm is not None]
    if not present:
        raise CalibrationError("no person detected during calibration")
    side = side or choose_side(present)
    sizes = [face_size(lm, side) for lm in present]
    scale = float(np.median(sizes))
    ffs = _usable(lms, side, min_visibility, scale)
    if len(ffs) < min_samples or len(ffs) < min_fraction * len(lms):
        raise CalibrationError(
            f"only {len(ffs)}/{len(lms)} calibration frames had the head and {side} shoulder clearly visible"
        )
    median, spread = {}, {}
    for name in FEATURES:
        vals = np.array([f.get(name) for f in ffs], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_samples:
            median[name], spread[name] = math.nan, math.nan
            continue
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        median[name] = med
        spread[name] = max(mad, SPREAD_FLOOR[name])
    return Baseline(side, median, spread, len(ffs), scale_px=scale, meta=meta or {})


def learn_slouch_direction(baseline: Baseline, lms: list[Landmarks | None], features: tuple[str, ...],
                           min_visibility: float = 0.5) -> dict[str, float]:
    """Unit direction (in baseline-spread units) from upright to the user's own slouch sample."""
    ffs = _usable(lms, baseline.side, min_visibility, baseline.scale_px)
    if len(ffs) < 3:
        raise CalibrationError("slouch reference had too few clearly visible frames")
    d = {}
    for name in features:
        vals = np.array([f.get(name) for f in ffs])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0 or not math.isfinite(baseline.median.get(name, math.nan)):
            continue
        d[name] = (float(np.median(vals)) - baseline.median[name]) / baseline.spread[name]
    norm = math.sqrt(sum(x * x for x in d.values()))
    if norm < 2.0:
        raise CalibrationError("slouch reference is indistinguishable from upright")
    return {k: x / norm for k, x in d.items()}
