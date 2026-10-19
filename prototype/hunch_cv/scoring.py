"""Interpretable posture score: deviation from the user's baseline in units of
their own calibration jitter, with gates for low visibility and turning."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .calibration import Baseline
from .features import FEATURES, FrameFeatures

GOOD = "good"
POSSIBLE = "possible_slouch"
UNCERTAIN = "uncertain"
INSUFFICIENT = "insufficient_visibility"

# feature -> (direction that means worse posture, weight)
RULE_FEATURES: dict[str, tuple[int, float]] = {
    "head_y": (+1, 1.0),
    "shoulder_y": (+1, 1.0),
    "ear_forward": (+1, 0.5),
}


@dataclass
class ScoringConfig:
    method: str = "rule"  # "rule" or "learned" (needs baseline.slouch_direction)
    threshold: float = 2.5
    min_visibility: float = 0.5
    # Absolute change in the yaw ratio. From an angled camera the far ear is
    # occluded and guessed by the model, so the ratio wanders ~0.2 while upright.
    yaw_tolerance: float = 0.3
    pitch_gate: float | None = None  # deviation units; None = flag only
    pitch_flag: float = 4.0


@dataclass
class Assessment:
    state: str
    score: float
    visibility: float
    deviations: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def deviations(ff: FrameFeatures, baseline: Baseline) -> dict[str, float]:
    out = {}
    for name in FEATURES:
        x, m, s = ff.get(name), baseline.median.get(name, math.nan), baseline.spread.get(name, math.nan)
        if math.isfinite(x) and math.isfinite(m) and math.isfinite(s) and s > 0:
            out[name] = (x - m) / s
    return out


def raw_score(dev: dict[str, float], baseline: Baseline, method: str) -> float:
    if method == "learned":
        if not baseline.slouch_direction:
            raise ValueError("learned scoring needs a slouch reference in the baseline")
        terms = [dev[k] * w for k, w in baseline.slouch_direction.items() if k in dev]
        return float(sum(terms)) if terms else math.nan
    num = den = 0.0
    for name, (direction, weight) in RULE_FEATURES.items():
        if name in dev:
            num += weight * max(0.0, direction * dev[name])
            den += weight
    return num / den if den else math.nan


def assess(ff: FrameFeatures | None, baseline: Baseline, cfg: ScoringConfig) -> Assessment:
    if ff is None:
        return Assessment(INSUFFICIENT, math.nan, 0.0, reasons=["no_person"])
    if ff.flags or ff.visibility < cfg.min_visibility:
        return Assessment(INSUFFICIENT, math.nan, ff.visibility, reasons=ff.flags or ["low_visibility"])
    dev = deviations(ff, baseline)
    score = raw_score(dev, baseline, cfg.method)
    reasons = []
    yaw0 = baseline.median.get("yaw", math.nan)
    if math.isfinite(yaw0) and abs(ff.get("yaw") - yaw0) > cfg.yaw_tolerance:
        reasons.append("turned")
    pitch_dev = dev.get("head_pitch", 0.0)
    if pitch_dev > cfg.pitch_flag:
        reasons.append("looking_down")
    gated = "turned" in reasons or (cfg.pitch_gate is not None and pitch_dev > cfg.pitch_gate)
    if gated or not math.isfinite(score):
        return Assessment(UNCERTAIN, score, ff.visibility, dev, reasons)
    state = POSSIBLE if score >= cfg.threshold else GOOD
    return Assessment(state, score, ff.visibility, dev, reasons)


def window_score(scores: list[float], min_valid: int) -> float:
    """Median of the scorable samples in a window, or nan if too few are scorable."""
    valid = [s for s in scores if math.isfinite(s)]
    return float(np.median(valid)) if len(valid) >= min_valid else math.nan


class WindowMedian:
    """Median score over the last n assessments, tolerating n - min_valid unscorable ones."""

    def __init__(self, n: int = 3, min_valid: int | None = None):
        self.n = n
        self.min_valid = n - 1 if min_valid is None else min_valid
        self._scores: deque[float] = deque(maxlen=n)

    def push(self, a: Assessment) -> float:
        self._scores.append(a.score if a.state in (GOOD, POSSIBLE) else math.nan)
        return window_score(list(self._scores), self.min_valid) if len(self._scores) == self.n else math.nan
