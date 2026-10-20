"""Temporal logic on top of per-sample scoring: sustained-slouch timer,
reminder cooldown, snooze, and leaving / returning to the desk.

Pure and clock-agnostic: callers pass timestamps, so the same code drives
live monitoring and offline replay of recorded landmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .calibration import Baseline
from .features import compute_features
from .landmarks import Landmarks
from .scoring import GOOD, INSUFFICIENT, POSSIBLE, UNCERTAIN, Assessment, ScoringConfig, WindowMedian, assess

SUSTAINED = "sustained_slouch"
AWAY = "away"
SETTLING = "settling"

REMINDER = "reminder"
LEFT = "left"
RETURNED = "returned"
RECALIBRATE = "recalibrate_suggested"


@dataclass
class MonitorConfig:
    window: int = 3
    sustain_s: float = 30.0
    cooldown_s: float = 300.0
    # Once the timer runs, posture must drop below threshold * release_ratio to reset it.
    release_ratio: float = 0.8
    # Unscorable stretches longer than this reset the timer instead of pausing it.
    max_gap_s: float = 10.0
    absence_s: float = 60.0
    settle_s: float = 20.0
    # Median head/shoulder deviation after returning, in baseline spread units,
    # beyond which recalibration is suggested.
    reseat_tolerance: float = 4.0


@dataclass
class Event:
    t: float
    kind: str
    detail: dict = field(default_factory=dict)


@dataclass
class Status:
    t: float
    state: str
    score: float
    window: float
    slouch_s: float
    assessment: Assessment | None = None


class PostureMonitor:
    def __init__(self, baseline: Baseline, scoring: ScoringConfig | None = None,
                 cfg: MonitorConfig | None = None, start_away: bool = False):
        """`start_away` treats the first person seen as just sitting down (settle + shift check)."""
        self.baseline = baseline
        self.scoring = scoring or ScoringConfig()
        self.cfg = cfg or MonitorConfig()
        self._window = WindowMedian(self.cfg.window)
        self._slouch_start: float | None = None
        self._last_scored: float | None = None
        self._last_seen: float | None = None
        self._last_reminder = -math.inf
        self._snooze_until = -math.inf
        self._away = start_away
        self._settle_until: float | None = None
        self._settle_devs: list[tuple[float, float]] = []

    def snooze(self, until: float) -> None:
        self._snooze_until = until

    def reset_baseline(self, baseline: Baseline) -> None:
        self.baseline = baseline
        self._window = WindowMedian(self.cfg.window)
        self._slouch_start = None
        self._settle_until = None

    def update(self, t: float, lm: Landmarks | None) -> tuple[Status, list[Event]]:
        events: list[Event] = []
        ff = compute_features(lm, self.baseline.side, self.baseline.scale_px) if lm is not None else None
        a = assess(ff, self.baseline, self.scoring)

        if lm is not None:
            if self._away:
                self._away = False
                self._settle_until = t + self.cfg.settle_s
                self._settle_devs = []
                events.append(Event(t, RETURNED))
            self._last_seen = t
        elif self._last_seen is None:
            self._last_seen = t
        if not self._away and lm is None and t - self._last_seen >= self.cfg.absence_s:
            self._away = True
            self._slouch_start = None
            self._window = WindowMedian(self.cfg.window)
            events.append(Event(t, LEFT))
        if self._away:
            return Status(t, AWAY, math.nan, math.nan, 0.0, a), events

        win = self._window.push(a)

        if self._settle_until is not None:
            if a.state in (GOOD, POSSIBLE):
                d = a.deviations
                self._settle_devs.append((d.get("head_y", math.nan), d.get("shoulder_y", math.nan)))
            if t < self._settle_until:
                return Status(t, SETTLING, a.score, win, 0.0, a), events
            self._settle_until = None
            if self._settle_devs:
                med = np.nanmedian(np.array(self._settle_devs), axis=0)
                if np.nanmax(np.abs(med)) > self.cfg.reseat_tolerance:
                    events.append(Event(t, RECALIBRATE, {"head_y": float(med[0]), "shoulder_y": float(med[1])}))

        thr = self.scoring.threshold
        if math.isfinite(win):
            self._last_scored = t
            if win >= thr:
                if self._slouch_start is None:
                    self._slouch_start = t
            elif win < thr * self.cfg.release_ratio:
                self._slouch_start = None
        elif self._slouch_start is not None and self._last_scored is not None \
                and t - self._last_scored > self.cfg.max_gap_s:
            self._slouch_start = None

        slouch_s = t - self._slouch_start if self._slouch_start is not None else 0.0
        if self._slouch_start is not None and slouch_s >= self.cfg.sustain_s:
            state = SUSTAINED
            if t - self._last_reminder >= self.cfg.cooldown_s and t >= self._snooze_until:
                self._last_reminder = t
                events.append(Event(t, REMINDER, {"slouch_s": slouch_s, "window": win}))
        elif self._slouch_start is not None:
            state = POSSIBLE
        elif not math.isfinite(win):
            state = INSUFFICIENT if a.state == INSUFFICIENT else UNCERTAIN
        else:
            state = GOOD
        return Status(t, state, a.score, win, slouch_s, a), events
