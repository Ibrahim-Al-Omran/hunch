import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hunch_cv.calibration import calibrate  # noqa: E402
from hunch_cv.monitor import (AWAY, LEFT, RECALIBRATE, REMINDER, RETURNED, SETTLING, SUSTAINED,  # noqa: E402
                              MonitorConfig, PostureMonitor)
from test_posture import pose, upright_set  # noqa: E402

DT = 2.0
SLOUCH = {"head_dx": 6, "head_dy": 14, "shoulder_dy": 4}


def run(monitor, script, t0=0.0, seed=0):
    """script: list of (seconds, pose kwargs or None for nobody in view)."""
    rng = np.random.default_rng(seed)
    t, events, statuses = t0, [], []
    for seconds, kw in script:
        for _ in range(int(seconds / DT)):
            lm = None if kw is None else pose(rng=rng, noise=0.6, **kw)
            status, ev = monitor.update(t, lm)
            statuses.append(status)
            events += ev
            t += DT
    return events, statuses, t


def reminders(events):
    return [e for e in events if e.kind == REMINDER]


def new_monitor(**cfg):
    return PostureMonitor(calibrate(upright_set()), cfg=MonitorConfig(**cfg))


def test_sustained_slouch_reminds_once_after_30s():
    events, statuses, _ = run(new_monitor(), [(60, {}), (90, SLOUCH)])
    r = reminders(events)
    assert len(r) == 1
    # window of 3 samples (4 s to fill) + 30 s sustain, from slouch onset at t=60
    assert 60 + 30 <= r[0].t <= 60 + 30 + 3 * DT
    assert statuses[-1].state == SUSTAINED


def test_brief_slouch_and_upright_never_remind():
    events, _, _ = run(new_monitor(), [(120, {}), (20, SLOUCH), (120, {}), (20, SLOUCH), (60, {})])
    assert reminders(events) == []


def test_improving_resets_timer():
    events, _, _ = run(new_monitor(), [(20, SLOUCH), (20, {}), (20, SLOUCH), (20, {})])
    assert reminders(events) == []


def test_cooldown_and_snooze():
    m = new_monitor(cooldown_s=300)
    events, _, t = run(m, [(200, SLOUCH)])
    assert len(reminders(events)) == 1
    events, _, t = run(m, [(200, SLOUCH)], t0=t)
    assert len(reminders(events)) == 1  # 300 s after the first one
    m2 = new_monitor()
    m2.snooze(until=1000)
    events, _, _ = run(m2, [(400, SLOUCH)])
    assert reminders(events) == []


def test_short_occlusion_pauses_but_long_absence_resets():
    events, _, _ = run(new_monitor(), [(20, SLOUCH), (6, {"vis": 0.2}), (16, SLOUCH)])
    assert len(reminders(events)) == 1
    events, _, _ = run(new_monitor(), [(20, SLOUCH), (30, {"vis": 0.2}), (16, SLOUCH)])
    assert reminders(events) == []


def test_leaving_and_returning_settles_and_flags_shift():
    m = new_monitor(absence_s=60, settle_s=20)
    events, statuses, t = run(m, [(30, {}), (70, None)])
    assert [e.kind for e in events] == [LEFT]
    assert statuses[-1].state == AWAY
    events, statuses, _ = run(m, [(30, {"head_dy": 10, "shoulder_dy": 10})], t0=t)
    kinds = [e.kind for e in events]
    assert kinds[0] == RETURNED and RECALIBRATE in kinds
    assert statuses[0].state == SETTLING
    m2 = new_monitor(absence_s=60, settle_s=20)
    _, _, t = run(m2, [(30, {}), (70, None)])
    events, _, _ = run(m2, [(30, {})], t0=t)
    assert RECALIBRATE not in [e.kind for e in events]
