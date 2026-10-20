import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_passive  # noqa: E402
from hunch_cv.calibration import calibrate  # noqa: E402
from hunch_cv.monitor import MonitorConfig  # noqa: E402
from hunch_cv.scoring import ScoringConfig  # noqa: E402
from test_posture import pose, upright_set  # noqa: E402

SLOUCH = {"head_dx": 6, "head_dy": 14, "shoulder_dy": 4}


def write_log(path: Path) -> None:
    rng = np.random.default_rng(1)
    baseline = calibrate(upright_set())
    t = 1_000_000.0
    rows = [{"type": "header", "t": t, "size": [320, 240], "interval": 2.0, "baseline": asdict(baseline)}]

    def samples(seconds, kw):
        nonlocal t
        for _ in range(int(seconds / 2)):
            r = {"type": "sample", "t": t, "ms": 20.0}
            if kw is not None:
                r.update(pose(rng=rng, noise=0.6, **kw).to_json())
            rows.append(r)
            t += 2.0

    samples(180, {})
    rows.append({"type": "label", "t": t + 1, "label": "good"})
    samples(10, {})
    samples(30, SLOUCH)
    rows.append({"type": "label", "t": t, "label": "slouching"})
    samples(60, SLOUCH)
    rows.append({"type": "label", "t": t, "label": "wrong_reminder"})
    samples(90, None)
    rows.append({"type": "paused", "t": t})
    t += 600
    rows.append({"type": "resumed", "t": t})
    samples(120, {})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_replay_and_labels(tmp_path):
    log_path = tmp_path / "log.jsonl"
    write_log(log_path)
    rp = analyze_passive.replay(analyze_passive.load(log_path), ScoringConfig(), MonitorConfig())
    assert len(rp.reminders) == 1
    assert abs(rp.paused_s - 600) < 1
    stats = analyze_passive.label_stats(rp, 2.0)
    assert stats["caught"] == 1 and stats["slouch_flagged"] == 1
    assert stats["good_ok"] == 1 and stats["good_flagged"] == 0
    assert stats["wrong_matched"] == 1


def test_report_written(tmp_path):
    log_path = tmp_path / "log.jsonl"
    write_log(log_path)
    report = analyze_passive.main([str(log_path), "--out", str(tmp_path / "out")])
    text = report.read_text()
    assert "| log |" in text and "Threshold sweep" in text
    assert (tmp_path / "out" / "log.png").exists()
