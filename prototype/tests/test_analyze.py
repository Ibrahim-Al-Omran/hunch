"""End-to-end check of tools/analyze.py on synthetic recorded sessions."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import analyze  # noqa: E402
from hunch_cv.protocol import PROTOCOL  # noqa: E402
from test_posture import pose  # noqa: E402

CONDITION_POSE = {
    "A": {}, "B": {}, "K": {}, "I": {},
    "A2": {"head_dx": 9, "head_dy": 11}, "C": {"head_dx": 4, "head_dy": 3},
    "D": {"head_dx": 10, "head_dy": 8}, "E": {"head_dx": 6, "head_dy": 14, "shoulder_dy": 4},
    "F": {"head_dy": 6}, "G": {"turn": -14}, "H": {"scale": 1.1},
}


def write_session(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = [{"type": "header", "session": path.stem, "camera": 0, "notes": "synthetic"}]
    t = 0.0
    for step in PROTOCOL:
        for i in range(int(step.seconds * 5)):
            if step.code == "J":
                result = None
            else:
                lm = pose(rng=rng, noise=0.6, **CONDITION_POSE[step.code])
                result = {**lm.to_json(), "w": lm.width, "h": lm.height, "ms": 20.0}
            lines.append({"type": "sample", "t": t, "cond": step.code, "phase": "step", "step_t": i / 5,
                          "results": {"mp-lite@320": result}})
            t += 0.2
    lines.append({"type": "end", "completed": True})
    path.write_text("\n".join(json.dumps(x) for x in lines))


def test_report_on_synthetic_sessions(tmp_path):
    paths = [tmp_path / "s1.jsonl", tmp_path / "s2.jsonl"]
    for i, p in enumerate(paths):
        write_session(p, i)
    report = analyze.main([str(p) for p in paths] + ["--out", str(tmp_path / "reports")])
    text = report.read_text(encoding="utf-8")
    assert "## mp-lite@320" in text and "## Exit criteria" in text
    row = next(line for line in text.splitlines() if line.startswith("| mp-lite@320 |"))
    assert "| yes" in row, row  # the synthetic slouches are large and noise-free
    assert list(report.parent.glob("*.png"))
