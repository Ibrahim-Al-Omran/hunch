"""Summarise passive logs from tools/passive.py by replaying them through the
monitor, so thresholds and timings can be changed without re-recording.

    python tools/analyze_passive.py                      # all logs in data/passive
    python tools/analyze_passive.py data/passive/X.jsonl --threshold 3

Writes data/reports/passive-<stamp>/report.md plus one timeline plot per log.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv.calibration import Baseline  # noqa: E402
from hunch_cv.landmarks import Landmarks  # noqa: E402
from hunch_cv.monitor import AWAY, REMINDER, SUSTAINED, MonitorConfig, PostureMonitor  # noqa: E402
from hunch_cv.scoring import GOOD, POSSIBLE, ScoringConfig  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SWEEP = (2.0, 2.5, 3.0, 3.5, 4.0)
# A "slouching" label counts as caught by a reminder in [label - before, label + after].
CATCH_BEFORE_S, CATCH_AFTER_S = 300.0, 60.0
# A "wrong reminder" label refers to the latest reminder at most this long before it.
WRONG_WINDOW_S = 180.0


@dataclass
class PassiveLog:
    path: Path
    header: dict
    # (t, kind, payload) in file order: sample / calibrated / paused / resumed / label / event
    items: list[tuple[float, str, dict]] = field(default_factory=list)

    @property
    def interval(self) -> float:
        return float(self.header.get("interval", 2.0))


def load(path: Path) -> PassiveLog:
    header, items = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["type"] == "header":
            header = r
        else:
            items.append((r["t"], r["type"], r))
    return PassiveLog(path, header, items)


@dataclass
class Replay:
    t: np.ndarray
    state: list[str]
    window: np.ndarray
    present: np.ndarray
    ms: np.ndarray
    reminders: list[float]
    labels: list[tuple[float, str]]
    live_reminders: int
    paused_s: float


def replay(log: PassiveLog, scoring: ScoringConfig, mcfg: MonitorConfig) -> Replay | None:
    w, h = log.header.get("size", [320, 240])
    base = log.header.get("baseline")
    monitor = PostureMonitor(Baseline(**base), scoring, mcfg, start_away=True) if base else None
    t, state, window, present, ms, reminders, labels = [], [], [], [], [], [], []
    live, paused_s, paused_at = 0, 0.0, None
    for ts, kind, r in log.items:
        if kind == "calibrated":
            b = Baseline(**r["baseline"])
            if monitor is None:
                monitor = PostureMonitor(b, scoring, mcfg, start_away=True)
            else:
                monitor.reset_baseline(b)
        elif kind == "paused":
            paused_at = ts
        elif kind == "resumed" and monitor is not None:
            paused_s += ts - (paused_at or ts)
            paused_at = None
            monitor = PostureMonitor(monitor.baseline, scoring, mcfg, start_away=True)
        elif kind == "label":
            labels.append((ts, r["label"]))
        elif kind == "event" and r.get("kind") == REMINDER:
            live += 1
        elif kind == "sample" and monitor is not None:
            lm = Landmarks.from_json(r, w, h) if "lm" in r else None
            status, events = monitor.update(ts, lm)
            t.append(ts)
            state.append(status.state)
            window.append(status.window)
            present.append(lm is not None)
            ms.append(r.get("ms", math.nan))
            reminders += [e.t for e in events if e.kind == REMINDER]
    if not t:
        return None
    return Replay(np.array(t), state, np.array(window, dtype=float), np.array(present), np.array(ms, dtype=float),
                  reminders, labels, live, paused_s)


def nearest_state(rp: Replay, ts: float, tol: float) -> str | None:
    i = int(np.argmin(np.abs(rp.t - ts)))
    return rp.state[i] if abs(rp.t[i] - ts) <= tol else None


def label_stats(rp: Replay, interval: float) -> dict:
    slouch = [ts for ts, lab in rp.labels if lab == "slouching"]
    good = [ts for ts, lab in rp.labels if lab == "good"]
    wrong = [ts for ts, lab in rp.labels if lab == "wrong_reminder"]
    rem = np.array(rp.reminders)
    caught = sum(bool(np.any((rem >= ts - CATCH_BEFORE_S) & (rem <= ts + CATCH_AFTER_S))) for ts in slouch)
    slouch_flagged = sum(nearest_state(rp, ts, 2 * interval) in (POSSIBLE, SUSTAINED) for ts in slouch)
    good_states = [nearest_state(rp, ts, 2 * interval) for ts in good]
    good_flagged = sum(s in (POSSIBLE, SUSTAINED) for s in good_states)
    good_ok = sum(s == GOOD for s in good_states)
    wrong_matched = sum(bool(np.any((rem <= ts) & (rem >= ts - WRONG_WINDOW_S))) for ts in wrong)
    return {"slouch": len(slouch), "caught": caught, "slouch_flagged": slouch_flagged,
            "good": len(good), "good_ok": good_ok, "good_flagged": good_flagged,
            "wrong": len(wrong), "wrong_matched": wrong_matched}


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def plot(rp: Replay, thr: float, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minutes = (rp.t - rp.t[0]) / 60
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(minutes, np.clip(rp.window, None, 15), lw=0.8, color="tab:blue", label="window score")
    ax.axhline(thr, color="tab:red", ls="--", lw=0.8, label=f"threshold {thr:g}")
    away = np.array([s == AWAY for s in rp.state])
    ax.fill_between(minutes, 0, 15, where=away, color="0.85", step="mid", label="away")
    for i, r in enumerate(rp.reminders):
        ax.axvline((r - rp.t[0]) / 60, color="tab:red", lw=1.5, label="reminder" if i == 0 else None)
    marks = {"slouching": ("v", "tab:orange"), "good": ("^", "tab:green"), "wrong_reminder": ("x", "black")}
    seen = set()
    for ts, lab in rp.labels:
        m, c = marks.get(lab, ("o", "gray"))
        ax.scatter([(ts - rp.t[0]) / 60], [14], marker=m, color=c, s=60, zorder=5,
                   label=f"label: {lab}" if lab not in seen else None)
        seen.add(lab)
    ax.set_xlabel("minutes")
    ax.set_ylim(0, 15)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--threshold", type=float, default=ScoringConfig.threshold)
    ap.add_argument("--sustain", type=float, default=MonitorConfig.sustain_s)
    ap.add_argument("--cooldown", type=float, default=MonitorConfig.cooldown_s)
    ap.add_argument("--min-minutes", type=float, default=1.0, help="skip shorter logs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    paths = args.logs or sorted((DATA / "passive").glob("*.jsonl"))
    out = args.out or DATA / "reports" / f"passive-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    mcfg = MonitorConfig(sustain_s=args.sustain, cooldown_s=args.cooldown)
    scoring = ScoringConfig(threshold=args.threshold)

    logs = []
    for p in paths:
        log = load(p)
        rp = replay(log, scoring, mcfg)
        if rp is None or (rp.t[-1] - rp.t[0]) / 60 < args.min_minutes:
            continue
        logs.append((log, rp))

    md = [f"# Passive logs ({len(logs)})", "",
          f"Replayed with threshold {args.threshold:g}, sustain {args.sustain:g} s, cooldown {args.cooldown:g} s. "
          "Labels are self-reported via hotkeys, so they are sparse and biased towards moments you noticed.", "",
          "| log | span (h) | paused (min) | present | scorable when present | good | possible | sustained "
          "| reminders | per present hour | live reminders | inference ms (median) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot_present_h, tot_rem = 0.0, 0
    agg = {k: 0 for k in ("slouch", "caught", "slouch_flagged", "good", "good_ok", "good_flagged", "wrong", "wrong_matched")}
    for log, rp in logs:
        iv = log.interval
        span_h = (rp.t[-1] - rp.t[0]) / 3600
        present_h = rp.present.sum() * iv / 3600
        pres_states = [s for s, p in zip(rp.state, rp.present) if p]
        n = len(pres_states) or 1
        share = {k: sum(s == k for s in pres_states) / n for k in (GOOD, POSSIBLE, SUSTAINED)}
        scorable = np.isfinite(rp.window[rp.present]).mean() if rp.present.any() else math.nan
        md.append(f"| {log.path.stem} | {span_h:.2f} | {rp.paused_s / 60:.0f} | {rp.present.mean():.0%} | {scorable:.0%} "
                  f"| {share[GOOD]:.0%} | {share[POSSIBLE]:.0%} | {share[SUSTAINED]:.0%} | {len(rp.reminders)} "
                  f"| {len(rp.reminders) / present_h:.1f} | {rp.live_reminders} | {np.nanmedian(rp.ms):.0f} |"
                  if present_h else f"| {log.path.stem} | {span_h:.2f} | {rp.paused_s / 60:.0f} | 0% | | | | | 0 | | {rp.live_reminders} | |")
        tot_present_h += present_h
        tot_rem += len(rp.reminders)
        for k, v in label_stats(rp, iv).items():
            agg[k] += v
        plot(rp, args.threshold, out / f"{log.path.stem}.png", f"{log.path.stem} (threshold {args.threshold:g})")
    md += ["", f"Total: {tot_present_h:.1f} h present, {tot_rem} reminders "
               f"({tot_rem / tot_present_h:.1f} per present hour)." if tot_present_h else "No time present.", ""]

    md += ["## Labels", "",
           "| label | count | agreement |", "|---|---|---|",
           f"| slouching | {agg['slouch']} | reminder within {CATCH_BEFORE_S / 60:.0f} min before to "
           f"{CATCH_AFTER_S:.0f} s after: {pct(agg['caught'], agg['slouch'])}; flagged at that moment: "
           f"{pct(agg['slouch_flagged'], agg['slouch'])} |",
           f"| good | {agg['good']} | rated good: {pct(agg['good_ok'], agg['good'])}; flagged: "
           f"{pct(agg['good_flagged'], agg['good'])} |",
           f"| wrong reminder | {agg['wrong']} | matched to a replayed reminder: {pct(agg['wrong_matched'], agg['wrong'])} "
           f"(share of all reminders marked wrong: {pct(agg['wrong'], tot_rem)}) |", ""]

    md += ["## Threshold sweep", "", "| threshold | reminders per present hour | slouching labels caught "
           "| good labels flagged |", "|---|---|---|---|"]
    for thr in SWEEP:
        sc = ScoringConfig(threshold=thr)
        rem, ph, a = 0, 0.0, {k: 0 for k in agg}
        for log, _ in logs:
            rp = replay(log, sc, mcfg)
            rem += len(rp.reminders)
            ph += rp.present.sum() * log.interval / 3600
            for k, v in label_stats(rp, log.interval).items():
                a[k] += v
        md.append(f"| {thr:g} | {rem / ph:.1f} | {pct(a['caught'], a['slouch'])} | {pct(a['good_flagged'], a['good'])} |"
                  if ph else f"| {thr:g} | n/a | | |")
    md += ["", "## Timelines", ""] + [f"![{log.path.stem}]({log.path.stem}.png)" for log, _ in logs]

    report = out / "report.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return report


if __name__ == "__main__":
    main()
