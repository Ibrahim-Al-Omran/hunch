"""Offline analysis of recorded sessions -> markdown report + plots.

    python tools/analyze.py                      # all sessions in data/sessions
    python tools/analyze.py data/sessions/s1.jsonl data/sessions/s2.jsonl

Sampling every --interval seconds is simulated from the denser recording, and
decisions use the median of --window consecutive samples. Thresholds are
picked on other sessions (leave-one-session-out) when more than one session
exists; a single session can only give optimistic in-session numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv import protocol as P  # noqa: E402
from hunch_cv.calibration import Baseline, CalibrationError, calibrate, learn_slouch_direction  # noqa: E402
from hunch_cv.features import FEATURES, FrameFeatures, compute_features  # noqa: E402
from hunch_cv.landmarks import LEFT_EAR, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_SHOULDER, Landmarks  # noqa: E402
from hunch_cv.scoring import (GOOD, INSUFFICIENT, POSSIBLE, RULE_FEATURES, UNCERTAIN, Assessment,  # noqa: E402
                              ScoringConfig, assess, window_score)

DATA = Path(__file__).resolve().parents[1] / "data"
EXIT = {"visibility": 0.90, "detection": 0.90, "upright_fp": 0.05}
# Thresholds aim below the 5% target on training sessions so held-out sessions have margin.
TRAIN_FP = 0.025


@dataclass
class Row:
    t: float
    cond: str
    phase: str
    step_t: float
    lm: Landmarks | None
    ms: float | None
    ff: FrameFeatures | None = None
    a: Assessment | None = None


@dataclass
class Session:
    name: str
    header: dict
    rows: dict[str, list[Row]]  # variant -> rows
    completed: bool


def load(path: Path) -> Session:
    header, rows, completed = {}, defaultdict(list), False
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["type"] == "header":
            header = d
        elif d["type"] == "end":
            completed = d["completed"]
        elif d["type"] == "sample":
            for key, r in d["results"].items():
                lm = Landmarks.from_json(r, r["w"], r["h"]) if r else None
                rows[key].append(Row(d["t"], d["cond"], d["phase"], d["step_t"], lm, r["ms"] if r else None))
    return Session(header.get("session", path.stem), header, dict(rows), completed)


def step_rows(rows: list[Row], cond: str) -> list[Row]:
    return [r for r in rows if r.cond == cond and r.phase == "step"]


def make_baseline(rows: list[Row]) -> Baseline:
    b = calibrate([r.lm for r in step_rows(rows, P.CALIBRATION)])
    try:
        b.slouch_direction = learn_slouch_direction(b, [r.lm for r in step_rows(rows, P.SLOUCH_REFERENCE)],
                                                    tuple(RULE_FEATURES))
    except CalibrationError:
        b.slouch_direction = None
    return b


def score_rows(rows: list[Row], baseline: Baseline, cfg: ScoringConfig) -> list[Row]:
    out = []
    for r in rows:
        ff = compute_features(r.lm, baseline.side, baseline.scale_px) if r.lm is not None else None
        out.append(Row(r.t, r.cond, r.phase, r.step_t, r.lm, r.ms, ff, assess(ff, baseline, cfg)))
    return out


def windows(rows: list[Row], interval: float, n: int, offsets: int = 5) -> dict[str, list[tuple[float, float]]]:
    """Per condition: (time since step start, window median score or nan)."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_cond: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.phase == "step":
            by_cond[r.cond].append(r)
    for cond, rs in by_cond.items():
        ts = np.array([r.step_t for r in rs])
        end = ts.max() if len(ts) else 0
        for off in np.arange(offsets) * interval / offsets:
            picks = []
            for target in np.arange(off, end + 1e-9, interval):
                i = int(np.argmin(np.abs(ts - target)))
                ok = abs(ts[i] - target) <= 0.3 and rs[i].a.state in (GOOD, POSSIBLE)
                picks.append((target, rs[i].a.score if ok else math.nan))
            for j in range(n - 1, len(picks)):
                vals = [p[1] for p in picks[j - n + 1:j + 1]]
                out[cond].append((picks[j][0], window_score(vals, n - 1)))
    return out


def flagged_rate(ws: dict, conds, thr: float) -> float:
    vals = [v for c in conds for _, v in ws.get(c, [])]
    return float(np.mean([math.isfinite(v) and v >= thr for v in vals])) if vals else math.nan


def threshold_for_fp(ws_list: list[dict], target_fp: float) -> float:
    vals = np.array([v for ws in ws_list for c in P.UPRIGHT for _, v in ws.get(c, []) if math.isfinite(v)])
    if len(vals) == 0:
        return math.nan
    # smallest threshold whose flagged fraction of upright windows stays within target
    return float(np.quantile(vals, 1 - target_fp)) + 1e-6


def time_to_detect(ws: dict, conds, thr: float) -> float:
    firsts = []
    for c in conds:
        hits = sorted(t for t, v in ws.get(c, []) if math.isfinite(v) and v >= thr)
        firsts.append(hits[0] if hits else math.nan)
    return float(np.nanmedian(firsts)) if any(map(math.isfinite, firsts)) else math.nan


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    if len(neg) == 0 or len(pos) == 0:
        return math.nan
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def cohen_d(neg: np.ndarray, pos: np.ndarray) -> float:
    if len(neg) < 2 or len(pos) < 2:
        return math.nan
    s = math.sqrt((neg.var(ddof=1) + pos.var(ddof=1)) / 2)
    return float((pos.mean() - neg.mean()) / s) if s > 0 else math.nan


def devs(rows: list[Row], conds, feature: str) -> np.ndarray:
    return np.array([r.a.deviations[feature] for r in rows
                     if r.cond in conds and r.phase == "step" and r.a and feature in r.a.deviations])


def visible_fraction(rows: list[Row], conds, min_vis: float) -> float:
    rs = [r for r in rows if r.cond in conds and r.phase == "step"]
    return float(np.mean([r.ff is not None and r.ff.visibility >= min_vis for r in rs])) if rs else math.nan


def state_mix(rows: list[Row], cond: str) -> str:
    rs = step_rows(rows, cond)
    if not rs:
        return "-"
    counts = defaultdict(int)
    for r in rs:
        counts[r.a.state] += 1
    order = [(GOOD, "good"), (POSSIBLE, "slouch"), (UNCERTAIN, "uncert"), (INSUFFICIENT, "insuff")]
    return " / ".join(f"{100 * counts[s] / len(rs):.0f}" for s, _ in order)


def backend_agreement(sess: Session, a_key: str, b_key: str) -> dict | None:
    if a_key not in sess.rows or b_key not in sess.rows:
        return None
    errs, both = [], 0
    for ra, rb in zip(sess.rows[a_key], sess.rows[b_key]):
        if ra.lm is None or rb.lm is None:
            continue
        both += 1
        face = float(np.linalg.norm(ra.lm.xy[NOSE] - ra.lm.xy[LEFT_EAR]) + np.linalg.norm(ra.lm.xy[NOSE] - ra.lm.xy[RIGHT_EAR])) / 2
        idx = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER]
        d = np.linalg.norm(ra.lm.xy[idx] - rb.lm.xy[idx], axis=1)
        errs.append((d, d / max(face, 1e-6)))
    if not errs:
        return None
    px = np.stack([e[0] for e in errs])
    rel = np.stack([e[1] for e in errs])
    det_a = sum(r.lm is not None for r in sess.rows[a_key])
    det_b = sum(r.lm is not None for r in sess.rows[b_key])
    return {"frames": both, "detected": (det_a, det_b), "median_px": float(np.median(px)),
            "p95_px": float(np.quantile(px, 0.95)), "median_rel_face": float(np.median(rel))}


def fmt(x: float, pct: bool = False, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{100 * x:.0f}%" if pct else f"{x:.{digits}f}"


def plot_session(path: Path, rows: list[Row], thr: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.6))
    cmap = {**{c: "#d8f0d8" for c in P.UPRIGHT}, **{c: "#f8d0c8" for c in P.SLOUCH}, "C": "#fbe7c6",
            **{c: "#d8e4f8" for c in P.CONFOUNDERS}, "J": "#e4e4e4", "A": "#eeeeee", "A2": "#f3dcdc"}
    spans: dict[str, list[float]] = {}
    for r in rows:
        if r.phase == "step":
            spans.setdefault(r.cond, [r.t, r.t])[1] = r.t
    for c, (t0, t1) in spans.items():
        ax.axvspan(t0, t1, color=cmap.get(c, "#ffffff"), lw=0)
        ax.text((t0 + t1) / 2, 0.98, c, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ok = [r for r in rows if r.a and r.a.state in (GOOD, POSSIBLE)]
    unc = [r for r in rows if r.a and r.a.state == UNCERTAIN and math.isfinite(r.a.score)]
    ax.plot([r.t for r in ok], [r.a.score for r in ok], ".", ms=2.5, color="#333")
    ax.plot([r.t for r in unc], [r.a.score for r in unc], "x", ms=3, color="#36c")
    if math.isfinite(thr):
        ax.axhline(thr, color="#c33", lw=1, ls="--")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("seconds")
    ax.set_ylabel("posture score")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--method", default="rule", choices=["rule", "learned"])
    ap.add_argument("--pitch-gate", type=float, default=None)
    ap.add_argument("--yaw-tolerance", type=float, default=ScoringConfig.yaw_tolerance,
                    help="turning gate; use a large value to disable")
    ap.add_argument("--out", type=Path, default=DATA / "reports")
    ap.add_argument("--rule", help="override scored features, e.g. head_y:1,shoulder_y:1 (direction from defaults or +1)")
    ap.add_argument("--variants", help="comma-separated subset of recorded variants")
    args = ap.parse_args(argv)
    if args.rule:
        rule = {}
        for item in args.rule.split(","):
            name, weight = item.split(":")
            rule[name] = (RULE_FEATURES.get(name, (1, 0))[0], float(weight))
        RULE_FEATURES.clear()
        RULE_FEATURES.update(rule)

    paths = args.sessions or sorted((DATA / "sessions").glob("*.jsonl"))
    if not paths:
        sys.exit("no sessions found; record one with tools/record.py")
    sessions = [load(p) for p in paths]
    variants = sorted({v for s in sessions for v in s.rows}, key=lambda v: (v.split("@")[1], v))
    if args.variants:
        variants = [v for v in variants if v in args.variants.split(",")]
    cfg = ScoringConfig(method=args.method, pitch_gate=args.pitch_gate, yaw_tolerance=args.yaw_tolerance)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    md = [f"# Phase 1 analysis ({stamp})", "",
          f"Sessions: {', '.join(s.name + ('' if s.completed else ' (incomplete)') for s in sessions)}  ",
          f"Simulated sampling every {args.interval:g} s, decisions on the median of {args.window} samples "
          f"({args.interval * args.window:g} s windows, at least {args.window - 1} scorable). Scoring method: `{args.method}`, pitch gate: {args.pitch_gate}, yaw tolerance: {args.yaw_tolerance:g}.  ",
          "Scored features: " + ", ".join(f"{k} (weight {w:g})" for k, (_, w) in RULE_FEATURES.items()), ""]
    for s in sessions:
        md.append(f"- **{s.name}**: camera {s.header.get('camera')}, notes: {s.header.get('notes') or '-'}")
    md.append("")

    summary = []
    for v in variants:
        md += [f"## {v}", ""]
        scored: dict[str, list[Row]] = {}
        baselines: dict[str, Baseline] = {}
        for s in sessions:
            if v not in s.rows:
                continue
            try:
                baselines[s.name] = make_baseline(s.rows[v])
            except CalibrationError as e:
                md.append(f"- {s.name}: calibration failed: {e}")
                continue
            scored[s.name] = score_rows(s.rows[v], baselines[s.name], cfg)
        if not scored:
            md.append("")
            continue
        ws = {n: windows(rows, args.interval, args.window) for n, rows in scored.items()}

        md += ["### Visibility and states", "",
               "Visible = weakest of nose / near eye / near ear / near shoulder has visibility >= 0.5. "
               "State mix per condition = good / possible slouch / uncertain / insufficient (% of samples).", "",
               "| session | side | calib n | visible (upright+slouch) | " + " | ".join(P.NAMES[c] for c in "BCDEFGHIJK") + " |",
               "|---|---|---|---|" + "---|" * 10]
        for n, rows in scored.items():
            b = baselines[n]
            md.append(f"| {n} | {b.side} | {b.samples} | {fmt(visible_fraction(rows, P.UPRIGHT + P.SLOUCH, cfg.min_visibility), True)} | "
                      + " | ".join(state_mix(rows, c) for c in "BCDEFGHIJK") + " |")
        md.append("")

        md += ["### Feature separation (upright B+K vs slouch D+E, deviations from baseline)", "",
               "AUC 0.5 = no separation, 1.0 = perfect (in the expected direction for scored features). "
               "Jitter = std of deviations while upright (about 1 means calibration captured normal variation).", "",
               "| feature | scored | AUC D+E | AUC C | Cohen d | jitter upright |", "|---|---|---|---|---|---|"]
        all_rows = [r for rows in scored.values() for r in rows]
        for f in FEATURES:
            sign = RULE_FEATURES.get(f, (1, 0))[0]
            neg, pos, mild = (sign * devs(all_rows, P.UPRIGHT, f), sign * devs(all_rows, P.SLOUCH, f),
                              sign * devs(all_rows, P.MILD, f))
            md.append(f"| {f} | {'yes' if f in RULE_FEATURES else 'gate/info'} | {fmt(auc(neg, pos))} | "
                      f"{fmt(auc(neg, mild))} | {fmt(cohen_d(neg, pos))} | {fmt(float(np.std(neg)) if len(neg) else math.nan)} |")
        md.append("")

        md += ["### Window-level detection", "",
               "Detection counts every D/E window, so windows without enough visibility count as misses. "
               "Threshold `held-out` is chosen on the other sessions to flag <= 2.5% of their upright windows (margin for the 5% target).", "",
               "| session | threshold | source | detect D+E | detect C | upright flagged | look down | turning | lean in | recline | median s to detect |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        names = list(scored)
        per_session = {}
        for n in names:
            others = [ws[m] for m in names if m != n]
            if others:
                thr, src = threshold_for_fp(others, TRAIN_FP), "held-out"
            else:
                thr, src = threshold_for_fp([ws[n]], TRAIN_FP), "in-session (optimistic)"
            for t, label in ((thr, src), (cfg.threshold, "default")):
                w = ws[n]
                row = (flagged_rate(w, P.SLOUCH, t), flagged_rate(w, P.MILD, t), flagged_rate(w, P.UPRIGHT, t),
                       *(flagged_rate(w, (c,), t) for c in P.CONFOUNDERS), time_to_detect(w, P.SLOUCH, t))
                md.append(f"| {n} | {fmt(t)} | {label} | " + " | ".join(fmt(x, True) for x in row[:-1])
                          + f" | {fmt(row[-1], digits=0)} |")
                if label == src:
                    per_session[n] = (thr, src, row)
            plot_session(out_dir / f"{v.replace('@', '_')}_{n}.png", scored[n], thr, f"{v} - {n} (threshold {src})")
        md.append("")
        md.append("Plots: " + ", ".join(f"[{n}]({v.replace('@', '_')}_{n}.png)" for n in names))
        md.append("")

        if len(names) > 1:
            md += ["### Re-seating: calibrate on one session, evaluate on another", "",
                   "| calibrated on \\ evaluated on | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for ci in names:
                cells = []
                thr = per_session[ci][0]
                for ej in names:
                    s_ej = next(s for s in sessions if s.name == ej)
                    w = windows(score_rows(s_ej.rows[v], baselines[ci], cfg), args.interval, args.window)
                    cells.append(f"detect {fmt(flagged_rate(w, P.SLOUCH, thr), True)}, upright {fmt(flagged_rate(w, P.UPRIGHT, thr), True)}")
                md.append(f"| {ci} | " + " | ".join(cells) + " |")
            md.append("")

        ms = [r.ms for s in sessions for r in s.rows.get(v, []) if r.ms is not None]
        vis = np.nanmean([visible_fraction(rows, P.UPRIGHT + P.SLOUCH, cfg.min_visibility) for rows in scored.values()])
        det = np.nanmean([per_session[n][2][0] for n in names])
        fp = np.nanmean([per_session[n][2][2] for n in names])
        summary.append((v, vis, det, fp, per_session[names[0]][1], float(np.median(ms)) if ms else math.nan,
                        float(np.quantile(ms, 0.95)) if ms else math.nan))

    md += ["## Exit criteria", "",
           f"Targets: visibility >= {EXIT['visibility']:.0%}, detection of D+E >= {EXIT['detection']:.0%}, "
           f"upright windows flagged <= {EXIT['upright_fp']:.0%} (by construction of the threshold on its own sessions; "
           "held-out sessions show whether it generalises).", "",
           "| variant | visibility | detection | upright flagged | threshold source | inference median ms | p95 ms | pass |",
           "|---|---|---|---|---|---|---|---|"]
    for v, vis, det, fp, src, m50, m95 in summary:
        ok = vis >= EXIT["visibility"] and det >= EXIT["detection"] and fp <= EXIT["upright_fp"] + 1e-9
        md.append(f"| {v} | {fmt(vis, True)} | {fmt(det, True)} | {fmt(fp, True)} | {src} | {fmt(m50, digits=1)} | "
                  f"{fmt(m95, digits=1)} | {'yes' if ok else 'no'}{'' if 'held' in src else ' (needs more sessions)'} |")
    md.append("")
    md.append("Inference times were measured while several variants ran back to back; use tools/bench.py for clean numbers.")
    md.append("")

    md += ["## ONNX port vs MediaPipe (same frames)", "",
           "| session | pair | frames | detected (mp / onnx) | median px | p95 px | median / face size |",
           "|---|---|---|---|---|---|---|"]
    for s in sessions:
        for a_key in [k for k in s.rows if k.startswith("mp-")]:
            b_key = a_key.replace("mp-", "onnx-")
            g = backend_agreement(s, a_key, b_key)
            if g:
                md.append(f"| {s.name} | {a_key} vs {b_key} | {g['frames']} | {g['detected'][0]} / {g['detected'][1]} | "
                          f"{g['median_px']:.2f} | {g['p95_px']:.2f} | {g['median_rel_face']:.3f} |")
    md.append("")

    report = out_dir / "report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {report}")
    return report


if __name__ == "__main__":
    main()
