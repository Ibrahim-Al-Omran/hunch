# Detection prototype (Phases 1–2)

A throwaway validation harness, not the shipped app. It answers one question:
can a single angled (about 10 o'clock), wide-angle webcam reliably separate
upright from slouched posture? Only landmark coordinates are recorded; camera
frames stay in memory unless `--save-frames` is passed explicitly.

```
hunch_cv/      pose backends, features, calibration, scoring, monitor (pure, portable logic)
tools/         record.py, live.py, analyze.py, passive.py, analyze_passive.py, bench.py, model tooling
tests/         unit tests + end-to-end analysis on synthetic sessions
../spike/      Rust: native camera capture + ONNX Runtime pipeline, cost measurement
```

## Setup (Windows, Python 3.11)

```powershell
cd prototype
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python tools\fetch_assets.py          # MediaPipe models + reference photo
# ONNX models (only needed for onnx-* backends and the Rust spike):
python -m venv .venv-convert
.\.venv-convert\Scripts\pip install -r requirements-convert.txt
.\.venv-convert\Scripts\python tools\convert_models.py
.\.venv\Scripts\python -m pytest -q
```

## Running the experiment

1. Find the camera index: `python tools\live.py --camera 0` (the USB camera
   enumerated first on this machine). Press `c` to calibrate, `s` to store a
   slouch reference, then try slouching; `q` quits.
2. Record at least three sessions (different days or lighting, one after
   standing up and sitting back down):
   `python tools\record.py --camera 0 --session s1 --notes "10 o'clock, 60 cm, daylight"`
   About 7 minutes, guided on screen. Each run records `mp-lite@320`,
   `mp-full@320`, `mp-lite@640` and `onnx-lite@320` on the same frames.
3. `python tools\analyze.py` writes `data/reports/<time>/report.md` plus plots.

### Passive logging (normal work)

Prompted slouches are exaggerated, so real accuracy comes from normal use:

```powershell
.\.venv\Scripts\python tools\passive.py --camera 0     # leave it running; Ctrl+C or Ctrl+Alt+Q stops
.\.venv\Scripts\python tools\analyze_passive.py         # data/reports/passive-<time>/report.md
```

It samples once every 2 s with `mp-lite@320` and runs the monitor live.
Reminders appear as a Windows notification with a beep. It writes landmarks,
monitor events and labels to `data/passive/<start>.jsonl`, and never writes
frames. It reuses `data/baseline.json` when there is one; otherwise it
calibrates first. Global hotkeys (via `pynput`, LGPL; prototype only):

| hotkey | action |
|---|---|
| Ctrl+Alt+S | label "I am slouching now" |
| Ctrl+Alt+G | label "my posture is good now" |
| Ctrl+Alt+W | label "that reminder was wrong" |
| Ctrl+Alt+C | recalibrate (sit upright for 10 s after the beep) |
| Ctrl+Alt+P | pause / resume; the camera is released while paused |
| Ctrl+Alt+Q | quit |

`analyze_passive.py` replays the logs through the monitor, so threshold and
timings can change without new recordings (`--threshold`, `--sustain`,
`--cooldown`). It reports time present, the state mix, reminders per hour,
and how reminders line up with the labels. It also sweeps the threshold.

### Protocol

| code | condition | expected |
|---|---|---|
| A | calibration, upright (10 s) | baseline |
| A2 | calibration slouch reference (8 s) | learns the user's slouch direction |
| B, K | upright working (45 s each, K after J) | good |
| C | mild forward head | informational |
| D, E | pronounced forward head, slouch | detect |
| F, G, H, I | look down, turning, lean in, recline | no warning / uncertain |
| J | out of frame | insufficient visibility |

Analysis simulates one sample every 2 s and decides on the median of 3 samples
(6 s windows). The report scores two thresholds: the fixed default (2.5) and
one picked on the other sessions (leave-one-out) to flag at most 2.5% of
their upright windows. With only two sessions the held-out pick is unstable,
so the fixed default is the one used. The report also replays each session
through the monitor and checks applying one session's calibration to another.

**Exit criteria (agreed):** visibility ≥ 90% in upright and slouch conditions,
D+E detected in ≥ 90% of windows, ≤ 5% of upright windows flagged, holding on
held-out sessions.

## Method

The near side of the body is chosen by the model's depth estimate. MediaPipe
reports visibility near 1.0 even for the hidden ear, so visibility can't decide
it. Features are in units of the calibrated face size (median nose-to-near-ear
distance during calibration). The per-frame face size varies about 10% and
made ratio features noisy in session s1.

Scored (chosen from s1, to be confirmed on further sessions):

- `head_y`: mean height of near ear and nose in the frame (weight 1)
- `shoulder_y`: mean shoulder height in the frame (weight 1)
- `ear_forward`: near ear ahead of near shoulder (weight 0.5)

Recorded for diagnosis: `neck_ratio` (weak, because slouching from this angle
lowers head and shoulders together), `face_scale`, `world_neck_deg`,
`shoulder_width`, `torso_deg`, `yaw`. A sample is *uncertain* (turned) when
`head_turn` changes by more than 0.2 face sizes from baseline. `head_turn` is
the nose's horizontal offset from the near eye. An earlier gate on `yaw`
(the ear-distance ratio) was dropped: the far ear is guessed, so `yaw`
drifted with the working head angle and marked about 65% of upright s2
samples uncertain. `head_pitch` flags looking down without gating.

Calibration stores the per-feature median and a robust spread (MAD with a
floor). The rule score is the weighted mean of deviations in the "worse"
direction, in units of that spread. A 6 s window is the median of its
scorable samples and needs at least 2 of 3. Windows at or above **2.5** count
as slouched; this one fixed threshold is used for every session. The optional
`learned` score projects onto the direction from the upright baseline to the
user's own slouch reference; on s1 it was no better than the rule score.

Head and shoulder *height in the frame* carry most of the signal. That makes
the score depend on chair height and camera position, so recalibrate after
changing either. The re-seating comparison across sessions measures how much
normal re-sitting costs.

### Monitor (when to remind)

`hunch_cv/monitor.py` has no clock or UI of its own. The caller passes
timestamps, so the same code drives live use and offline replay.

- **Sustained slouch:** a timer starts when a window reaches the threshold.
  A reminder fires once the timer reaches 30 s. The timer resets only when the
  window drops below 0.8× the threshold, so scores hovering at the threshold
  don't restart it.
- **Unscorable samples** (turned, occluded) pause the timer for up to 10 s;
  longer gaps reset it.
- **Cooldown** of 5 minutes between reminders. Snooze suppresses reminders
  until a given time.
- **Leaving the desk:** after 60 s with nobody detected the state becomes
  *away* and the timer resets. On return the first 20 s are *settling*, with
  no reminders. If head and shoulder height then differ from calibration by
  more than 4 spreads, the monitor suggests recalibrating instead of
  reminding. Calibration is never replaced automatically.

## Measured so far (Ryzen 7 7730U, Windows 11)

Inference on the reference photo, 320×240, one sample = detector + landmarks:

| implementation | wall ms (median) | CPU ms / sample | process RSS |
|---|---|---|---|
| Python MediaPipe Lite | 26 | 26 | ~170 MB (Python + TF Lite runtime) |
| Python MediaPipe Full | 33 | 33 | |
| Rust + ONNX Runtime, Lite, 1 thread | 26 | 27 | ~80 MB |
| Rust + ONNX Runtime, Lite, 4 threads | 17 | 43 | ~80 MB |
| Rust + ONNX Runtime, Full, 1 thread | 30 | 31 | ~87 MB |

Camera plus inference every 2 s in the Rust spike (USB camera, 320×240, 30 s runs):

| mode | process CPU, % of one core | % of all 16 threads |
|---|---|---|
| camera kept open, no inference, 30 fps | 0.57 | 0.04 |
| camera kept open, no inference, native 5 fps | 0.16 | 0.01 |
| kept open + inference, 5 fps | 0.89 | 0.06 |
| kept open + inference, 30 fps | 1.30 | 0.08 |
| reopened per sample + inference | 0.99 | 0.06 (but ~0.9 s per capture) |

Notes:

- These camera runs saw nobody in frame, so only the detector ran (~13 ms);
  a frame with a person costs ~26 ms, adding about 0.65% of one core at 2 s
  sampling.
- In Python, OpenCV's capture thread decodes every frame: 3.5% of one core at
  320×240 and 11% at 640×480, and it ignores the 5 fps request. `nokhwa` in Rust
  only decodes the frames it asks for.
- Windows' camera service (FrameServer) and driver work happen outside our
  process and were not measured; that needs admin rights or ETW (Event Tracing
  for Windows), and is planned for Phase 4.

## Model and runtime decision

- **MediaPipe Pose Landmarker (BlazePose GHUM), Lite.** Apache-2.0 code and
  model card. It gives ear, eye and mouth points on both sides with per-point
  visibility, which a three-quarter view needs. Full stays the fallback if
  Lite's landmark jitter turns out too high in the recordings.
- **Shipping path: Rust + ONNX Runtime (MIT), no Python.** The `.task` models
  convert to ONNX after densifying 38 sparse constants in the detector.
  Converted models match TFLite to within 0.003 px on 256 px landmark output.
- **Python port vs MediaPipe on the same image:** median 0.05–0.15 face sizes
  apart. Rust vs Python differ by a similar amount, from resampling details
  alone. In other words, the landmark model itself moves about 0.1 face size
  under tiny input changes; treat that as the noise floor. Because baselines
  are per backend, this only matters if Rust's frame-to-frame jitter is larger
  than MediaPipe's. The `onnx-lite@320` variant in each recording measures it.
- **Excluded:** YOLO-pose (AGPL-3.0), OpenPose (non-commercial).

## Known limitations (so far)

- **Two prompted sessions so far; accuracy in normal work is untested.**
  Features were chosen on s1. With the fixed threshold 2.5 and each session
  calibrated at its start, the share of forward-head and slouch windows caught
  (and of upright windows flagged) was:

  | variant | s1 | s2 |
  |---|---|---|
  | Lite 320 | 98% (3%) | 100% (1%) |
  | ONNX Lite 320 | 100% (2%) | 100% (1%) |
  | Full 320 | 98% (3%) | 100% (2%) |

  Thresholds tuned on the other session did worse (Lite flagged 33% of
  upright s1 windows), because one session is too little to tune on. Turning
  was flagged in 0–5% of windows and reclining in 0–8%. Looking down was 0–8%
  for Lite, but Full flagged it in 34% of s2 windows. Mild forward head was 3%
  in s1 but 100% in s2, so it's inconsistent.
- **Monitor replay:** each recording replayed through the monitor gives
  exactly one reminder, during the D+E slouch block, for every variant. Upright
  typing, looking down, turning, leaning in and reclining (30 s or shorter)
  gave none.
- **Calibration carries over between these two sittings now.** Applying one
  session's calibration to the other caught 97–100% of slouch windows, with
  1–2% of upright windows flagged (Lite, ONNX and Full at 320, threshold 2.5). An earlier
  "does not carry over" finding (35%) came from the `yaw` gate marking most
  of s2 uncertain, not from the posture features. Two sittings at the same
  desk prove little, so the monitor still checks the position after each
  return and suggests recalibrating.
- **Leaning in from the hips is flagged like slouching** (Lite 95% / 40% of
  windows in s1 / s2). Brief leans won't remind, because the slouch must last
  30 s. Sustained leaning toward the screen will remind. Whether that is
  wanted is still an open product decision.
- Deliberate, prompted slouches are exaggerated, so a pass means
  feasibility, not real-world accuracy. Passive logging is meant to
  measure it.
- Looking down (F) and slouching share signals; `head_pitch` flags it, but
  whether it should gate warnings is decided from the data (`--pitch-gate`).
- Leaning in from the hips (H) moves the head toward the camera much like
  forward-head posture and is expected to be the hardest confounder.
- 3D world landmarks come from a single camera and are the model's guess,
  not a measurement; they are kept only if they prove stable.
- Lens distortion is not corrected. Analysis compares sessions; if position in
  the frame shifts features, a precomputed undistortion map is the next step.
- Running the detector on every sample doubles cost; reusing the previous
  region of interest, as MediaPipe's video mode does, is a Phase 4 option.
