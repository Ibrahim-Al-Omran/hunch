# Phase 1 prototype: posture detection feasibility

A throwaway validation harness, not the shipped app. It answers one question:
can a single angled (about 10 o'clock), wide-angle webcam reliably separate
upright from slouched posture? Only landmark coordinates are recorded; camera
frames stay in memory unless `--save-frames` is passed explicitly.

```
hunch_cv/      pose backends, features, calibration, scoring (pure, portable logic)
tools/         record.py, live.py, analyze.py, bench.py, model tooling
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
(6 s windows). Thresholds are picked on the other sessions (leave-one-out) to
keep upright windows flagged at 5% or less; a single session only gives
optimistic in-session numbers.

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
`shoulder_width`, `torso_deg`. `yaw` marks a sample *uncertain* when it moves
more than 0.3 from baseline; it varies about 0.2 while upright because the
far ear is guessed. `head_pitch` flags looking down without gating.

Calibration stores the per-feature median and a robust spread (MAD with a
floor). The rule score is the weighted mean of deviations in the "worse"
direction, in units of that spread. A 6 s window is the median of its
scorable samples and needs at least 2 of 3. The optional `learned` score
projects onto the direction from the upright baseline to the user's own slouch
reference; on s1 it was no better than the rule score.

Head and shoulder *height in the frame* carry most of the signal. That makes
the score depend on chair height and camera position, so recalibrate after
changing either. The re-seating comparison across sessions measures how much
normal re-sitting costs.

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

- **One session so far (s1).** Thresholds and features were tuned on it, so its
  numbers are optimistic: Lite at 320×240 flagged 100% of forward-head and
  slouch windows, 2% upright, 3% looking down, 3% turning, 0% mild forward head.
- **Leaning in from the hips is flagged like slouching** (Lite 86% of windows,
  Full 20%). With a 30 s sustain requirement, brief leans won't remind;
  sustained leaning toward the screen will.
- Deliberate, prompted slouches are exaggerated, so a pass means
  feasibility, not real-world accuracy.
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
