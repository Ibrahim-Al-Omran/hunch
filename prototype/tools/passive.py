"""Passive logging while you work normally: one pose sample every few seconds,
live sustained-slouch reminders, and optional hotkey labels.

Only landmarks, monitor events and your labels are written, to
data/passive/<start>.jsonl. Frames are never saved or shown. Pausing releases
the camera.

Global hotkeys (work while other apps have focus):
    Ctrl+Alt+S  label "I am slouching now"
    Ctrl+Alt+G  label "my posture is good now"
    Ctrl+Alt+W  label "that reminder was wrong"
    Ctrl+Alt+C  recalibrate (sit upright for 10 s after the beep)
    Ctrl+Alt+P  pause / resume (camera off while paused)
    Ctrl+Alt+Q  quit  (Ctrl+C in this console also works)

    python tools/passive.py --camera 1
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv.calibration import Baseline, CalibrationError, calibrate  # noqa: E402
from hunch_cv.capture import LatestFrameReader, open_camera, to_rgb  # noqa: E402
from hunch_cv.monitor import RECALIBRATE, REMINDER, MonitorConfig, PostureMonitor  # noqa: E402
from hunch_cv.pose import make_backend  # noqa: E402
from hunch_cv.scoring import ScoringConfig  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
BASELINE = DATA / "baseline.json"
CAPTURE = (640, 480)  # same capture size as the validation recordings, then downscaled
RETRY_S = 10.0

HOTKEYS = {
    "<ctrl>+<alt>+s": "label:slouching",
    "<ctrl>+<alt>+g": "label:good",
    "<ctrl>+<alt>+w": "label:wrong_reminder",
    "<ctrl>+<alt>+c": "recalibrate",
    "<ctrl>+<alt>+p": "pause",
    "<ctrl>+<alt>+q": "quit",
}

_TOAST = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $x.GetElementsByTagName('text')
$t.Item(0).AppendChild($x.CreateTextNode('{title}')) > $null
$t.Item(1).AppendChild($x.CreateTextNode('{body}')) > $null
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}}\WindowsPowerShell\v1.0\powershell.exe').Show([Windows.UI.Notifications.ToastNotification]::new($x))
"""


def beep() -> None:
    if sys.platform == "win32":
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    else:
        print("\a", end="", flush=True)


def notify(title: str, body: str) -> None:
    beep()
    try:
        if sys.platform == "win32":
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command",
                              _TOAST.format(title=title.replace("'", "''"), body=body.replace("'", "''"))],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        elif sys.platform == "darwin":
            subprocess.Popen(["osascript", "-e", f'display notification "{body}" with title "{title}"'])
    except OSError:
        pass


def say(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields) -> None:
        self._f.write(json.dumps({"type": kind, "t": round(time.time(), 3), **fields}) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class Camera:
    """Opens lazily, survives disconnects, and is fully released on pause."""

    def __init__(self, index: int):
        self.index = index
        self.reader: LatestFrameReader | None = None
        self._next_try = 0.0

    def frame(self):
        if self.reader is not None and self.reader.failed:
            self.release()
            self._next_try = time.monotonic() + RETRY_S
            raise ConnectionError("camera stopped delivering frames")
        if self.reader is None:
            if time.monotonic() < self._next_try:
                return None
            try:
                self.reader = LatestFrameReader(open_camera(self.index, *CAPTURE))
            except RuntimeError:
                self._next_try = time.monotonic() + RETRY_S
                raise
        got = self.reader.latest()
        return got[0] if got else None

    def release(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None


def run_calibration(cam: Camera, backend, size, log: Log, seconds: float = 10.0) -> Baseline | None:
    say("calibrating: sit the way you consider upright and look at your screen...")
    beep()
    time.sleep(2.0)
    lms, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            frame = cam.frame()
        except (ConnectionError, RuntimeError) as e:
            say(f"calibration failed: {e}")
            return None
        if frame is not None:
            lms.append(backend.detect(to_rgb(frame, size)))
        time.sleep(0.25)
    try:
        baseline = calibrate(lms, meta={"backend": backend_name(backend), "size": list(size), "source": "passive"})
    except CalibrationError as e:
        say(f"calibration failed: {e}")
        log.write("calibration_failed", reason=str(e))
        return None
    baseline.save(BASELINE)
    log.write("calibrated", baseline=asdict(baseline))
    beep()
    say(f"calibrated ({baseline.samples} samples, near side {baseline.side})")
    return baseline


def backend_name(backend) -> str:
    return getattr(backend, "name", type(backend).__name__)


def start_hotkeys(q: queue.Queue):
    try:
        from pynput import keyboard
    except ImportError:
        say("pynput not installed: hotkeys disabled (pip install pynput); use Ctrl+C to quit")
        return None
    hk = keyboard.GlobalHotKeys({combo: (lambda a=action: q.put(a)) for combo, action in HOTKEYS.items()})
    hk.start()
    return hk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="mp-lite")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between pose samples")
    ap.add_argument("--minutes", type=float, default=None, help="stop automatically after this long")
    ap.add_argument("--recalibrate", action="store_true", help="ignore the saved baseline and calibrate now")
    ap.add_argument("--threshold", type=float, default=ScoringConfig.threshold)
    ap.add_argument("--sustain", type=float, default=MonitorConfig.sustain_s)
    ap.add_argument("--cooldown", type=float, default=MonitorConfig.cooldown_s)
    args = ap.parse_args()

    size = (args.width, args.height)
    backend = make_backend(args.backend)
    scoring = ScoringConfig(threshold=args.threshold)
    mcfg = MonitorConfig(sustain_s=args.sustain, cooldown_s=args.cooldown)
    log = Log(DATA / "passive" / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    cam = Camera(args.camera)
    q: queue.Queue = queue.Queue()
    hk = start_hotkeys(q)

    baseline = None
    if BASELINE.exists() and not args.recalibrate:
        baseline = Baseline.load(BASELINE)
        say(f"using saved calibration from {baseline.created[:16]} (Ctrl+Alt+C to recalibrate)")
    log.write("header", version=1, camera=args.camera, backend=args.backend, size=list(size),
              capture=list(CAPTURE), interval=args.interval, scoring=asdict(scoring), monitor=asdict(mcfg),
              baseline=asdict(baseline) if baseline else None, hotkeys=HOTKEYS)
    if baseline is None:
        while baseline is None:
            baseline = run_calibration(cam, backend, size, log)
            if baseline is None:
                say("retrying calibration in 10 s (make sure you are in view)")
                time.sleep(10)
    monitor = PostureMonitor(baseline, scoring, mcfg, start_away=True)

    say(f"logging to {log.path}")
    say("hotkeys: Ctrl+Alt+S slouching, G good, W wrong reminder, C recalibrate, P pause, Q quit")
    paused, last_state, next_sample = False, None, 0.0
    stop_at = time.monotonic() + args.minutes * 60 if args.minutes else float("inf")
    try:
        while time.monotonic() < stop_at:
            try:
                action = q.get(timeout=max(0.0, min(0.25, next_sample - time.monotonic())))
            except queue.Empty:
                action = None
            if action == "quit":
                break
            if action == "pause":
                paused = not paused
                if paused:
                    cam.release()
                    log.write("paused")
                    say("paused, camera released (Ctrl+Alt+P to resume)")
                else:
                    log.write("resumed")
                    say("resumed")
                    monitor = PostureMonitor(monitor.baseline, scoring, mcfg, start_away=True)
                    last_state = None
                continue
            if action and action.startswith("label:"):
                label = action.split(":", 1)[1]
                log.write("label", label=label)
                beep()
                say(f"label: {label}")
                continue
            if action == "recalibrate" and not paused:
                new = run_calibration(cam, backend, size, log)
                if new is not None:
                    monitor.reset_baseline(new)
                continue

            if paused or time.monotonic() < next_sample:
                continue
            next_sample = time.monotonic() + args.interval

            try:
                frame = cam.frame()
            except (ConnectionError, RuntimeError) as e:
                log.write("camera_error", error=str(e))
                say(f"camera problem: {e}; retrying every {RETRY_S:.0f} s")
                continue
            if frame is None:
                continue
            t0 = time.perf_counter()
            lm = backend.detect(to_rgb(frame, size))
            ms = (time.perf_counter() - t0) * 1e3
            now = time.time()
            status, events = monitor.update(now, lm)
            sample = {"ms": round(ms, 1), "state": status.state}
            if lm is not None:
                sample.update(lm.to_json())
            log.write("sample", **sample)
            for ev in events:
                log.write("event", kind=ev.kind, at=ev.t, detail=ev.detail)
                if ev.kind == REMINDER:
                    say(f"REMINDER: slouching for {ev.detail['slouch_s']:.0f} s")
                    notify("hunch", "You've been slouching for a while. Sit back and lift your head.")
                elif ev.kind == RECALIBRATE:
                    say("your position differs a lot from calibration; press Ctrl+Alt+C if you are sitting upright")
                    notify("hunch", "Position changed since calibration. Sit upright and press Ctrl+Alt+C.")
                else:
                    say(ev.kind)
            if status.state != last_state:
                say(f"state: {status.state}")
                last_state = status.state
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        log.write("end")
        log.close()
        if hk is not None:
            hk.stop()
        say(f"stopped; log saved to {log.path}")


if __name__ == "__main__":
    main()
