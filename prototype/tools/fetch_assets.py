"""Download the MediaPipe pose models and the public reference photo.

    python tools/fetch_assets.py

ONNX versions are produced separately by tools/convert_models.py.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
# "Boy at desk while in the process of programming", Wikimedia Commons, CC BY-SA 3.0
PHOTO_URL = ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/"
             "Boy_at_desk_while_in_the_process_of_programming.jpg/"
             "960px-Boy_at_desk_while_in_the_process_of_programming.jpg")


def fetch(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"have {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "hunch-prototype/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dst.write_bytes(r.read())
    print(f"fetched {dst.relative_to(ROOT)}")


def main() -> None:
    for v in ("lite", "full"):
        task = MODELS / f"pose_landmarker_{v}.task"
        fetch(MODEL_URL.format(v=v), task)
        zipfile.ZipFile(task).extractall(MODELS / v)
    fetch(PHOTO_URL, ROOT / "data" / "testimg" / "desk.jpg")


if __name__ == "__main__":
    main()
