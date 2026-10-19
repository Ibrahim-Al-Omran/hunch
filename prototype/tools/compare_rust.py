"""Dev check: Rust spike landmarks vs the Python ONNX port and MediaPipe on one image.

    python tools/compare_rust.py data/testimg/desk.jpg
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hunch_cv.capture import to_rgb  # noqa: E402
from hunch_cv.pose import MediaPipeBackend, OnnxBackend  # noqa: E402

SPIKE = ROOT.parent / "spike" / "target" / "release" / ("hunch-spike.exe" if sys.platform == "win32" else "hunch-spike")
KEY = [0, 2, 5, 7, 8, 11, 12]


def main() -> None:
    img = cv2.imread(sys.argv[1])
    for variant in ("lite", "full"):
        mp_, ox = MediaPipeBackend(variant), OnnxBackend(variant)
        for size in ((320, 240), (img.shape[1], img.shape[0])):
            bgr = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "in.png"
                cv2.imwrite(str(p), bgr)
                out = subprocess.run([str(SPIKE), "image", str(p), "--variant", variant],
                                     capture_output=True, text=True, check=True).stdout
            rust = json.loads(out)
            rgb = to_rgb(bgr)
            py, ref = ox.detect(rgb), mp_.detect(rgb)
            r_xy = np.array(rust["landmarks"])[:, :2]
            face = (np.linalg.norm(ref.xy[0] - ref.xy[7]) + np.linalg.norm(ref.xy[0] - ref.xy[8])) / 2
            e_py = np.linalg.norm(r_xy[KEY] - py.xy[KEY], axis=1) / face
            e_mp = np.linalg.norm(r_xy[KEY] - ref.xy[KEY], axis=1) / face
            print(f"{variant} {size}: rust vs python-onnx median {np.median(e_py):.3f} max {e_py.max():.3f} | "
                  f"rust vs mediapipe median {np.median(e_mp):.3f} max {e_mp.max():.3f} (face sizes); "
                  f"rust {rust['ms']:.1f} ms")


if __name__ == "__main__":
    main()
