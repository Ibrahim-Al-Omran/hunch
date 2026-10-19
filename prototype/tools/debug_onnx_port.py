"""Dev check: how closely does the ONNX port reproduce MediaPipe landmarks?

    python tools/debug_onnx_port.py data/testimg/desk.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv.capture import to_rgb  # noqa: E402
from hunch_cv.pose import MediaPipeBackend, OnnxBackend  # noqa: E402

KEY = [0, 2, 5, 7, 8, 11, 12]


def main() -> None:
    img = cv2.imread(sys.argv[1])
    for variant in ("lite", "full"):
        mp_ = MediaPipeBackend(variant)
        for size in ((320, 240), (640, 480), (img.shape[1], img.shape[0])):
            rgb = to_rgb(img, size)
            ref = mp_.detect(rgb)
            face = np.linalg.norm(ref.xy[0] - ref.xy[7]) + np.linalg.norm(ref.xy[0] - ref.xy[8])
            for refine in (False, True):
                got = OnnxBackend(variant, refine_heatmap=refine).detect(rgb)
                if got is None:
                    print(f"{variant} {size} refine={refine}: not detected")
                    continue
                err = np.linalg.norm(ref.xy[KEY] - got.xy[KEY], axis=1) / (face / 2)
                print(f"{variant} {size} refine={refine}: error / face size  "
                      f"median {np.median(err):.3f}  max {err.max():.3f}")


if __name__ == "__main__":
    main()
