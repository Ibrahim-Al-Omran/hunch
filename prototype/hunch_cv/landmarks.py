from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

NUM_LANDMARKS = 33

# Subset of the BlazePose topology used for the preview overlay.
CONNECTIONS = [
    (LEFT_EAR, LEFT_EYE), (LEFT_EYE, NOSE), (NOSE, RIGHT_EYE), (RIGHT_EYE, RIGHT_EAR),
    (MOUTH_LEFT, MOUTH_RIGHT),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, 13), (13, 15), (RIGHT_SHOULDER, 14), (14, 16),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
]


@dataclass
class Landmarks:
    """One person's pose in a single frame.

    `norm` holds x, y (normalized to image width/height), z, visibility and
    presence for the 33 BlazePose landmarks. `world` is the model's metric
    hip-centred estimate (metres, y down) or None.
    """

    norm: np.ndarray  # (33, 5)
    world: np.ndarray | None  # (33, 3)
    width: int
    height: int

    @property
    def xy(self) -> np.ndarray:
        """Pixel coordinates; aspect-correct, unlike the normalized values."""
        return self.norm[:, :2] * np.array([self.width, self.height], dtype=np.float64)

    @property
    def visibility(self) -> np.ndarray:
        return self.norm[:, 3]

    @property
    def presence(self) -> np.ndarray:
        return self.norm[:, 4]

    def to_json(self) -> dict:
        out = {"lm": np.round(self.norm, 5).tolist()}
        if self.world is not None:
            out["world"] = np.round(self.world, 5).tolist()
        return out

    @classmethod
    def from_json(cls, d: dict, width: int, height: int) -> Landmarks:
        world = np.asarray(d["world"], dtype=np.float64) if d.get("world") else None
        return cls(np.asarray(d["lm"], dtype=np.float64), world, width, height)
