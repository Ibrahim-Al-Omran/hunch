import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunch_cv import landmarks as L  # noqa: E402
from hunch_cv.calibration import CalibrationError, calibrate, learn_slouch_direction  # noqa: E402
from hunch_cv.features import compute_features  # noqa: E402
from hunch_cv.landmarks import Landmarks  # noqa: E402
from hunch_cv.pose import OnnxBackend, ssd_anchors  # noqa: E402
from hunch_cv.scoring import (GOOD, INSUFFICIENT, POSSIBLE, RULE_FEATURES, UNCERTAIN, ScoringConfig,  # noqa: E402
                              WindowMedian, assess)

W, H = 320, 240


def pose(head_dx=0.0, head_dy=0.0, shoulder_dy=0.0, scale=1.0, shift=(0.0, 0.0), turn=0.0, vis=0.95,
         rng=None, noise=0.0):
    """Three-quarter view of a seated person facing image-right, in pixels."""
    p = {
        L.LEFT_SHOULDER: (150, 170), L.RIGHT_SHOULDER: (215, 165),
        L.LEFT_EAR: (158, 100), L.RIGHT_EAR: (200, 98),
        L.LEFT_EYE: (178, 92), L.RIGHT_EYE: (196, 91), L.NOSE: (190, 104),
        L.MOUTH_LEFT: (180, 116), L.MOUTH_RIGHT: (192, 116),
        L.LEFT_HIP: (150, 260), L.RIGHT_HIP: (205, 260),
    }
    head = {L.LEFT_EAR, L.RIGHT_EAR, L.LEFT_EYE, L.RIGHT_EYE, L.NOSE, L.MOUTH_LEFT, L.MOUTH_RIGHT}
    xy = np.zeros((33, 2))
    for i, (x, y) in p.items():
        if i in head:
            x, y = x + head_dx, y + head_dy
        if i in (L.LEFT_SHOULDER, L.RIGHT_SHOULDER):
            y += shoulder_dy
        xy[i] = (x, y)
    if turn:
        xy[L.NOSE, 0] += turn
    xy = xy * scale + np.asarray(shift)
    if rng is not None and noise:
        xy = xy + rng.normal(0, noise, xy.shape)
    norm = np.zeros((33, 5))
    norm[:, 0] = xy[:, 0] / W
    norm[:, 1] = xy[:, 1] / H
    norm[:, 3] = vis
    norm[:, 4] = vis
    norm[[L.LEFT_HIP, L.RIGHT_HIP], 3] = 0.2  # hips usually out of view at a desk
    return Landmarks(norm, None, W, H)


def upright_set(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [pose(rng=rng, noise=0.6) for _ in range(n)]


def test_features_scale_with_calibrated_unit():
    a = compute_features(pose(), "left", scale=30.0).values
    b = compute_features(pose(scale=1.3), "left", scale=39.0).values
    for k in ("head_y", "shoulder_y", "ear_forward", "neck_ratio", "face_scale", "head_pitch", "yaw"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_relative_features_ignore_shift():
    a = compute_features(pose(), "left", scale=30.0).values
    b = compute_features(pose(shift=(-20, 10)), "left", scale=30.0).values
    for k in ("ear_forward", "neck_ratio", "head_pitch", "yaw", "shoulder_width"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), k
    assert b["head_y"] - a["head_y"] == pytest.approx(10 / 30.0)


def test_side_chosen_by_depth_then_visibility():
    lm = pose()
    lm.norm[[L.RIGHT_EAR, L.RIGHT_SHOULDER], 3] = 0.3
    assert calibrate([lm] * 10).side == "left"
    near_right = pose()
    near_right.norm[[L.LEFT_EAR, L.LEFT_SHOULDER], 2] = -0.3
    near_right.norm[[L.RIGHT_EAR, L.RIGHT_SHOULDER], 2] = -0.6
    assert calibrate([near_right] * 10).side == "right"


def feats(lm, base):
    return compute_features(lm, base.side, base.scale_px)


def test_forward_head_and_slouch_flagged_upright_not():
    base = calibrate(upright_set())
    cfg = ScoringConfig()
    rng = np.random.default_rng(5)
    for _ in range(20):
        a = assess(feats(pose(rng=rng, noise=0.6), base), base, cfg)
        assert a.state == GOOD, a
    forward = assess(feats(pose(head_dx=10, head_dy=8), base), base, cfg)
    slouch = assess(feats(pose(head_dx=6, head_dy=14, shoulder_dy=4), base), base, cfg)
    assert forward.state == POSSIBLE and slouch.state == POSSIBLE


def test_low_visibility_is_insufficient_not_slouch():
    base = calibrate(upright_set())
    a = assess(feats(pose(head_dx=10, head_dy=8, vis=0.3), base), base, ScoringConfig())
    assert a.state == INSUFFICIENT
    assert assess(None, base, ScoringConfig()).state == INSUFFICIENT


def test_turning_is_uncertain():
    base = calibrate(upright_set())
    a = assess(feats(pose(turn=-14), base), base, ScoringConfig())
    assert a.state == UNCERTAIN and "turned" in a.reasons


def test_calibration_rejects_poor_visibility():
    lms = upright_set(10) + [pose(vis=0.2) for _ in range(20)]
    with pytest.raises(CalibrationError):
        calibrate(lms)
    with pytest.raises(CalibrationError):
        calibrate([None] * 10)


def test_learned_direction_is_unit_and_scores_slouch():
    base = calibrate(upright_set())
    rng = np.random.default_rng(3)
    ref = [pose(head_dx=8, head_dy=10, rng=rng, noise=0.6) for _ in range(8)]
    base.slouch_direction = learn_slouch_direction(base, ref, tuple(RULE_FEATURES))
    assert math.sqrt(sum(v * v for v in base.slouch_direction.values())) == pytest.approx(1.0)
    cfg = ScoringConfig(method="learned")
    assert assess(feats(pose(head_dx=8, head_dy=10), base), base, cfg).state == POSSIBLE
    assert assess(feats(pose(), base), base, cfg).state == GOOD


def test_window_median_tolerates_one_unscorable_sample():
    base = calibrate(upright_set())
    w = WindowMedian(3)
    good = assess(feats(pose(), base), base, ScoringConfig())
    bad = assess(None, base, ScoringConfig())
    assert math.isnan(w.push(good)) and math.isnan(w.push(good))
    assert math.isfinite(w.push(good))
    assert math.isfinite(w.push(bad))
    assert math.isnan(w.push(bad))


def test_anchor_count_matches_detector_output():
    assert ssd_anchors().shape == (2254, 2)


def test_roi_rotation_upright_is_zero():
    (_, _), side, rot = OnnxBackend._roi_from_keypoints(np.array([[100.0, 200.0], [100.0, 120.0]]))
    assert rot == pytest.approx(0.0, abs=1e-9)
    assert side == pytest.approx(2 * 80 * 1.25)
