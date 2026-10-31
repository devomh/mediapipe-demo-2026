import math

import numpy as np
import pytest
from synthetic import PALM_LENGTH, make_hand

from ai_playground.gestures import features as F


def feats(cfg, hand, size=(640, 480)):
    return F.compute(hand.landmarks, hand.world, size, cfg.gestures.fingers)


def test_palm_length():
    assert F.palm_length(make_hand().world) == pytest.approx(PALM_LENGTH, rel=1e-3)


@pytest.mark.parametrize("d", [0.05, 0.15, 0.3, 0.6, 1.0])
def test_pinch_ratio_matches_constructed_distance(d):
    assert F.pinch_ratio(make_hand(pinch=d).world) == pytest.approx(d, rel=1e-3)


def test_pinch_ratio_ranges(cfg):
    assert feats(cfg, make_hand(pinch=0.15)).pinch_ratio < cfg.gestures.pinch.enter
    assert 0.8 <= feats(cfg, make_hand()).pinch_ratio <= 1.2  # open hand


def test_pinch_ratio_independent_of_distance_to_camera(cfg):
    near = feats(cfg, make_hand(pinch=0.3, scale=0.3))
    far = feats(cfg, make_hand(pinch=0.3, scale=0.1))
    assert near.pinch_ratio == pytest.approx(far.pinch_ratio, rel=1e-4)
    assert near.hand_scale_image == pytest.approx(0.3, rel=1e-3)
    assert far.hand_scale_image == pytest.approx(0.1, rel=1e-3)


@pytest.mark.parametrize(
    "extended",
    [
        (True, True, True, True, True),
        (False, True, True, True, True),
        (False, True, True, False, False),
        (False, True, False, False, False),
        (True, False, False, False, False),
        (False, False, False, False, False),
    ],
)
def test_finger_extended_and_count(cfg, extended):
    f = feats(cfg, make_hand(extended=extended))
    assert f.finger_extended == extended
    assert f.finger_count == sum(extended)


def test_fist_has_no_extended_fingers(cfg):
    assert feats(cfg, make_hand(fist=True)).finger_count == 0


def test_left_hand_is_mirror_of_right(cfg):
    r = feats(cfg, make_hand(handedness="Right"))
    lf = feats(cfg, make_hand(handedness="Left"))
    assert r.finger_extended == lf.finger_extended
    assert r.pinch_ratio == pytest.approx(lf.pinch_ratio)
    # Roll: index->pinky points right for the right hand, left for the left hand.
    assert abs(r.palm_roll) < math.pi / 4
    assert abs(lf.palm_roll) > 3 * math.pi / 4


def test_palm_center_and_pinch_point_follow_position(cfg):
    a = feats(cfg, make_hand(pinch=0.1, center=(0.3, 0.4)))
    b = feats(cfg, make_hand(pinch=0.1, center=(0.6, 0.5)))
    assert b.palm_center[0] - a.palm_center[0] == pytest.approx(0.3, abs=1e-4)
    assert b.pinch_point[1] - a.pinch_point[1] == pytest.approx(0.1, abs=1e-4)


def test_joint_angle():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert F.joint_angle_deg(pts, 0, 1, 2) == pytest.approx(180.0)
    assert F.joint_angle_deg(pts, 0, 1, 3) == pytest.approx(90.0)


def test_velocity_tracker_constant_motion():
    v = F.VelocityTracker(window_s=0.1)
    out = (0.0, 0.0)
    for i in range(20):
        t = i / 30
        out = v.update(t, (0.2 + 0.5 * t, 0.5 - 0.25 * t))
    assert out[0] == pytest.approx(0.5, rel=1e-6)
    assert out[1] == pytest.approx(-0.25, rel=1e-6)


def test_velocity_tracker_first_sample_is_zero():
    assert F.VelocityTracker(0.1).update(1.0, (0.5, 0.5)) == (0.0, 0.0)
