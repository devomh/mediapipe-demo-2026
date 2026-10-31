"""Detector state machines driven directly with chosen feature values."""

import pytest

from ai_playground.gestures.detectors import BuiltinGestureDetector, DetectorInput, PinchDetector
from ai_playground.gestures.features import HandFeatures


def x(ratio=1.0, gesture="None", score=0.0, t=0.0, vel=(0.0, 0.0)):
    f = HandFeatures(
        palm_length=0.09,
        pinch_ratio=ratio,
        pinch_point=(0.4, 0.6),
        finger_extended=(True,) * 5,
        finger_count=5,
        palm_center=(0.5, 0.5),
        palm_roll=0.0,
        hand_scale_image=0.2,
    )
    return DetectorInput(
        hand_id=1,
        handedness="Right",
        t=t,
        features=f,
        gesture=gesture,
        gesture_score=score,
        pinch_vel=vel,
        palm_vel=(0.0, 0.0),
    )


def types(det, inputs):
    return [[e.type for e in det.update(i)] for i in inputs]


def test_pinch_needs_min_frames(cfg):
    p = cfg.gestures.pinch
    out = types(PinchDetector(p), [x(1.0), x(0.1), x(1.0), x(0.1), x(0.1), x(0.1)])
    assert out == [[], [], [], [], ["pinch_start"], ["pinch_move"]]


def test_pinch_hysteresis_between_thresholds(cfg):
    p = cfg.gestures.pinch
    mid = (p.enter + p.exit) / 2
    d = PinchDetector(p)
    # Between the thresholds: doesn't start from idle...
    assert types(d, [x(mid)] * 10) == [[]] * 10
    # ...and doesn't end once pinched.
    types(d, [x(0.1)] * p.min_frames)
    assert d.active
    assert types(d, [x(mid)] * 10) == [["pinch_move"]] * 10
    assert d.active


def test_pinch_boundaries_are_strict(cfg):
    p = cfg.gestures.pinch
    d = PinchDetector(p)
    types(d, [x(p.enter)] * 5)
    assert not d.active  # needs ratio < enter
    types(d, [x(p.enter - 1e-3)] * p.min_frames)
    assert d.active
    types(d, [x(p.exit)] * 5)
    assert d.active  # needs ratio > exit
    out = types(d, [x(p.exit + 1e-3)] * p.min_frames)
    assert out[-1] == ["pinch_end"] and not d.active


def test_pinch_end_carries_release_velocity_and_reason(cfg):
    p = cfg.gestures.pinch
    d = PinchDetector(p)
    for _ in range(p.min_frames):
        d.update(x(0.1))
    evs = []
    for _ in range(p.min_frames):
        evs += d.update(x(1.0, vel=(1.5, -0.5)))
    end = evs[-1]
    assert end.type == "pinch_end"
    assert end.vel == (1.5, -0.5)
    assert end.data == {"reason": "released"}


def test_pinch_strength(cfg):
    p = cfg.gestures.pinch
    d = PinchDetector(p)
    for _ in range(p.min_frames):
        d.update(x(0.0))
    (e,) = d.update(x(0.0))
    assert e.strength == pytest.approx(1.0)
    (e,) = d.update(x(p.enter))
    assert e.strength == pytest.approx(0.0)
    (e,) = d.update(x(p.enter / 2))
    assert e.strength == pytest.approx(0.5)


def test_pinch_blocked_while_mediapipe_sees_a_fist(cfg):
    p = cfg.gestures.pinch
    d = PinchDetector(p)
    assert types(d, [x(0.2, "Closed_Fist", 0.9)] * 10) == [[]] * 10


def test_pinch_end_on_lost(cfg):
    p = cfg.gestures.pinch
    d = PinchDetector(p)
    assert d.end(x(), "lost") == []
    for _ in range(p.min_frames):
        d.update(x(0.1))
    (e,) = d.end(x(), "lost")
    assert e.type == "pinch_end" and e.data == {"reason": "lost"}


def test_builtin_start_end_debounce(cfg):
    b = cfg.gestures.builtin
    d = BuiltinGestureDetector("Closed_Fist", "fist", b)
    seen = x(gesture="Closed_Fist", score=0.9)
    gone = x(gesture="None")
    out = types(d, [seen] * b.start_frames + [gone] * b.end_frames)
    assert out[b.start_frames - 1] == ["fist_start"]
    assert out[-1] == ["fist_end"]
    assert sum(len(o) for o in out) == 2


def test_builtin_low_score_does_not_start(cfg):
    b = cfg.gestures.builtin
    d = BuiltinGestureDetector("Closed_Fist", "fist", b)
    assert types(d, [x(gesture="Closed_Fist", score=b.min_score - 0.01)] * 10) == [[]] * 10


def test_builtin_low_score_does_not_end(cfg):
    """Once started, only absence of the category ends it (hysteresis)."""
    b = cfg.gestures.builtin
    d = BuiltinGestureDetector("Open_Palm", "palm", b)
    types(d, [x(gesture="Open_Palm", score=0.9)] * b.start_frames)
    assert types(d, [x(gesture="Open_Palm", score=0.3)] * 10) == [[]] * 10
    assert d.active


def test_builtin_brief_dropout_does_not_end(cfg):
    b = cfg.gestures.builtin
    d = BuiltinGestureDetector("Open_Palm", "palm", b)
    on, off = x(gesture="Open_Palm", score=0.9), x(gesture="None")
    out = types(d, [on] * b.start_frames + [off] * (b.end_frames - 1) + [on] * 5)
    assert sum(len(o) for o in out) == 1 and d.active


def test_builtin_suppressed(cfg):
    b = cfg.gestures.builtin
    d = BuiltinGestureDetector("Closed_Fist", "fist", b)
    for _ in range(10):
        assert d.update(x(gesture="Closed_Fist", score=0.9), suppressed=True) == []
