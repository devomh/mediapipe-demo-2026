"""Engine end to end: synthetic frames -> HandTracker -> GestureEngine -> events."""

import numpy as np
from synthetic import frame, make_hand

from ai_playground.gestures.engine import GestureEngine
from ai_playground.gestures.features import pinch_ratio
from ai_playground.tracking.identity import HandTracker


def run(cfg, hand_seq, fps=30.0):
    tracker, engine = HandTracker(cfg.tracking), GestureEngine(cfg.gestures)
    per_frame, states = [], []
    for i, hands in enumerate(hand_seq):
        t = i / fps
        evs, st = engine.update(tracker.update(frame(i + 1, t, hands)), t)
        per_frame.append(evs)
        states.append(st)
    return per_frame, states


def flat(per_frame, skip_moves=True):
    return [e for evs in per_frame for e in evs if not (skip_moves and e.type.endswith("_move"))]


def names(per_frame):
    return [e.type for e in flat(per_frame)]


def test_hand_enter_once(cfg):
    per_frame, states = run(cfg, [[make_hand()]] * 20)
    assert names(per_frame) == ["hand_enter"]
    assert states[-1][0].features.finger_count == 5


def test_no_hands_no_events(cfg):
    per_frame, _ = run(cfg, [[]] * 30)
    assert flat(per_frame, skip_moves=False) == []


def test_lost_hand_ends_active_detectors(cfg):
    g = cfg.tracking.grace_frames
    pinch = make_hand(pinch=0.1)
    per_frame, _ = run(cfg, [[pinch]] * 10 + [[]] * (g + 1))
    evs = flat(per_frame)
    assert [e.type for e in evs] == ["hand_enter", "pinch_start", "pinch_end", "hand_exit"]
    assert evs[2].data == {"reason": "lost"}
    assert evs[3].data == {"reason": "lost"}


def test_grace_dropout_is_silent(cfg):
    g = cfg.tracking.grace_frames
    pinch = make_hand(pinch=0.1)
    per_frame, states = run(cfg, [[pinch]] * 10 + [[]] * g + [[pinch]] * 10)
    assert names(per_frame) == ["hand_enter", "pinch_start"]
    assert all(st and st[0].pinching for st in states[10:])
    assert not any(st[0].visible for st in states[10 : 10 + g])


def test_fast_pinch_fires_within_two_frames(cfg):
    """PLAN exit criterion: pinch_start within <= 2 frames of the real (raw) crossing."""
    closing = [1.0, 0.7, 0.4, 0.15, 0.1]  # ~130 ms from open to pinched
    seq = [make_hand(pinch=1.0)] * 15 + [make_hand(pinch=d) for d in closing]
    seq += [make_hand(pinch=0.1)] * 10
    per_frame, _ = run(cfg, [[h] for h in seq])
    raw_cross = next(
        i for i, h in enumerate(seq) if pinch_ratio(h.world) < cfg.gestures.pinch.enter
    )
    fired = next(i for i, evs in enumerate(per_frame) if any(e.type == "pinch_start" for e in evs))
    assert 0 <= fired - raw_cross <= 2


def test_noisy_long_hold_does_not_flicker(cfg):
    """30 s pinch hold with MediaPipe-like jitter and a moving hand: one start, no end."""
    rng = np.random.default_rng(7)
    seq = [make_hand(pinch=1.0, noise=0.002, rng=rng)] * 10
    for i in range(900):
        c = (0.5 + 0.2 * np.sin(i / 40), 0.5 + 0.1 * np.cos(i / 55))
        seq.append(make_hand(pinch=0.18, center=c, noise=0.002, rng=rng))
    per_frame, _ = run(cfg, [[h] for h in seq])
    assert names(per_frame) == ["hand_enter", "pinch_start"]


def test_static_open_hand_has_no_false_pinches(cfg):
    rng = np.random.default_rng(11)
    seq = [[make_hand(noise=0.002, rng=rng)] for _ in range(600)]
    per_frame, _ = run(cfg, seq)
    assert names(per_frame) == ["hand_enter"]


def test_fist_cycle_no_pinch_false_positives(cfg):
    """open -> fist x4 with MediaPipe labels lagging the pose by 2 frames."""
    seq = []
    for _ in range(4):
        seq += [make_hand(gesture="Open_Palm", gesture_score=0.9)] * 20
        seq += [make_hand(fist=True, gesture="None")] * 2  # label lags the pose
        seq += [make_hand(fist=True, gesture="Closed_Fist", gesture_score=0.9)] * 20
    per_frame, _ = run(cfg, [[h] for h in seq])
    got = names(per_frame)
    assert "pinch_start" not in got
    assert got.count("fist_start") == 4
    assert got.count("fist_end") == 3  # the clip ends mid-fist
    assert got.count("palm_start") == 4


def test_fist_suppressed_during_pinch(cfg):
    pinch_fist_label = make_hand(pinch=0.1, gesture="Closed_Fist", gesture_score=0.9)
    seq = [make_hand(pinch=0.1)] * 5 + [pinch_fist_label] * 10
    per_frame, _ = run(cfg, [[h] for h in seq])
    assert "fist_start" not in names(per_frame)


def test_two_hands_pinch_independently(cfg):
    def pair(r, lft):
        return [
            make_hand(handedness="Right", pinch=r, center=(0.7, 0.5)),
            make_hand(handedness="Left", pinch=lft, center=(0.3, 0.5)),
        ]

    seq = [pair(1.0, 1.0)] * 5 + [pair(0.1, 1.0)] * 10 + [pair(0.1, 0.1)] * 10
    seq += [pair(1.0, 0.1)] * 10
    per_frame, _ = run(cfg, seq)
    evs = [e for e in flat(per_frame) if e.type.startswith("pinch")]
    assert [(e.type, e.handedness) for e in evs] == [
        ("pinch_start", "Right"),
        ("pinch_start", "Left"),
        ("pinch_end", "Right"),
    ]


def test_pinch_moves_follow_the_hand(cfg):
    seq = [make_hand(pinch=0.1, center=(0.3 + 0.01 * i, 0.5)) for i in range(30)]
    per_frame, _ = run(cfg, [[h] for h in seq])
    moves = [e for e in flat(per_frame, skip_moves=False) if e.type == "pinch_move"]
    assert moves[-1].pos[0] > moves[0].pos[0]
    assert moves[-1].vel[0] > 0.2  # 0.3 units/s, smoothed


def test_event_json_shape(cfg):
    per_frame, _ = run(cfg, [[make_hand(pinch=0.1)]] * 5)
    e = next(e for e in flat(per_frame) if e.type == "pinch_start")
    j = e.to_json()
    assert j["gesture"] == "pinch_start"
    assert set(j) >= {"gesture", "position", "strength", "velocity"}
    assert len(j["position"]) == 2
