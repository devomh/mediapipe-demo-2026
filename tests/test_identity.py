from dataclasses import replace

import numpy as np
from synthetic import frame, make_hand

from ai_playground.tracking.identity import HandTracker


def track(cfg, hand_seq):
    tr = HandTracker(cfg.tracking)
    return [tr.update(frame(i + 1, i / 30, hs)) for i, hs in enumerate(hand_seq)]


def ids(out):
    return [th.id for th in out if th.missed == 0]


def test_small_far_hands_are_ignored(cfg):
    out = track(cfg, [[make_hand(scale=0.05, center=(0.2, 0.3)), make_hand(scale=0.2)]] * 5)
    assert all(len(o) == 1 for o in out)


def test_ids_stable_while_moving(cfg):
    seq = [[make_hand(center=(0.3 + 0.01 * i, 0.5))] for i in range(40)]
    out = track(cfg, seq)
    assert {th.id for o in out for th in o} == {1}
    assert out[-1][0].age == 40


def test_crossing_hands_keep_ids_and_handedness(cfg):
    seq = []
    for i in range(40):
        x = 0.25 + 0.5 * i / 39
        seq.append(
            [
                make_hand(handedness="Right", center=(x, 0.4)),
                make_hand(handedness="Left", center=(1.0 - x, 0.65)),
            ]
        )
    out = track(cfg, seq)
    first = {th.id: th.hand.handedness for th in out[0]}
    for o in out:
        assert {th.id: th.hand.handedness for th in o} == first


def test_teleporting_hand_gets_new_id(cfg):
    out = track(cfg, [[make_hand(center=(0.2, 0.5))], [make_hand(center=(0.8, 0.5))]])
    assert ids(out[0]) == [1]
    assert ids(out[1]) == [2]


def test_grace_frames_bridge_short_dropouts(cfg):
    g = cfg.tracking.grace_frames
    h = make_hand()
    out = track(cfg, [[h]] * 3 + [[]] * g + [[h]])
    assert all(o and o[0].id == 1 for o in out)
    assert [o[0].missed for o in out[3 : 3 + g]] == list(range(1, g + 1))
    assert out[-1][0].missed == 0


def test_track_dropped_after_grace(cfg):
    g = cfg.tracking.grace_frames
    h = make_hand()
    out = track(cfg, [[h]] * 3 + [[]] * (g + 1) + [[h]])
    assert out[3 + g] == []
    assert ids(out[-1]) == [2]  # IDs only increase


def test_handedness_majority_vote_absorbs_label_flips(cfg):
    right = make_hand(handedness="Right")
    seq = [[right]] * 10 + [[replace(right, handedness="Left")]] * 2 + [[right]] * 3
    out = track(cfg, seq)
    assert all(o[0].hand.handedness == "Right" for o in out)
    assert {o[0].id for o in out} == {1}


def test_smoothing_reduces_landmark_jitter(cfg):
    rng = np.random.default_rng(3)
    raw = [make_hand(noise=0.002, rng=rng) for _ in range(120)]
    out = track(cfg, [[h] for h in raw])
    raw_std = np.std([h.landmarks[8, 0] for h in raw[30:]])
    smooth_std = np.std([o[0].hand.landmarks[8, 0] for o in out[30:]])
    assert smooth_std < raw_std / 2
