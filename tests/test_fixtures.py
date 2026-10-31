"""Phase 0 §9 fixture expectations, replayed through tracker + engine.

Clips are recorded on the MacBook with a real camera (see README "Recording fixtures"). A test
is skipped until its clip exists in tests/fixtures/.
"""

from collections import Counter
from pathlib import Path

import pytest

from ai_playground.gestures.engine import GestureEngine
from ai_playground.gestures.features import pinch_ratio
from ai_playground.sources.replay import load_clip
from ai_playground.tracking.identity import HandTracker

FIXTURES = Path(__file__).parent / "fixtures"


def clip_or_skip(name):
    for suffix in (".jsonl.gz", ".jsonl"):
        p = FIXTURES / (name + suffix)
        if p.exists():
            return load_clip(p)
    pytest.skip(f"fixture {name} not recorded yet")


def run(cfg, clip):
    tracker, engine = (
        HandTracker(cfg.tracking),
        GestureEngine(cfg.gestures, tuple(clip.header["size"])),
    )
    events, ids_per_frame = [], []
    for fr in clip.frames:
        tracked = tracker.update(fr)
        evs, states = engine.update(tracked, fr.t_capture)
        events += [e for e in evs if not e.type.endswith("_move")]
        ids_per_frame.append({s.id: s.handedness for s in states if s.visible})
    return events, ids_per_frame


def count(events):
    return Counter(e.type for e in events)


def test_no_hands(cfg):
    events, _ = run(cfg, clip_or_skip("no_hands"))
    assert events == []


def test_right_static_open(cfg):
    """An open, still hand: it enters once and never pinches or fists.

    palm_start/palm_end are expected, since MediaPipe classifies a still open hand as Open_Palm.
    """
    events, _ = run(cfg, clip_or_skip("right_static_open"))
    c = count(events)
    assert c["hand_enter"] == 1
    assert c["pinch_start"] == 0 and c["fist_start"] == 0


def test_right_pinch_x5(cfg):
    clip = clip_or_skip("right_pinch_x5")
    events, _ = run(cfg, clip)
    c = count(events)
    assert c["pinch_start"] == 5 and c["pinch_end"] == 5
    starts = [e.t for e in events if e.type == "pinch_start"]

    # Space marks are human-timed: sometimes pressed as the fingers touch (just before the
    # 2-frame confirmation), often well after, even after the release. So they only check
    # pairing: each mark falls in its own pinch's window, from slightly before it starts until
    # slightly before the next one starts.
    early = 0.5
    marks = [m.t for m in clip.marks]
    if len(marks) == 5:
        bounds = [*starts[1:], float("inf")]
        for i, (m, s, nxt) in enumerate(zip(marks, starts, bounds, strict=True)):
            assert s - early <= m < nxt - early, (
                f"mark {i + 1} at {m:.2f}s doesn't belong to pinch {i + 1} (starts {s:.2f}s)"
            )

    # Detection delay (PLAN exit criterion: <= 2 frames), measured against the clip's own raw
    # landmarks: the frame where the unsmoothed pinch ratio first dropped below `enter`.
    enter = cfg.gestures.pinch.enter
    raw = [min((pinch_ratio(h.world) for h in f.hands), default=float("inf")) for f in clip.frames]
    index = {f.t_capture: i for i, f in enumerate(clip.frames)}
    for t in starts:
        k = index[t]
        j = k
        while j > 0 and raw[j - 1] < enter:
            j -= 1
        assert raw[j] < enter, f"pinch_start at {t:.2f}s without a raw crossing"
        assert k - j <= 2, f"pinch_start at {t:.2f}s fired {k - j} frames after the raw crossing"


def test_right_pinch_hold(cfg):
    c = count(run(cfg, clip_or_skip("right_pinch_hold"))[0])
    assert c["pinch_start"] == 1 and c["pinch_end"] == 1


def test_both_pinch(cfg):
    events, _ = run(cfg, clip_or_skip("both_pinch"))
    per_hand = Counter((e.handedness, e.type) for e in events if e.type.startswith("pinch"))
    for hand in ("Left", "Right"):
        assert per_hand[(hand, "pinch_start")] >= 1
        assert per_hand[(hand, "pinch_start")] == per_hand[(hand, "pinch_end")]


def test_fist_palm_cycle(cfg):
    c = count(run(cfg, clip_or_skip("fist_palm_cycle"))[0])
    assert c["fist_start"] == 4 and c["fist_end"] >= 3
    assert c["pinch_start"] == 0


def test_hands_crossing(cfg):
    _, ids_per_frame = run(cfg, clip_or_skip("hands_crossing"))
    seen: dict[int, str] = {}
    for frame_ids in ids_per_frame:
        for hid, hd in frame_ids.items():
            assert seen.setdefault(hid, hd) == hd, f"hand {hid} changed handedness"
    assert len(seen) == 2, f"expected 2 stable IDs, got {sorted(seen)}"


def test_enter_exit(cfg):
    events, _ = run(cfg, clip_or_skip("enter_exit"))
    c = count(events)
    assert c["hand_enter"] == 4 and c["hand_exit"] >= 3
    assert all(e.data.get("reason") == "lost" for e in events if e.type == "hand_exit")


def test_far_bystander(cfg):
    events, _ = run(cfg, clip_or_skip("far_bystander"))
    assert events == []
