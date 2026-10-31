"""Magic Geometry scene: gesture events -> shapes. Driven directly, no threads."""

import math
from dataclasses import replace
from pathlib import Path

import pygame
import pytest
from synthetic import make_hand

from ai_playground.gestures import features
from ai_playground.gestures.engine import GestureEngine, HandState
from ai_playground.render.background import Background
from ai_playground.render.viewport import Viewport
from ai_playground.scenes.base import AppContext
from ai_playground.scenes.geometry import GeometryScene
from ai_playground.sources.replay import load_clip
from ai_playground.tracking.identity import HandTracker
from ai_playground.types import GestureEvent, TrackedHand

FIXTURES = Path(__file__).parent / "fixtures"
WINDOW = (1280, 720)


@pytest.fixture
def scene(cfg):
    pygame.init()
    surface = pygame.display.set_mode(WINDOW)
    ctx = AppContext(
        config=cfg,
        viewport=Viewport(WINDOW, (640, 480)),
        background=Background(),
        switch_scene=lambda name: None,
    )
    s = GeometryScene(seed=1)
    s.enter(ctx)
    s.surface = surface
    yield s
    pygame.quit()


def ev(type_, t, pos=(0.5, 0.5), vel=(0.0, 0.0), hand=1, **data):
    return GestureEvent(type_, hand, "Right", t, pos, vel=vel, data=data)


def test_pinch_hold_release_spawns_an_inflated_shape(scene, cfg):
    scene.handle([ev("pinch_start", 0.0)], [])
    assert len(scene.pending) == 1 and scene.world.pieces == []
    r0 = scene.pending[1].radius
    scene.handle([ev("pinch_move", cfg.geometry.inflate_s / 2, pos=(0.6, 0.4))], [])
    p = scene.pending[1]
    assert r0 < p.radius < cfg.geometry.max_size * WINDOW[1]
    assert p.pos == scene.ctx.viewport.to_px(0.6, 0.4)  # follows the pinch
    scene.handle([ev("pinch_move", 10.0)], [])
    assert scene.pending[1].radius == pytest.approx(cfg.geometry.max_size * WINDOW[1])
    scene.handle([ev("pinch_end", 10.1, reason="released")], [])
    assert scene.pending == {}
    (piece,) = scene.world.pieces
    assert piece.radius == pytest.approx(cfg.geometry.max_size * WINDOW[1])


def test_release_velocity_becomes_a_throw(scene):
    vp = scene.ctx.viewport
    scene.handle(
        [ev("pinch_start", 0.0), ev("pinch_end", 0.5, vel=(0.5, -0.25), reason="released")], []
    )
    (piece,) = scene.world.pieces
    assert tuple(piece.body.velocity) == pytest.approx(
        (0.5 * vp.dest_size[0], -0.25 * vp.dest_size[1])
    )


def test_lost_hand_drops_without_throwing(scene):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.5, vel=(2.0, -2.0), reason="lost")], [])
    (piece,) = scene.world.pieces
    assert tuple(piece.body.velocity) == (0.0, 0.0)


def test_kinds_cycle_and_hues_differ(scene):
    for i in range(6):
        scene.handle([ev("pinch_start", i), ev("pinch_end", i + 0.2, reason="released")], [])
    kinds = [p.kind for p in scene.world.pieces]
    assert kinds == ["circle", "bar", "triangle", "square", "pentagon", "circle"]
    hues = [p.hue for p in scene.world.pieces]
    assert all(a != b for a, b in zip(hues, hues[1:], strict=False))


FINGERS = {
    0: dict(fist=True),
    1: dict(extended=(False, True, False, False, False)),
    2: dict(extended=(False, True, True, False, False)),
    3: dict(extended=(False, True, True, True, False)),
    4: dict(extended=(False, True, True, True, True)),
    5: dict(extended=(True, True, True, True, True)),
}


def state(cfg, hid, fingers=5, pinching=False, handedness="Left"):
    """A HandState whose features come from a synthetic hand holding up `fingers` fingers."""
    hand = make_hand(handedness=handedness, center=(0.25, 0.5), **FINGERS[fingers])
    f = features.compute(hand.landmarks, hand.world, (640, 480), cfg.gestures.fingers)
    return HandState(TrackedHand(hid, hand, 10), f, (0.0, 0.0), pinching, fingers == 0, False)


def hold(scene, cfg, states, frames=None):
    for _ in range(frames or cfg.geometry.choose_frames):
        scene.handle([], states)


def test_synthetic_finger_counts_are_what_they_claim(cfg):
    for n in FINGERS:
        assert state(cfg, 2, n).features.finger_count == n


@pytest.mark.parametrize(
    "n,kind", [(1, "circle"), (2, "bar"), (3, "triangle"), (4, "square"), (5, "pentagon")]
)
def test_other_hand_fingers_choose_the_next_shape(scene, cfg, n, kind):
    other = state(cfg, 2, n)
    hold(scene, cfg, [other])
    assert scene.selection() == (kind, True)
    scene.handle([ev("pinch_start", 1.0, hand=1)], [other])
    assert scene.pending[1].kind == kind and scene.pending[1].chosen


def test_choice_needs_a_steady_count(scene, cfg):
    hold(scene, cfg, [state(cfg, 2, 3)], frames=cfg.geometry.choose_frames - 1)
    assert scene.selection() == ("circle", False)  # not yet: still the automatic "next"
    scene.handle([], [state(cfg, 2, 4)])  # count changed: starts over
    hold(scene, cfg, [state(cfg, 2, 4)], frames=cfg.geometry.choose_frames - 2)
    assert scene.selection()[1] is False
    scene.handle([], [state(cfg, 2, 4)])
    assert scene.selection() == ("square", True)


def test_fingers_change_the_shape_being_held(scene, cfg):
    scene.handle([ev("pinch_start", 0.0, hand=1)], [])
    assert scene.pending[1].kind == "circle" and not scene.pending[1].chosen
    hold(scene, cfg, [state(cfg, 2, 4)])
    assert scene.pending[1].kind == "square" and scene.pending[1].chosen
    scene.handle([ev("pinch_end", 1.0, hand=1, reason="released")], [state(cfg, 2, 4)])
    assert scene.world.pieces[0].kind == "square"


def test_fist_or_no_other_hand_keeps_the_automatic_cycle(scene, cfg):
    hold(scene, cfg, [state(cfg, 2, 0)])  # a fist chooses nothing
    for i in range(3):
        scene.handle([ev("pinch_start", i), ev("pinch_end", i + 0.2, reason="released")], [])
    assert [p.kind for p in scene.world.pieces] == ["circle", "bar", "triangle"]


def test_a_hand_closing_in_on_a_pinch_does_not_choose(scene, cfg):
    """On its way into a pinch a hand reads as 1 finger; that must not turn the shape round."""
    scene.handle([ev("pinch_start", 0.0, hand=1)], [])
    hold(scene, cfg, [state(cfg, 2, 4)])
    assert scene.pending[1].kind == "square"
    approaching = state(cfg, 2, 1)
    approaching = replace(approaching, features=replace(approaching.features, pinch_ratio=0.45))
    hold(scene, cfg, [approaching], frames=3 * cfg.geometry.choose_frames)
    assert scene.pending[1].kind == "square"


def test_the_pinching_hand_never_chooses_for_itself(scene, cfg):
    me = state(cfg, 1, 5, pinching=True)  # an open-ish hand that is pinching
    hold(scene, cfg, [me])
    scene.handle([ev("pinch_start", 0.0, hand=1)], [me])
    assert not scene.pending[1].chosen


def test_second_pinch_stretches_the_held_shape(scene, cfg):
    vp = scene.ctx.viewport
    scene.handle([ev("pinch_start", 0.0, hand=1, pos=(0.3, 0.5))], [])
    scene.handle([ev("pinch_start", 0.1, hand=2, pos=(0.5, 0.5))], [])
    assert set(scene.pending) == {1} and scene.pending[1].partner == 2  # one shape, two hands
    assert scene.selection() == (scene.pending[1].kind, False)  # palette shows the held shape
    scene.handle([ev("pinch_move", 0.2, hand=2, pos=(0.5, 0.3))], [])
    p = scene.pending[1]
    (ax, ay), (bx, by) = vp.to_px(0.3, 0.5), vp.to_px(0.5, 0.3)
    assert p.pos == pytest.approx(((ax + bx) / 2, (ay + by) / 2))
    gain = cfg.geometry.stretch_gain
    assert p.radius == pytest.approx(gain * math.hypot(bx - ax, by - ay) / 2)
    assert p.angle == pytest.approx(math.atan2(by - ay, bx - ax))
    scene.handle([ev("pinch_end", 0.3, hand=2, reason="released")], [])
    (piece,) = scene.world.pieces
    assert piece.body.angle == pytest.approx(p.angle) and piece.radius == pytest.approx(p.radius)
    scene.handle([ev("pinch_end", 0.4, hand=1, reason="released")], [])  # nothing left to drop
    assert len(scene.world.pieces) == 1 and scene.pending == {}


def test_stretch_size_is_clamped(scene, cfg):
    h = WINDOW[1]
    scene.handle([ev("pinch_start", 0.0, hand=1, pos=(0.0, 0.5))], [])
    scene.handle([ev("pinch_start", 0.1, hand=2, pos=(1.0, 0.5))], [])
    assert scene.pending[1].radius == pytest.approx(cfg.geometry.stretch_max_size * h)
    scene.handle([ev("pinch_move", 0.2, hand=2, pos=(0.0, 0.5))], [])
    assert scene.pending[1].radius == pytest.approx(cfg.geometry.min_size * h)


def test_stretch_release_throws_with_both_hands(scene, cfg):
    vp = scene.ctx.viewport
    scene.handle([ev("pinch_start", 0.0, hand=1, pos=(0.3, 0.5))], [])
    scene.handle([ev("pinch_start", 0.1, hand=2, pos=(0.5, 0.5))], [])
    scene.handle([ev("pinch_move", 0.2, hand=1, pos=(0.3, 0.5), vel=(0.2, 0.0))], [])
    scene.handle([ev("pinch_end", 0.3, hand=2, vel=(0.4, 0.0), reason="released")], [])
    (piece,) = scene.world.pieces
    assert piece.body.velocity.x == pytest.approx(0.3 * vp.dest_size[0])  # mean of both hands


def test_lost_stretch_hand_drops_the_shape(scene, cfg):
    scene.handle(
        [ev("pinch_start", 0.0, hand=1), ev("pinch_start", 0.1, hand=2, pos=(0.6, 0.5))], []
    )
    scene.handle([ev("pinch_end", 0.2, hand=2, vel=(3.0, 3.0), reason="lost")], [])
    (piece,) = scene.world.pieces
    assert tuple(piece.body.velocity) == (0.0, 0.0)


def test_update_and_draw_run(scene):
    scene.handle([ev("pinch_start", 0.0)], [])
    for _ in range(30):
        scene.update(1 / 60)
        scene.draw(scene.surface)
    scene.handle([ev("pinch_end", 0.6, reason="released")], [])
    y0 = scene.world.pieces[0].body.position.y
    for _ in range(30):
        scene.update(1 / 60)
        scene.draw(scene.surface)
    assert scene.world.pieces[0].body.position.y > y0


def test_clear(scene):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.2, reason="released")], [])
    scene.handle([ev("pinch_start", 0.3)], [])
    scene.clear()
    assert scene.world.pieces == [] and scene.pending == {}


@pytest.mark.parametrize("name,shapes", [("right_pinch_x5", 5), ("right_pinch_hold", 1)])
def test_recorded_pinches_make_shapes(scene, cfg, name, shapes):
    """Real hands: each recorded pinch becomes exactly one shape."""
    path = FIXTURES / f"{name}.jsonl.gz"
    if not path.exists():
        pytest.skip(f"fixture {name} not recorded")
    clip = load_clip(path)
    tracker = HandTracker(cfg.tracking)
    engine = GestureEngine(cfg.gestures, tuple(clip.header["size"]))
    prev = clip.frames[0].t_capture
    for fr in clip.frames:
        events, states = engine.update(tracker.update(fr), fr.t_capture)
        scene.handle(events, states)
        scene.update(fr.t_capture - prev)
        prev = fr.t_capture
    assert len(scene.world.pieces) == shapes
    assert scene.pending == {}


def _replay(scene, cfg, clip, on_frame=None):
    tracker = HandTracker(cfg.tracking)
    engine = GestureEngine(cfg.gestures, tuple(clip.header["size"]))
    prev = clip.frames[0].t_capture
    for fr in clip.frames:
        events, states = engine.update(tracker.update(fr), fr.t_capture)
        scene.handle(events, states)
        scene.update(fr.t_capture - prev)
        prev = fr.t_capture
        if on_frame:
            on_frame(fr, states)


def test_finger_count_fixture_chooses_reliably(scene, cfg):
    """A.2 exit criterion: choice is reliable for 1/3/4/5 fingers on real hands.

    Clip: one hand holds up 1, 2, 3, 4, 5 fingers in turn, Space at each new count.
    """
    path = FIXTURES / "finger_count_1_to_5.jsonl.gz"
    if not path.exists():
        pytest.skip("fixture finger_count_1_to_5 not recorded")
    clip = load_clip(path)
    marks = [m.t for m in clip.marks]
    assert len(marks) == 5, "record with one Space mark per finger count"
    ends = [*marks[1:], clip.frames[-1].t_capture + 0.5]
    seen: dict[int, list] = {n: [] for n in range(1, 6)}

    def record(fr, states):
        vis = [s for s in states if s.visible]
        for n, (start, end) in enumerate(zip(marks, ends, strict=True), 1):
            # Space is pressed ~0.2-0.4 s *after* the new count is shown (reaction time), so
            # skip the first 0.3 s after a mark (still settling) and the last 0.5 s before the
            # next one (the next count is often already up).
            if start + 0.3 <= fr.t_capture < end - 0.5:
                seen[n].append(scene.chooser.stable(vis[0].id) if vis else None)

    _replay(scene, cfg, clip, record)
    for n in (1, 3, 4, 5):
        hits = sum(c == n for c in seen[n]) / max(1, len(seen[n]))
        assert hits >= 0.9, f"{n} fingers chose {n} in only {hits:.0%} of frames"


def test_two_hand_stretch_fixture(scene, cfg):
    """Clip: one hand pinches, the other pinches too and pulls apart, then both let go."""
    path = FIXTURES / "two_hand_stretch.jsonl.gz"
    if not path.exists():
        pytest.skip("fixture two_hand_stretch not recorded")
    radii = []
    _replay(
        scene,
        cfg,
        load_clip(path),
        lambda fr, st: radii.extend(p.radius for p in scene.pending.values() if p.partner),
    )
    assert radii, "the second pinch never joined the first"
    # Moving the hands apart and back must visibly resize the shape (it used to sit at the cap).
    assert (max(radii) - min(radii)) / WINDOW[1] > 0.05
    assert len(scene.world.pieces) == 1, "one stretched shape, not two separate ones"
    # The open left hand chose a pentagon while the right held the shape. On its way into the
    # pinch it read as 2 then 1 finger, which used to turn the shape into a circle.
    assert scene.world.pieces[0].kind == "pentagon"
