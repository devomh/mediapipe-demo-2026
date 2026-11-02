"""Magic Geometry scene: gesture events -> shapes. Driven directly, no threads."""

import math
from dataclasses import replace
from pathlib import Path

import pygame
import pytest
from synthetic import make_hand

from ai_playground.gestures import features
from ai_playground.gestures.engine import GestureEngine, HandState
from ai_playground.physics.world import KINDS
from ai_playground.render import text
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
    text.reset()  # fonts cached by an earlier test's pygame session are invalid now
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


def spread(i):
    """Pinch positions far apart, so a new pinch never lands on (and grabs) an earlier shape."""
    return (0.1 + 0.15 * i, 0.3)


def test_kinds_cycle_and_hues_differ(scene):
    for i in range(6):
        scene.handle(
            [ev("pinch_start", i, pos=spread(i)), ev("pinch_end", i + 0.2, reason="released")], []
        )
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


def state(cfg, hid, fingers=5, pinching=False, handedness="Left", center=(0.25, 0.5), palm=None):
    """A HandState whose features come from a synthetic hand holding up `fingers` fingers.

    `fingers=0` is a fist (fist detector active); `fingers=5` an open palm (palm detector active
    unless `palm=False`).
    """
    if palm is None:
        palm = fingers == 5 and not pinching
    hand = make_hand(handedness=handedness, center=center, **FINGERS[fingers])
    f = features.compute(hand.landmarks, hand.world, (640, 480), cfg.gestures.fingers)
    return HandState(TrackedHand(hid, hand, 10), f, (0.0, 0.0), pinching, fingers == 0, palm)


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
    # The hand that will pinch; a fist, so it doesn't choose anything itself.
    maker = state(cfg, 1, 0, handedness="Right", center=(0.7, 0.5))
    other = state(cfg, 2, n)
    hold(scene, cfg, [maker, other])
    assert scene.selection() == (kind, True)
    scene.handle([ev("pinch_start", 1.0, hand=1)], [maker, other])
    assert scene.pending[1].kind == kind and scene.pending[1].chosen


def test_choice_needs_a_steady_count(scene, cfg):
    maker = state(cfg, 1, 0, handedness="Right", center=(0.7, 0.5))  # a fist chooses nothing

    def show(n, frames):
        hold(scene, cfg, [maker, state(cfg, 2, n)], frames=frames)

    show(3, cfg.geometry.choose_frames - 1)
    assert scene.selection() == ("circle", False)  # not yet: still the automatic "next"
    show(4, 1)  # count changed: starts over
    show(4, cfg.geometry.choose_frames - 2)
    assert scene.selection()[1] is False
    show(4, 1)
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
        scene.handle(
            [ev("pinch_start", i, pos=spread(i)), ev("pinch_end", i + 0.2, reason="released")], []
        )
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


# --- A.3: paddle, grab, clear ---------------------------------------------------------------


def palm_px(scene, st):
    return scene.ctx.viewport.to_px(*st.features.palm_center)


def fist_event(type_, st, t, **data):
    return GestureEvent(type_, st.id, st.handedness, t, st.features.palm_center, data=data)


def arm(scene, cfg, states, extra=2):
    """Hold the given hands (open palms) long enough for them to become paddles."""
    for _ in range(round(cfg.geometry.paddle_hold_s * 60) + extra):
        scene.handle([], states)
        scene.update(1 / 60)


def test_open_palm_arms_into_a_paddle_sized_to_the_hand(scene, cfg):
    open_hand = state(cfg, 1, 5)
    scene.handle([], [open_hand])
    assert 1 not in scene.world.hands and scene.play.arming[1][2] == 0.0  # arming, ring fading in
    arm(scene, cfg, [open_hand])
    assert scene.play.arming == {}
    hb = scene.world.hands[1]
    expected = cfg.geometry.paddle_scale * open_hand.features.hand_scale_image * 960  # dest h
    assert hb.shape is not None and hb.shape.radius == pytest.approx(expected)
    assert hb.target == pytest.approx(palm_px(scene, open_hand))


def test_pinching_or_shape_making_hands_are_not_paddles(scene, cfg):
    scene.handle([], [state(cfg, 1, 5, pinching=True)])
    assert 1 not in scene.world.hands
    scene.handle([ev("pinch_start", 0.0, hand=2)], [state(cfg, 2, 5)])  # pending shape owner
    assert 2 not in scene.world.hands


def test_paddle_bats_a_shape(scene, cfg):
    st = state(cfg, 1, 5, center=(0.3, 0.8))
    x0, y0 = palm_px(scene, st)
    arm(scene, cfg, [st])
    piece = scene.world.spawn("square", (x0 + 120, y0), 30)
    for i in range(30):  # sweep the open hand right through the square
        st = state(cfg, 1, 5, center=(0.3 + 0.01 * (i + 1), 0.8))
        scene.handle([], [st])
        scene.update(1 / 60)
    assert piece.body.velocity.x > 50 or piece.body.position.x > x0 + 200


def test_fist_on_a_shape_grabs_carries_and_throws_it(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.4, 0.5))
    piece = scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    assert scene.play.held == {1: piece} and scene.play.grabs == 1
    for i in range(20):  # carry it right
        fist = state(cfg, 1, 0, center=(0.4 + 0.01 * (i + 1), 0.5))
        scene.handle([], [fist])
        scene.update(1 / 60)
    assert piece.body.position.x > palm_px(scene, state(cfg, 1, 0, center=(0.5, 0.5)))[0] - 60
    opened = state(cfg, 1, 5, center=(0.6, 0.5))
    scene.handle([fist_event("fist_end", opened, 1.0, reason="released")], [opened])
    assert scene.play.held == {} and 1 in scene.play.arming  # open again: arming a paddle
    assert piece.body.velocity.x > 100  # thrown with the carry velocity


def test_fist_away_from_shapes_grabs_nothing(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.2, 0.3))
    x, y = palm_px(scene, fist)
    scene.world.spawn("square", (x + 400, y), 30)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    assert scene.play.held == {}


def test_grab_picks_the_nearest_shape(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.5, 0.5))
    x, y = palm_px(scene, fist)
    scene.world.spawn("circle", (x + 50, y), 30)
    near = scene.world.spawn("circle", (x - 35, y), 30)  # palm is inside this one
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    assert scene.play.held == {1: near}


def test_lost_fist_drops_without_throwing(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.4, 0.5))
    piece = scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    for i in range(10):
        fist = state(cfg, 1, 0, center=(0.4 + 0.02 * (i + 1), 0.5))
        scene.handle([], [fist])
        scene.update(1 / 60)
    scene.handle([fist_event("fist_end", fist, 1.0, reason="lost")], [])
    assert tuple(piece.body.velocity) == (0.0, 0.0)
    assert 1 not in scene.world.hands  # the hand is gone from the world too


def test_both_fists_held_clear_once(scene, cfg):
    for i in range(5):
        scene.world.spawn("circle", (200 + 150 * i, 300), 30)
    fists = [state(cfg, 1, 0, center=(0.3, 0.5)), state(cfg, 2, 0, center=(0.7, 0.5))]
    steps = round(cfg.geometry.clear_hold_s * 60)
    for _ in range(steps - 5):
        scene.handle([], fists)
        scene.update(1 / 60)
    assert len(scene.world.pieces) == 5 and 0.9 < scene.play.clear_progress < 1.0
    for _ in range(10):
        scene.handle([], fists)
        scene.update(1 / 60)
    assert scene.world.pieces == []
    scene.world.spawn("square", (640, 300), 30)  # keep holding: no second clear
    for _ in range(2 * steps):
        scene.handle([], fists)
        scene.update(1 / 60)
    assert len(scene.world.pieces) == 1


def test_one_fist_or_a_brief_double_fist_does_not_clear(scene, cfg):
    scene.world.spawn("circle", (640, 300), 30)
    one = [state(cfg, 1, 0, center=(0.3, 0.5)), state(cfg, 2, 5, center=(0.7, 0.5))]
    for _ in range(200):
        scene.handle([], one)
        scene.update(1 / 60)
    assert len(scene.world.pieces) == 1 and scene.play.clear_progress == 0.0
    both = [state(cfg, 1, 0, center=(0.3, 0.5)), state(cfg, 2, 0, center=(0.7, 0.5))]
    for _ in range(30):
        scene.handle([], both)
        scene.update(1 / 60)
    scene.handle([], one)  # one hand opens: progress resets on the next frame
    scene.update(1 / 60)
    assert scene.play.clear_progress == 0.0 and len(scene.world.pieces) == 1


def test_hand_leaving_releases_its_shape(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.4, 0.5))
    piece = scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    scene.handle([], [])  # tracker dropped it without a fist_end (e.g. straight to exit)
    assert scene.play.held == {} and scene.world.hands == {}
    assert piece in scene.world.pieces and len(scene.world.space.constraints) == 0


def test_hands_draw(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.4, 0.5))
    scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist, state(cfg, 2, 5, center=(0.7, 0.5))])
    scene.handle([], [fist, state(cfg, 2, 0, center=(0.7, 0.5))])
    for _ in range(10):
        scene.update(1 / 60)
        scene.draw(scene.surface)


def test_throw_uses_the_peak_velocity_before_the_late_fist_end(scene, cfg):
    """The hand swings fast, then slows while opening; fist_end only arrives after that."""
    fist = state(cfg, 1, 0, center=(0.2, 0.5))
    piece = scene.world.spawn("circle", palm_px(scene, fist), 30)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    x = 0.2
    for dx in [0.03] * 12 + [0.003] * 5:  # fast swing, then ~85 ms nearly still (opening)
        x += dx
        fist = state(cfg, 1, 0, center=(x, 0.5))
        scene.handle([], [fist])
        scene.update(1 / 60)
    slow = piece.body.velocity.x
    opened = state(cfg, 1, 5, center=(x, 0.5))
    scene.handle([fist_event("fist_end", opened, 1.0, reason="released")], [opened])
    assert piece.body.velocity.x > 2 * slow  # the swing speed, not the slowed-down one


def test_grab_and_throw_fixture(scene, cfg):
    """Clip: pinch to make a shape and drop it, grab it with a fist, carry it, throw it."""
    path = FIXTURES / "grab_and_throw.jsonl.gz"
    if not path.exists():
        pytest.skip("fixture grab_and_throw not recorded")
    # The clip is cropped from a longer session where this was the second shape, so the
    # automatic cycle made it a bar; the grab aims at the bar's end. A crop can't carry that
    # scene state, so start the cycle where it was.
    scene._next_kind = KINDS.index("bar")
    _replay(scene, cfg, load_clip(path))
    assert scene.play.grabs >= 1, "the fist never grabbed the shape"
    assert scene.play.throws, "the grabbed shape was never released"
    # A real throw, not a drop: at least 1 window height per second.
    assert max(scene.play.throws) > WINDOW[1]


def test_paddle_needs_mediapipe_open_palm_not_just_five_fingers(scene, cfg):
    """Right after a pinch, the half-open hand already counts 5 fingers; that's not a paddle."""
    arm(scene, cfg, [state(cfg, 1, 5, palm=False)])
    assert 1 not in scene.world.hands and scene.play.arming == {}
    arm(scene, cfg, [state(cfg, 1, 5, palm=True)])
    assert scene.world.hands[1].shape is not None


def test_no_paddle_right_after_the_hand_ends_a_pinch(scene, cfg):
    """The released shape appears at the hand; a paddle there would bat it away."""
    scene.handle([ev("pinch_start", 0.0, hand=1)], [state(cfg, 1, 5, pinching=True)])
    scene.handle([ev("pinch_end", 0.5, hand=1, reason="released")], [state(cfg, 1, 5)])
    assert 1 not in scene.world.hands
    for _ in range(round(cfg.geometry.paddle_after_pinch_s * 60) - 2):
        scene.update(1 / 60)
        scene.handle([], [state(cfg, 1, 5)])
    assert 1 not in scene.world.hands and scene.play.arming == {}  # not even arming yet
    arm(scene, cfg, [state(cfg, 1, 5)], extra=6)
    assert scene.world.hands[1].shape is not None


def test_pinch_on_a_shape_grabs_it_instead_of_making_one(scene, cfg):
    pos = (0.5, 0.4)
    piece = scene.world.spawn("square", scene.ctx.viewport.to_px(*pos), 50)
    pinching = state(cfg, 1, 5, pinching=True, center=(0.5, 0.5))
    scene.handle([ev("pinch_start", 0.0, pos=pos)], [pinching])
    assert scene.pending == {} and scene.play.held == {1: piece}
    for i in range(15):  # carry it right by the pinch point
        p = (pos[0] + 0.01 * (i + 1), pos[1])
        moved = replace(pinching, features=replace(pinching.features, pinch_point=p))
        scene.handle([ev("pinch_move", 0.1 * i, pos=p)], [moved])
        scene.update(1 / 60)
    scene.handle([ev("pinch_end", 2.0, pos=p, reason="released")], [state(cfg, 1, 5, palm=False)])
    assert scene.play.held == {} and len(scene.world.pieces) == 1  # no new shape was made
    assert piece.body.velocity.x > 100 and scene.play.throws  # thrown by the pinch release


def test_pinch_near_but_outside_reach_still_makes_a_shape(scene, cfg):
    h = WINDOW[1]
    x, y = scene.ctx.viewport.to_px(0.5, 0.4)
    scene.world.spawn("circle", (x, y), 40)
    far = scene.ctx.viewport.to_norm(x + 40 + (cfg.geometry.pinch_grab_reach + 0.02) * h, y)
    scene.handle([ev("pinch_start", 0.0, pos=far)], [])
    assert 1 in scene.pending and scene.play.held == {}


def test_stretch_takes_priority_over_pinch_grab(scene, cfg):
    """While one hand holds a new shape, the other hand's pinch stretches it, even on a shape."""
    other = scene.world.spawn("square", scene.ctx.viewport.to_px(0.7, 0.4), 50)
    scene.handle([ev("pinch_start", 0.0, hand=1, pos=(0.3, 0.4))], [])
    scene.handle([ev("pinch_start", 0.1, hand=2, pos=(0.7, 0.4))], [])
    assert scene.pending[1].partner == 2 and scene.play.held == {}
    assert other in scene.world.pieces


def test_closing_the_palm_restarts_paddle_arming(scene, cfg):
    open_hand = state(cfg, 1, 5)
    for _ in range(round(cfg.geometry.paddle_hold_s * 60) - 4):
        scene.handle([], [open_hand])
        scene.update(1 / 60)
    scene.handle([], [state(cfg, 1, 0)])  # a fist in between
    scene.update(1 / 60)
    scene.handle([], [open_hand])
    assert scene.play.arming[1][2] < 0.1  # starts over


# --- A.4: particles, glow, cursors ----------------------------------------------------------


def test_hard_impact_sparks_but_resting_shapes_do_not(scene, cfg):
    piece = scene.world.spawn("square", (640, 100), 40, vel=(0, 1500))
    sparked = 0
    for _ in range(60):
        scene.update(1 / 60)
        sparked = max(sparked, scene.particles.alive)
    assert sparked > 0  # it hit the floor hard
    for _ in range(240):  # let it settle and the sparks die out
        scene.update(1 / 60)
    assert scene.particles.alive == 0 and piece.body.is_sleeping


def test_release_and_clear_emit_particles(scene, cfg):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.5, reason="released")], [])
    assert scene.particles.alive == cfg.effects.release_particles
    scene.particles.clear()
    for i in range(3):
        scene.world.spawn("circle", (300 + 200 * i, 300), 30)
    scene._burst()  # 3 new shapes plus the one the pinch made
    assert scene.particles.alive == 4 * cfg.effects.burst_particles
    assert scene.world.pieces == []


def test_every_cursor_state_draws(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.3, 0.5))
    scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])  # holding: grab ring
    arming = state(cfg, 2, 5, center=(0.7, 0.5))
    pinching = state(cfg, 3, 5, pinching=True, center=(0.5, 0.3))
    idle = state(cfg, 4, 2, center=(0.5, 0.7))
    hands = [fist, arming, pinching, idle]
    for _ in range(40):  # long enough for hand 2 to arm into a paddle
        scene.handle([], hands)
        scene.update(1 / 60)
        scene.draw(scene.surface)
    assert scene.world.hands[2].shape is not None


def test_geometry_uses_a_gradient_without_camera(scene):
    assert scene.ctx.background.empty == "gradient"
    scene.draw(scene.surface)
    top, bottom = scene.surface.get_at((640, 2)), scene.surface.get_at((640, 717))
    assert sum(top[:3]) > sum(bottom[:3])  # lighter at the top


def test_a_lone_hand_is_not_shown_as_choosing(scene, cfg):
    """With one hand, that hand will make the next pinch, so the palette shows the cycle."""
    hold(scene, cfg, [state(cfg, 1, 5)])
    assert scene.selection() == ("circle", False)
    hold(scene, cfg, [state(cfg, 1, 5), state(cfg, 2, 3, center=(0.7, 0.5))])
    assert scene.selection()[1] is True  # two hands: one of them may be choosing


def test_effects_toggle_hides_glow_and_particles_but_not_feedback(scene, cfg):
    assert scene.effects_on is cfg.effects.enabled
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.5, reason="released")], [])
    assert scene.particles.alive > 0
    scene.effects_on = False
    assert scene.particles.alive == 0  # cleared, nothing stale when switched back on
    scene.handle([ev("pinch_start", 1.0, pos=(0.2, 0.3))], [])
    scene.handle([ev("pinch_end", 1.5, pos=(0.2, 0.3), reason="released")], [])
    scene._burst()
    assert scene.particles.alive == 0  # no emission while off
    scene.handle([ev("pinch_start", 2.0, pos=(0.6, 0.3))], [])
    scene.draw(scene.surface)
    assert scene.pending  # previews and shapes still work
    scene.effects_on = True
    scene.handle([ev("pinch_end", 2.5, pos=(0.6, 0.3), reason="released")], [])
    assert scene.particles.alive == cfg.effects.release_particles


# --- A.5: sound effects ---------------------------------------------------------------------


class FakeSfx:
    """Records what the scene asks to play."""

    def __init__(self):
        self.played: list[tuple[str, float, float]] = []
        self.inflating = False
        self.inflate_starts = 0

    def play(self, name, volume=1.0, x=0.5):
        self.played.append((name, volume, x))

    def start_inflate(self, x=0.5):
        self.inflating = True
        self.inflate_starts += 1

    def stop_inflate(self):
        self.inflating = False

    def names(self):
        return [n for n, _, _ in self.played]

    def impacts(self):
        return [p for p in self.played if p[0].startswith("impact.")]


@pytest.fixture
def sounds(scene):
    fake = FakeSfx()
    scene.ctx.sfx = fake
    return fake


def test_pinch_pops_inflates_and_a_slow_release_just_drops(scene, sounds):
    scene.handle([ev("pinch_start", 0.0, pos=(0.2, 0.5))], [])
    assert sounds.names() == ["spawn"] and sounds.played[0][2] < 0.5  # panned left
    scene.update(1 / 60)
    assert sounds.inflating and sounds.inflate_starts == 1
    scene.handle([ev("pinch_end", 0.5, vel=(0.01, 0.0), reason="released")], [])
    scene.update(1 / 60)
    assert not sounds.inflating and "throw" not in sounds.names()


def test_a_fast_release_whooshes(scene, sounds):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.3, vel=(1.0, -0.5))], [])
    (whoosh,) = [p for p in sounds.played if p[0] == "throw"]
    assert 0.4 < whoosh[1] <= 1.0


def test_a_lost_hand_drops_silently(scene, sounds):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.3, vel=(2.0, 0.0), reason="lost")], [])
    assert "throw" not in sounds.names()


def test_stretching_stops_the_inflate_tone(scene, sounds):
    scene.handle([ev("pinch_start", 0.0, hand=1, pos=(0.3, 0.4))], [])
    scene.update(1 / 60)
    assert sounds.inflating
    scene.handle([ev("pinch_start", 0.1, hand=2, pos=(0.7, 0.4))], [])
    scene.update(1 / 60)
    assert not sounds.inflating and sounds.names().count("spawn") == 1


def test_grabs_click_and_throws_from_a_grab_whoosh(scene, cfg, sounds):
    fist = state(cfg, 1, 0, center=(0.4, 0.5))
    scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    assert sounds.names() == ["grab"]
    for i in range(20):
        fist = state(cfg, 1, 0, center=(0.4 + 0.015 * (i + 1), 0.5))
        scene.handle([], [fist])
        scene.update(1 / 60)
    opened = state(cfg, 1, 5, center=(0.7, 0.5))
    scene.handle([fist_event("fist_end", opened, 1.0, reason="released")], [opened])
    assert "throw" in sounds.names() and "spawn" not in sounds.names()


def test_pinch_grab_clicks_instead_of_popping(scene, cfg, sounds):
    pos = (0.5, 0.4)
    scene.world.spawn("square", scene.ctx.viewport.to_px(*pos), 50)
    scene.handle([ev("pinch_start", 0.0, pos=pos)], [state(cfg, 1, 5, pinching=True)])
    assert sounds.names() == ["grab"]


def test_two_fist_clear_sweeps(scene, cfg, sounds):
    scene.world.spawn("circle", (640, 300), 30)
    fists = [state(cfg, 1, 0, center=(0.3, 0.5)), state(cfg, 2, 0, center=(0.7, 0.5))]
    for _ in range(round(cfg.geometry.clear_hold_s * 60) + 5):
        scene.handle([], fists)
        scene.update(1 / 60)
    assert sounds.names().count("clear") == 1


def test_a_hard_hit_sounds_with_the_shapes_timbre_and_size(scene, cfg, sounds):
    scene.world.spawn("triangle", (300, 100), 0.03 * WINDOW[1], vel=(0, 1500))
    for _ in range(60):
        scene.update(1 / 60)
    hits = sounds.impacts()
    assert hits and hits[0][0] == "impact.triangle.0"  # smallest size: highest pitch
    assert hits[0][1] == 1.0 and hits[0][2] < 0.5  # a hard hit, on the left


def test_resting_shapes_are_quiet(scene, cfg, sounds):
    piece = scene.world.spawn("square", (640, 600), 40)
    for _ in range(240):
        scene.update(1 / 60)
    assert piece.body.is_sleeping
    sounds.played.clear()
    for _ in range(120):
        scene.update(1 / 60)
    assert sounds.played == []


def test_a_collapsing_pile_is_capped_and_then_falls_silent(scene, cfg, sounds):
    kinds = KINDS * 8
    for i, kind in enumerate(kinds):  # 40 shapes dropped in a grid
        x, y = 200 + 110 * (i % 8), 60 + 90 * (i // 8)
        scene.world.spawn(kind, (x, y), 30 + 3 * (i % 5))
    per_frame = []
    for _ in range(6 * 60):
        before = len(sounds.impacts())
        scene.update(1 / 60)
        per_frame.append(len(sounds.impacts()) - before)
    assert max(per_frame) <= cfg.sound.max_impacts_per_frame
    assert sum(per_frame[:120]) > 0  # the pile landing is heard...
    assert sum(per_frame[-60:]) == 0  # ...and once it settles, it is quiet


# --- A.6: hints and idle reset --------------------------------------------------------------
from ai_playground.scenes.geometry import CLEAR_HINT_AT, HINTS  # noqa: E402


def test_hints_follow_what_the_visitor_has_done(scene, cfg):
    assert scene.hint() == HINTS["create"]
    scene.handle([ev("pinch_start", 0.0)], [])
    assert scene.hint() == HINTS["throw"]  # a shape in hand
    scene.handle([ev("pinch_end", 0.5, vel=(0.01, 0.0), reason="released")], [])
    assert scene.hint() == HINTS["choose"]  # a slow drop doesn't count as a throw
    far = (0.15, 0.2)  # away from the dropped shape, or the pinch would grab it
    scene.handle(
        [ev("pinch_start", 1.0, pos=far), ev("pinch_end", 1.3, pos=far, vel=(1.0, -0.5))], []
    )
    assert "throw" in scene.hints.done
    maker = state(cfg, 1, 0, handedness="Right", center=(0.7, 0.5))
    other = state(cfg, 2, 3)
    hold(scene, cfg, [maker, other])
    scene.handle([ev("pinch_start", 2.0, hand=1, pos=(0.8, 0.2))], [maker, other])
    scene.handle([ev("pinch_end", 2.3, hand=1, pos=(0.8, 0.2), reason="released")], [])
    assert scene.hint() == HINTS["grab"]
    scene.hints.mark("grab")
    assert scene.hint() == HINTS["push"]
    arm(scene, cfg, [state(cfg, 3, 5, center=(0.5, 0.9))])
    assert scene.hint() is None  # under CLEAR_HINT_AT shapes: nothing left to teach


def test_clear_hint_needs_a_full_screen(scene):
    scene.hints.done |= {"create", "throw", "choose", "grab", "push"}
    for i in range(CLEAR_HINT_AT):
        scene.world.spawn("circle", (100 + 100 * i, 300), 30)
    assert scene.hint() == HINTS["clear"]
    scene._burst()
    assert scene.hint() is None


def test_grab_marks_its_hint(scene, cfg):
    fist = state(cfg, 1, 0, center=(0.4, 0.5))
    scene.world.spawn("square", palm_px(scene, fist), 40)
    scene.handle([fist_event("fist_start", fist, 0.0)], [fist])
    assert "grab" in scene.hints.done


def test_idle_reset_fades_shapes_and_brings_hints_back(scene, cfg, sounds):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.3, vel=(1.0, -0.5))], [])
    scene.handle([ev("pinch_start", 1.0, hand=2, pos=(0.15, 0.2))], [])  # one being made
    scene.update(1 / 60)
    assert scene.hints.done and scene.pending and sounds.inflating
    scene.idle_reset()
    assert scene.pending == {} and not sounds.inflating
    assert scene.hints.done == set() and scene.hint() == HINTS["create"]
    assert scene.world.pieces and all(p.fade is not None for p in scene.world.pieces)
    for _ in range(round(cfg.physics.fade_s * 60) + 5):
        scene.update(1 / 60)
    assert scene.world.pieces == []
    scene.draw(scene.surface)
