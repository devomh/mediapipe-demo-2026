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
