"""Air Writing scene, W.1 (spec §2, §3, §11): ink, eraser, tail trim, read ring, clear, idle."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest
from test_geometry import state

from ai_playground import config
from ai_playground.gestures.engine import GestureEngine
from ai_playground.render import effects, text
from ai_playground.render.background import Background
from ai_playground.render.viewport import Viewport
from ai_playground.scenes.base import AppContext
from ai_playground.scenes.writing import HINTS, NO_INK, NUDGE_S, TOO_SMALL, WritingScene
from ai_playground.sources.replay import load_clip
from ai_playground.tracking.identity import HandTracker
from ai_playground.types import GestureEvent

WINDOW = (1280, 720)
DT = 1 / 60


@pytest.fixture
def scene(cfg):
    pygame.init()
    text.reset()
    surface = pygame.display.set_mode(WINDOW)
    ctx = AppContext(
        config=cfg,
        viewport=Viewport(WINDOW, (640, 480)),
        background=Background(),
        switch_scene=lambda name: None,
    )
    s = WritingScene()
    s.enter(ctx)
    s.surface = surface
    yield s
    pygame.quit()


def ev(type_, t, pos=(0.5, 0.5), hand=1, **data):
    return GestureEvent(type_, hand, "Right", t, pos, data=data)


def write(scene, a, b, t0=0.0, frames=20, hand=1, end=True):
    """Pinch at `a`, move to `b` over `frames` camera frames (30 FPS), optionally let go.
    Returns the time of the last event."""
    scene.handle([ev("pinch_start", t0, pos=a, hand=hand)], [])
    t = t0
    for i in range(1, frames + 1):
        f = i / frames
        t = t0 + i / 30
        pos = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        scene.handle([ev("pinch_move", t, pos=pos, hand=hand)], [])
    if end:
        t += 1 / 30
        scene.handle([ev("pinch_end", t, pos=b, hand=hand, reason="released")], [])
    return t


def hold(scene, states, seconds):
    for _ in range(round(seconds / DT)):
        scene.handle([], states)
        scene.update(DT)


def test_pinch_draws_evenly_spaced_ink_in_the_hand_colour(scene, cfg):
    write(scene, (0.5, 0.3), (0.5, 0.7), end=False)
    b, color = scene.active[1]
    step = cfg.writing.resample_step * WINDOW[1]
    d = np.linalg.norm(np.diff(b.points, axis=0), axis=1)
    assert np.allclose(d, step, atol=1e-3) and color == effects.hand_color(1)
    scene.handle([ev("pinch_end", 1.0, pos=(0.5, 0.7), reason="released")], [])
    assert scene.active == {} and len(scene.ink) == 1


def test_letting_go_is_pen_up_not_done(scene):
    t = write(scene, (0.4, 0.7), (0.5, 0.3))
    t = write(scene, (0.5, 0.3), (0.6, 0.7), t0=t + 0.5)
    write(scene, (0.45, 0.55), (0.55, 0.55), t0=t + 0.5)  # an "A" in three strokes
    assert len(scene.ink) == 3 and scene.reads == []


def ratio(hid, r):
    """Just a pinch ratio for the pen-up rule; not visible, so no cursor or eraser."""
    return SimpleNamespace(id=hid, visible=False, features=SimpleNamespace(pinch_ratio=r))


def test_letting_go_drops_the_drag_of_opening_fingers(scene, cfg):
    """Opening fingers move the pinch point in the last frames before pinch_end."""
    scene.handle([ev("pinch_start", 0.0, pos=(0.5, 0.3))], [ratio(1, 0.2)])
    for i in range(1, 21):  # straight down, firmly pinched
        scene.handle([ev("pinch_move", i / 30, pos=(0.5, 0.3 + 0.02 * i))], [ratio(1, 0.2)])
    for i in range(1, 4):  # opening: the pinch point slides right
        scene.handle([ev("pinch_move", (20 + i) / 30, pos=(0.5 + 0.03 * i, 0.7))], [ratio(1, 0.45)])
    scene.handle([ev("pinch_end", 24 / 30, pos=(0.6, 0.7), reason="released")], [ratio(1, 0.55)])
    (piece,) = scene.ink
    x = piece.stroke.points[:, 0]
    assert x.max() - x.min() < 1.0  # no hook to the right
    assert piece.stroke.times[-1] == pytest.approx(20 / 30, abs=0.01)  # last firm frame


def test_a_loose_pinch_keeps_most_of_its_ink(scene, cfg):
    scene.handle([ev("pinch_start", 0.0, pos=(0.5, 0.3))], [ratio(1, 0.2)])
    for i in range(1, 31):  # the whole stroke held loosely, above release_ratio
        scene.handle([ev("pinch_move", i / 30, pos=(0.5, 0.3 + 0.015 * i))], [ratio(1, 0.45)])
    scene.handle([ev("pinch_end", 31 / 30, reason="released")], [ratio(1, 0.55)])
    (piece,) = scene.ink
    lost = 30 / 30 - piece.stroke.times[-1]  # from the last ink point
    assert lost <= cfg.writing.release_trim_max_s + 0.01


def test_a_fist_after_letting_go_trims_nothing_more(scene, cfg):
    t = write(scene, (0.5, 0.3), (0.5, 0.7))
    full = scene.ink[0].stroke
    scene.handle([ev("fist_start", t + 0.05)], [])
    assert scene.ink[0].stroke is full


def test_a_fist_while_pinching_ends_and_trims_the_stroke_and_no_ink_follows(scene, cfg):
    t = write(scene, (0.5, 0.3), (0.5, 0.7), end=False)
    last = scene.active[1][0].points[-1]
    scene.handle([ev("fist_start", t + 1 / 30)], [])
    assert scene.active == {} and len(scene.ink) == 1
    stroke = scene.ink[0].stroke
    assert stroke.times[-1] <= t - cfg.writing.tail_trim_s + 1e-9
    assert stroke.points[-1][1] < last[1] - 10  # the curl's hook is gone
    n = len(stroke)
    for i in range(10):  # the pinch detector may still report the pinch
        scene.handle([ev("pinch_move", t + (i + 2) / 30, pos=(0.7, 0.9))], [])
    scene.handle([ev("pinch_end", t + 0.5, reason="released")], [])
    assert len(scene.ink) == 1 and len(scene.ink[0].stroke) == n


def test_one_held_fist_reads_the_ink(scene, cfg):
    write(scene, (0.5, 0.3), (0.5, 0.7))
    fist = state(cfg, 1, 0)
    hold(scene, [fist], cfg.writing.read_hold_s - 0.1)
    assert scene.reads == [] and 0 < scene.read_hold.progress < 1
    hold(scene, [fist], 0.15)
    assert len(scene.reads) == 1 and scene.ink == [] and scene.fading
    assert "read" in scene.hints.done
    hold(scene, [fist], 1.0)  # keep holding: one read per fist
    assert len(scene.reads) == 1 and scene.fading == []


def test_reading_nothing_or_something_tiny_nudges(scene, cfg):
    fist = state(cfg, 1, 0)
    hold(scene, [fist], cfg.writing.read_hold_s + 0.05)
    assert scene.reads == [] and scene.hint() == NO_INK
    hold(scene, [], NUDGE_S)
    assert scene.hint() == HINTS["write"]
    write(scene, (0.5, 0.5), (0.5, 0.52))
    hold(scene, [fist], cfg.writing.read_hold_s + 0.05)
    assert scene.reads == [] and scene.hint() == TOO_SMALL and len(scene.ink) == 1


def test_two_fists_start_over_without_a_read(scene, cfg):
    write(scene, (0.5, 0.3), (0.5, 0.7))
    fists = [state(cfg, 1, 0, center=(0.3, 0.5)), state(cfg, 2, 0, center=(0.7, 0.5))]
    hold(scene, fists, cfg.geometry.clear_hold_s - 0.1)
    assert len(scene.ink) == 1 and scene.read_hold.progress == 0
    hold(scene, fists, 0.2)
    assert scene.ink == [] and scene.reads == [] and "clear" in scene.hints.done
    assert "read" not in scene.hints.done


def eraser_setup(scene, cfg):
    hand = state(cfg, 2, 5, center=(0.5, 0.5))
    cx, cy = hand.features.palm_center
    write(scene, (cx - 0.3, cy), (cx + 0.3, cy), hand=1)
    return hand


def test_open_palm_arms_then_erases_and_splits_the_stroke(scene, cfg):
    hand = eraser_setup(scene, cfg)
    hold(scene, [hand], cfg.geometry.paddle_hold_s - 0.1)
    assert len(scene.ink) == 1 and 0 < scene.erasers[2][2] < 1  # arming: ring fading in
    hold(scene, [hand], 0.2)
    assert len(scene.ink) == 2 and scene.erasers[2][2] == 1.0
    assert {i.color for i in scene.ink} == {effects.hand_color(1)}  # fragments keep their colour
    assert "erase" in scene.hints.done


def test_no_eraser_right_after_the_hand_ends_a_pinch(scene, cfg):
    hand = state(cfg, 1, 5, center=(0.5, 0.5))
    cx, cy = hand.features.palm_center
    write(scene, (cx - 0.3, cy), (cx + 0.3, cy), hand=1)
    hold(scene, [hand], cfg.geometry.paddle_after_pinch_s - 0.1)
    assert 1 not in scene.erasers
    hold(scene, [hand], 0.15 + cfg.geometry.paddle_hold_s)
    assert len(scene.ink) == 2  # later, a deliberate open hand erases


def test_hints_follow_the_visitor(scene, cfg):
    assert scene.hint() == HINTS["write"]
    write(scene, (0.5, 0.3), (0.5, 0.7))
    assert scene.hint() == HINTS["read"]
    hold(scene, [state(cfg, 1, 0)], cfg.writing.read_hold_s + 0.05)
    assert scene.hint() is None  # no ink: nothing to say
    write(scene, (0.5, 0.3), (0.5, 0.7), t0=5.0)
    assert scene.hint() == HINTS["erase"]
    scene.reads += [[], []]
    scene.hints.mark("erase")
    assert scene.hint() == HINTS["clear"]


def test_idle_reset_and_c_clear_everything(scene, cfg):
    write(scene, (0.5, 0.3), (0.5, 0.7))
    write(scene, (0.3, 0.3), (0.3, 0.7), t0=2.0, hand=2, end=False)
    scene.idle_reset()
    assert scene.ink == [] and scene.active == {} and scene.reads == []
    assert scene.hints.done == set() and scene.fading
    write(scene, (0.5, 0.3), (0.5, 0.7), t0=4.0)
    scene.clear()
    assert scene.ink == []


def test_draw_every_state(scene, cfg):
    write(scene, (0.5, 0.3), (0.5, 0.7))
    hold(scene, [state(cfg, 1, 0)], cfg.writing.read_hold_s + 0.05)  # a read: thumbnail
    write(scene, (0.4, 0.3), (0.4, 0.7), t0=3.0)
    write(scene, (0.6, 0.3), (0.6, 0.7), t0=4.0, hand=3, end=False)
    hands = [
        state(cfg, 2, 5, center=(0.3, 0.5)),
        state(cfg, 4, 0, center=(0.7, 0.5)),
        state(cfg, 5, 2, center=(0.5, 0.2)),
    ]
    hold(scene, hands, 0.2)
    before = pygame.surfarray.array3d(scene.surface).sum()
    scene.draw(scene.surface)
    assert pygame.surfarray.array3d(scene.surface).sum() != before
    scene.effects_on = False
    scene.draw(scene.surface)
    hold(scene, [hands[1], state(cfg, 6, 0, center=(0.2, 0.5))], 0.1)  # two fists: clear rings
    scene.draw(scene.surface)


# --- the air_write_digits fixture: real hands ------------------------------------------------
FIXTURE = Path(__file__).parent / "fixtures" / "air_write_digits.jsonl.gz"


def replay(cfg):
    """Replay the fixture through the real tracker and engine. Returns the scene, every read
    character, and a log of what happened."""
    pygame.init()
    text.reset()
    pygame.display.set_mode(WINDOW)
    clip = load_clip(FIXTURE)
    size = tuple(clip.header["size"])
    ctx = AppContext(cfg, Viewport(WINDOW, size), Background(), lambda name: None)
    scene = WritingScene()
    scene.enter(ctx)
    tracker, engine = HandTracker(cfg.tracking), GestureEngine(cfg.gestures, size)
    reads, log = [], {"erases": 0, "pen_down_in_a_fist": 0, "clears": 0}
    prev = clip.frames[0].t_capture
    for fr in clip.frames:
        events, states = engine.update(tracker.update(fr), fr.t_capture)
        n_ink, n_reads = len(scene.ink), len(scene.reads)
        scene.handle(events, states)
        if len(scene.ink) < n_ink:
            log["erases"] += 1
        if {s.id for s in states if s.fist} & set(scene.active):
            log["pen_down_in_a_fist"] += 1
        scene.update(fr.t_capture - prev)
        prev = fr.t_capture
        if len(scene.reads) > n_reads:
            reads.append(scene.reads[-1])
        elif len(scene.reads) < n_reads:
            log["clears"] += 1
    return scene, reads, log


def end_travel(stroke, h=WINDOW[1], window_s=0.1):
    """Path length of the stroke's last `window_s`, in window heights: a release hook is fast
    (0.03-0.08 in ~0.1 s), a real stroke slows down as it ends."""
    k = stroke.times >= stroke.times[-1] - window_s
    return float(np.linalg.norm(np.diff(stroke.points[k], axis=0), axis=1).sum() / h)


def test_fixture_reads_five_characters_erases_once_and_clears(cfg):
    try:
        scene, reads, log = replay(cfg)
    finally:
        pygame.quit()
    assert [len(r) for r in reads] == [1, 1, 1, 1, 2]  # 1, 4, 7, A, E
    assert log["erases"] == 1  # the deliberate erase, never between the E's strokes
    assert log["pen_down_in_a_fist"] == 0
    assert log["clears"] == 1 and scene.reads == [] and scene.ink == []  # no stray read


def test_fixture_strokes_end_without_hooks(cfg):
    try:
        _, reads, _ = replay(cfg)
        raw_cfg = config.load(
            overrides={"writing": {"release_trim_max_s": 0.0, "tail_trim_s": 0.0}}
        )
        _, raw, _ = replay(raw_cfg)
    finally:
        pygame.quit()
    trimmed = [end_travel(s) for ch in reads for s in ch]
    untrimmed = [end_travel(s) for ch in raw for s in ch]
    assert max(trimmed) <= 0.03, trimmed
    assert sum(t > 0.03 for t in untrimmed) >= 4, untrimmed  # the fixture does have hooks
