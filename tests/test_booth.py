"""Booth-minimum shell (Phase A spec §6.0): idle reset, hint line, hold-Esc quit."""

import pygame
import pytest
from synthetic import frame, make_hand

from ai_playground import config
from ai_playground.app import App, RunOptions
from ai_playground.booth import Hints, HoldKey, IdleTimer
from ai_playground.render import text
from ai_playground.render.booth import HintLine, draw_quit_hold
from ai_playground.scenes.vision import VisionScene
from ai_playground.sources.recorder import Recorder
from ai_playground.sources.replay import ReplaySource

DT = 1 / 60


def run_for(update, seconds, *args):
    fired = 0
    for _ in range(round(seconds / DT)):
        fired += bool(update(DT, *args))
    return fired


# --- idle timer ---------------------------------------------------------------------------------
def test_idle_fires_once_per_idle_spell():
    idle = IdleTimer(20.0)
    assert run_for(idle.update, 19.9, False) == 0
    assert run_for(idle.update, 30.0, False) == 1  # once, not every frame after
    assert run_for(idle.update, 0.1, True) == 0  # someone comes back...
    assert run_for(idle.update, 21.0, False) == 1  # ...and leaves again


def test_a_hand_restarts_the_idle_count():
    idle = IdleTimer(1.0)
    run_for(idle.update, 0.9, False)
    idle.update(DT, True)
    assert run_for(idle.update, 0.9, False) == 0


def test_idle_timeout_zero_never_fires():
    assert run_for(IdleTimer(0.0).update, 100.0, False) == 0


# --- hold to quit -------------------------------------------------------------------------------
def test_holding_completes_after_hold_s():
    k = HoldKey(2.0)
    assert not k.update(DT)  # nothing pressed
    k.press()
    assert run_for(k.update, 1.9) == 0 and 0.9 < k.progress < 1.0
    assert run_for(k.update, 0.2) > 0


def test_a_tap_cancels_and_shows_a_reminder():
    k = HoldKey(2.0)
    k.press()
    run_for(k.update, 0.5)
    k.press()  # a repeat key-down doesn't restart the count
    assert k.held_s == pytest.approx(0.5, abs=DT)
    k.release()
    assert k.progress == 0.0 and k.reminder_s == HoldKey.REMINDER_S
    assert run_for(k.update, 5.0) == 0 and k.reminder_s == 0.0


# --- hints --------------------------------------------------------------------------------------
def test_hints_in_order_skipping_done_and_inapplicable():
    h = Hints({"a": "A", "b": "B", "c": "C"})
    assert h.first("abc") == "A"
    assert h.first("bc") == "B"
    h.mark("b")
    assert h.first("bc") == "C"
    h.mark("a")
    h.mark("c")
    assert h.first("abc") is None
    h.reset()
    assert h.first("abc") == "A"
    with pytest.raises(KeyError):
        h.mark("nope")


def test_hint_line_settles_then_fades_in_and_cross_fades():
    line = HintLine(fade_s=0.25, settle_s=0.3)
    line.update(DT, "one")
    assert line.shown is None  # not yet: a brief state shouldn't flash a hint
    run_for(line.update, 0.35, "one")
    assert line.shown == "one"
    run_for(line.update, 0.3, "one")
    assert line.alpha == 1.0
    run_for(line.update, 0.1, "two")  # a short-lived target: fades a little, comes back
    assert line.shown == "one" and 0 < line.alpha < 1
    run_for(line.update, 0.3, "one")
    assert line.alpha == 1.0
    run_for(line.update, 0.7, "two")
    assert line.shown == "two" and line.alpha > 0.5
    run_for(line.update, 0.5, None)
    assert line.shown is None and line.alpha == 0.0


def test_vision_hints():
    s = VisionScene()
    assert s.hint().startswith("Raise your hand")
    class Visible:
        visible = True

    s.handle([], [Visible()])
    assert s.hint().startswith("Make a fist")

    class E:
        type = "fist_start"

    s.handle([E()], [])
    assert s.hint() is None
    s.idle_reset()
    assert s.hint().startswith("Raise your hand")


@pytest.fixture
def screen():
    pygame.init()
    text.reset()
    yield pygame.display.set_mode((1280, 720))
    pygame.quit()


def test_hint_and_quit_drawing(screen):
    line = HintLine()
    run_for(line.update, 1.0, "Pinch your thumb and index finger to make a shape")
    before = pygame.surfarray.array3d(screen).sum()
    line.draw(screen)
    assert pygame.surfarray.array3d(screen).sum() != before
    k = HoldKey(2.0)
    draw_quit_hold(screen, k)  # nothing held: nothing drawn, no crash
    k.press()
    k.update(1.0)
    draw_quit_hold(screen, k)
    k.release()
    draw_quit_hold(screen, k)
    HintLine(enabled=False).draw(screen)


# --- app ----------------------------------------------------------------------------------------
@pytest.fixture
def empty_clip(tmp_path):
    p = tmp_path / "empty.jsonl.gz"
    rec = Recorder(p, note="no hands")
    for i in range(30):
        rec.write(frame(i + 1, i / 30, []))
    rec.close()
    return p


@pytest.fixture
def hand_clip(tmp_path):
    p = tmp_path / "hand.jsonl.gz"
    rec = Recorder(p, note="one hand")
    for i in range(30):
        rec.write(frame(i + 1, i / 30, [make_hand(center=(0.5, 0.5))]))
    rec.close()
    return p


def test_esc_tap_does_not_quit_but_holding_does(empty_clip):
    cfg = config.load()
    pygame.init()
    try:
        app = App(cfg, ReplaySource(empty_clip), RunOptions(scene="geometry", max_frames=1))
        app._on_key(pygame.K_ESCAPE)
        app._booth_update(0.5)
        app.quit_hold.release()
        app._booth_update(DT)
        assert app.running
        app._on_key(pygame.K_ESCAPE)
        app._booth_update(cfg.booth.quit_hold_s - 0.1)
        assert app.running
        app._booth_update(0.2)
        assert not app.running
        hk = next(h for h in app.hotkeys if h.label == "Esc")
        assert "hold 2 s" in hk.description
    finally:
        pygame.quit()


def test_idle_reset_runs_in_the_app(empty_clip):
    cfg = config.load(overrides={"booth": {"idle_timeout_s": 0.2}})
    pygame.init()
    try:
        app = App(cfg, ReplaySource(empty_clip, loop=True), RunOptions(scene="geometry"))
        app.scene.world.spawn("circle", (640, 300), 40)
        app.scene.hints.mark("create")
        app.opts.max_frames = 60  # 0.2 s idle + 0.5 s fade-out, at 60 FPS
        assert app.run() == 0
        assert app.scene.world.pieces == [] and app.scene.hints.done == set()
        assert app.hint_line.shown is not None  # the first hint is back on screen
    finally:
        pygame.quit()


def test_no_idle_reset_while_a_hand_is_in_view(hand_clip):
    cfg = config.load(overrides={"booth": {"idle_timeout_s": 0.2}})
    pygame.init()
    try:
        app = App(cfg, ReplaySource(hand_clip, loop=True), RunOptions(scene="geometry"))
        piece = app.scene.world.spawn("circle", (640, 300), 40)
        app.opts.max_frames = 30
        assert app.run() == 0
        assert app.hands_tracked == 1 and piece.fade is None
    finally:
        pygame.quit()
