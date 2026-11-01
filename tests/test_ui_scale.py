"""Overlays scale with the window: text, panels and cursors are laid out for 720 px tall."""

import pygame
import pytest
from synthetic import frame, make_hand

from ai_playground.app import App, RunOptions
from ai_playground.render import text, ui
from ai_playground.render.keys import KeysOverlay
from ai_playground.sources.recorder import Recorder
from ai_playground.sources.replay import ReplaySource


def test_scale_follows_window_height_and_factor():
    ui.set_window_height(720)
    assert ui.scale() == 1.0 and ui.px(14) == 14
    ui.set_window_height(2880)
    assert ui.scale() == 4.0 and ui.px(14) == 56
    ui.set_factor(1.5)
    ui.set_window_height(720)
    assert ui.scale() == 1.5


def test_scale_is_clamped():
    ui.set_window_height(100)
    assert ui.scale() == ui.MIN_SCALE and ui.px(1) >= 1  # width 0 would fill the shape
    ui.set_window_height(100_000)
    assert ui.scale() == ui.MAX_SCALE


@pytest.fixture
def display():
    pygame.init()
    yield
    pygame.quit()


def test_fonts_scale(display):
    small = text.font(14).get_height()
    ui.set_window_height(1440)
    assert text.font(14).get_height() == pytest.approx(2 * small, rel=0.15)


def _keys_panel_height(app: App) -> int:
    """Height of the keys panel: its rows of dark background up from the bottom-left corner."""
    surface = pygame.Surface(app.surface.get_size())
    surface.fill((255, 255, 255))
    KeysOverlay().draw(surface, app.hotkeys)
    x, h = ui.px(10), surface.get_height()
    return sum(1 for y in range(h) if surface.get_at((x, y))[0] < 200)


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "scale.jsonl.gz"
    rec = Recorder(p, note="scale")
    for i in range(10):
        rec.write(frame(i + 1, i / 30, [make_hand(center=(0.5, 0.5))]))
    rec.close()
    return p


def test_window_resize_rescales_the_ui(cfg, clip, display):
    app = App(cfg, ReplaySource(clip, loop=True), RunOptions(max_frames=2))
    assert app.surface.get_height() == 720 and ui.scale() == 1.0
    small = _keys_panel_height(app)

    pygame.display.set_mode((2560, 1440), pygame.RESIZABLE)  # what maximizing does
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=2560, h=1440))
    assert app.run() == 0
    assert app.viewport.window_size == (2560, 1440)
    assert ui.scale() == 2.0
    assert _keys_panel_height(app) == pytest.approx(2 * small, abs=4)


def test_ui_scale_config_multiplies(cfg, clip, display):
    from dataclasses import replace

    big = replace(cfg, display=replace(cfg.display, ui_scale=1.25))
    App(big, ReplaySource(clip, loop=True), RunOptions(max_frames=1))
    assert ui.scale() == 1.25
