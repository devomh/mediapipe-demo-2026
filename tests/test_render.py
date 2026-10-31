"""Rendering paths that replay can't reach (camera image) and the error screen."""

import numpy as np
import pygame
import pytest

from ai_playground.app import show_error
from ai_playground.render.background import Background
from ai_playground.render.viewport import Viewport


@pytest.fixture
def screen():
    pygame.init()
    yield pygame.display.set_mode((1280, 720))
    pygame.quit()


@pytest.mark.parametrize("mode", ["full", "dim", "hidden"])
def test_camera_background_modes(screen, mode):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :320] = (255, 0, 0)  # left half red
    vp = Viewport(screen.get_size(), (640, 480))
    bg = Background(mode)
    for _ in range(2):  # second frame reuses the scaled surface
        bg.draw(screen, vp, img)
    r, g, b, _ = screen.get_at((100, 360))
    if mode == "full":
        assert (r, g, b) == (255, 0, 0)
    elif mode == "dim":
        assert 40 < r < 120 and g < 10
    else:
        assert r < 60  # grid


def test_viewport_cover_scaling_round_trip():
    vp = Viewport((1280, 720), (640, 480))
    assert vp.dest_size == (1280, 960)  # fills the width, crops top and bottom
    assert vp.to_px(0.5, 0.5) == (640, 360)
    x, y = vp.to_norm(*vp.to_px(0.2, 0.7))
    assert x == pytest.approx(0.2, abs=1e-3) and y == pytest.approx(0.7, abs=1e-3)


def test_error_screen_returns_on_escape(screen):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    msg = "Camera 0 could not be opened. " * 5
    assert show_error(screen, pygame.time.Clock(), "Camera not available", msg) == 1
