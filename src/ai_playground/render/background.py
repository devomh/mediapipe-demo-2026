"""Camera background (full / dimmed / hidden) or a dark grid when there is no image."""

from __future__ import annotations

import numpy as np
import pygame

from ai_playground.render.viewport import Viewport

MODES = ("full", "dim", "hidden")
BG_COLOR = (14, 16, 22)
GRID_COLOR = (30, 34, 46)


class Background:
    def __init__(self, mode: str = "full", dim_brightness: float = 0.33) -> None:
        self.mode = mode
        self.dim_brightness = dim_brightness  # camera brightness in "dim" mode, 0..1
        self._scaled: pygame.Surface | None = None
        self._dim: pygame.Surface | None = None
        self._dim_key: tuple | None = None

    def cycle(self) -> None:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]

    def draw(self, surface: pygame.Surface, vp: Viewport, image: np.ndarray | None) -> None:
        if image is None or self.mode == "hidden":
            draw_grid(surface, vp)
            return
        h, w = image.shape[:2]
        src = pygame.image.frombuffer(image.data, (w, h), "RGB")
        if self._scaled is None or self._scaled.get_size() != vp.dest_size:
            self._scaled = pygame.Surface(vp.dest_size, 0, src)  # scale() needs the same format
        pygame.transform.scale(src, vp.dest_size, self._scaled)
        surface.blit(self._scaled, (round(vp.offset[0]), round(vp.offset[1])))
        if self.mode == "dim":
            key = (surface.get_size(), self.dim_brightness)
            if self._dim is None or self._dim_key != key:
                self._dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                self._dim.fill((0, 0, 0, round(255 * (1.0 - self.dim_brightness))))
                self._dim_key = key
            surface.blit(self._dim, (0, 0))


def draw_grid(surface: pygame.Surface, vp: Viewport, step: float = 0.1) -> None:
    surface.fill(BG_COLOR)
    ww, wh = surface.get_size()
    for i in range(11):
        x, _ = vp.to_px(i * step, 0)
        _, y = vp.to_px(0, i * step)
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, wh))
        pygame.draw.line(surface, GRID_COLOR, (0, y), (ww, y))
