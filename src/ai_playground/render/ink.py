"""Air Writing ink: glowing round-capped strokes, the writing area, and a small thumbnail."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pygame

from ai_playground.ml.strokes import Stroke, bounds
from ai_playground.render import text
from ai_playground.render.particles import LEVELS, glow_sprite
from ai_playground.render.ui import px


class InkGlow:
    """Soft additive dots stamped along the ink. Cached per (color, radius, brightness)."""

    def __init__(self, intensity: float = 0.16) -> None:
        self.intensity = intensity
        self._cache: dict[tuple[tuple[int, int, int], int, int], pygame.Surface] = {}

    def sprite(self, color: tuple[int, int, int], radius: int, alpha: float) -> pygame.Surface:
        level = max(0, min(LEVELS - 1, round(alpha * (LEVELS - 1))))
        key = (color, radius, level)
        s = self._cache.get(key)
        if s is None:
            k = self.intensity * (level + 1) / LEVELS
            s = self._cache[key] = glow_sprite(color, radius, k)
        return s


def draw_ink(
    surface: pygame.Surface,
    points: np.ndarray,
    color: tuple[int, int, int],
    width: float,
    alpha: float = 1.0,
    glow: InkGlow | None = None,
) -> None:
    """One stroke: a glow stamped every half glow radius, then the solid line on top.

    Points are resampled closer together than the line is wide, so a dot per point (plus a
    segment between them) gives round caps and joints at any angle.
    """
    if len(points) == 0 or alpha <= 0:
        return
    r = max(1.0, width / 2)
    if glow is not None:
        gr = max(2, round(2.2 * width))
        sprite = glow.sprite(color, gr, alpha)
        spacing = np.linalg.norm(np.diff(points, axis=0), axis=1).mean() if len(points) > 1 else 1
        every = max(1, round(0.5 * gr / max(spacing, 1e-3)))
        for x, y in points[::every]:
            surface.blit(sprite, (round(x) - gr, round(y) - gr), special_flags=pygame.BLEND_ADD)
    if alpha >= 1.0:
        _line(surface, points, color, r)
        return
    pad = round(r) + 2
    x0, y0 = points.min(axis=0) - pad
    x1, y1 = points.max(axis=0) + pad
    layer = pygame.Surface((int(x1 - x0) + 1, int(y1 - y0) + 1), pygame.SRCALPHA)
    _line(layer, points - (x0, y0), color, r)
    layer.set_alpha(round(255 * alpha))
    surface.blit(layer, (round(x0), round(y0)))


def _line(surface: pygame.Surface, points: np.ndarray, color, r: float) -> None:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) > 1:
        pygame.draw.lines(surface, color, False, pts, max(1, round(2 * r)))
    for p in pts:
        pygame.draw.circle(surface, color, p, r)


def draw_area(surface: pygame.Surface, rect: pygame.Rect) -> None:
    """The faint writing box: only a guide to a sensible character size."""
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    radius = px(28)
    pygame.draw.rect(layer, (255, 255, 255, 10), layer.get_rect(), 0, radius)
    pygame.draw.rect(layer, (255, 255, 255, 40), layer.get_rect(), px(2), radius)
    surface.blit(layer, rect.topleft)


def draw_thumbnail(
    surface: pygame.Surface,
    strokes: Sequence[Stroke],
    title: str,
    color: tuple[int, int, int] = (235, 235, 235),
    image: np.ndarray | None = None,
) -> None:
    """A small panel on the right with the last character read and, below it, the 28x28 image
    the network gets (W.4 replaces this with the model's answer)."""
    b = bounds(strokes)
    if b is None:
        return
    box, pad, title_h = px(150), px(12), px(26)
    grid_h = 0 if image is None else box + title_h
    w, h = box + 2 * pad, box + 2 * pad + title_h + grid_h
    x = surface.get_width() - w - px(16)
    y = (surface.get_height() - h) // 2
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, 140), panel.get_rect(), 0, px(12))
    surface.blit(panel, (x, y))
    text.draw(surface, title, (x + w // 2, y + pad), (90, 220, 255), 14, anchor="midtop")
    x0, y0, x1, y1 = b
    scale = 0.85 * box / max(x1 - x0, y1 - y0, 1.0)
    cx, cy = x + w / 2, y + title_h + pad + box / 2
    off = np.array([cx - scale * (x0 + x1) / 2, cy - scale * (y0 + y1) / 2], np.float32)
    for s in strokes:
        _line(surface, s.points * scale + off, color, max(1.5, px(3)))
    if image is not None:
        gy = y + title_h + pad + box
        text.draw(
            surface, "What the AI sees", (x + w // 2, gy), (90, 220, 255), 14, anchor="midtop"
        )
        draw_pixels(surface, image, pygame.Rect(x + pad, gy + title_h, box, box))


def draw_pixels(surface: pygame.Surface, image: np.ndarray, rect: pygame.Rect) -> None:
    """A small grayscale image drawn as big square pixels."""
    g = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.repeat(g.T[:, :, None], 3, axis=2)
    small = pygame.surfarray.make_surface(rgb)
    surface.blit(pygame.transform.scale(small, rect.size), rect.topleft)
    pygame.draw.rect(surface, (70, 75, 90), rect, 1)


def draw_collect_prompt(
    surface: pygame.Surface, target: tuple[str, str, str] | None, progress: str
) -> None:
    """Collect mode: the character to write and how, on the left."""
    w, pad = px(190), px(14)
    lines = ["Thank you!", "All written. Press T to stop."] if target is None else []
    big = target[0] if target is not None else ""
    instr = text.wrap(target[2], 15, w - 2 * pad) if target is not None else lines
    h = pad * 3 + px(20) + (px(90) if big else 0) + px(20) * len(instr) + px(20)
    x, y = px(16), (surface.get_height() - h) // 2
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, 160), panel.get_rect(), 0, px(12))
    surface.blit(panel, (x, y))
    cy = y + pad
    text.draw(surface, "Collect: write", (x + w // 2, cy), (255, 200, 90), 14, anchor="midtop")
    cy += px(20)
    if big:
        text.draw(
            surface,
            big,
            (x + w // 2, cy),
            (255, 255, 255),
            72,
            mono=False,
            bold=True,
            anchor="midtop",
        )
        cy += px(90)
    for line in instr:
        text.draw(surface, line, (x + w // 2, cy), (225, 225, 225), 15, mono=False, anchor="midtop")
        cy += px(20)
    cy += pad
    text.draw(surface, progress, (x + w // 2, cy), (150, 150, 165), 13, anchor="midtop")
