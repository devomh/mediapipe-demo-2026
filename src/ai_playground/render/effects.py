"""Shape palette and drawing. Glow sprites and pooled particles arrive with A.4 polish."""

from __future__ import annotations

import math
import random

import pygame

from ai_playground.physics.world import KINDS, Piece, shape_vertices
from ai_playground.render import text

# 8 saturated hues that read well on a dark background.
PALETTE = (
    (255, 87, 87),  # coral red
    (255, 170, 50),  # amber
    (250, 230, 70),  # lemon
    (90, 230, 120),  # mint
    (60, 210, 230),  # cyan
    (80, 140, 255),  # azure
    (170, 110, 255),  # violet
    (255, 100, 200),  # pink
)


def next_hue(previous: int | None, rng: random.Random) -> int:
    """A random palette index different from the previous one."""
    choices = [i for i in range(len(PALETTE)) if i != previous]
    return rng.choice(choices)


def lighten(c: tuple[int, int, int], f: float = 0.45) -> tuple[int, int, int]:
    return tuple(round(v + (255 - v) * f) for v in c)  # type: ignore[return-value]


def fade_to(c: tuple[int, int, int], bg: tuple[int, int, int], alpha: float):
    return tuple(round(b + (v - b) * alpha) for v, b in zip(c, bg, strict=True))


def piece_points(piece: Piece) -> list[tuple[float, float]]:
    body = piece.body
    return [tuple(body.local_to_world(v)) for v in piece.shape.get_vertices()]


def draw_piece(surface: pygame.Surface, piece: Piece, bg=(10, 12, 18)) -> None:
    color = PALETTE[piece.hue]
    a = piece.alpha
    fill, edge = fade_to(color, bg, a), fade_to(lighten(color), bg, a)
    pos = piece.body.position
    if piece.kind == "circle":
        c = (round(pos.x), round(pos.y))
        r = round(piece.radius)
        pygame.draw.circle(surface, fill, c, r)
        pygame.draw.circle(surface, edge, c, r, 3)
        # A spoke, so rolling is visible.
        tip = piece.body.local_to_world((piece.radius * 0.7, 0))
        pygame.draw.line(surface, edge, c, (round(tip.x), round(tip.y)), 3)
    else:
        pts = piece_points(piece)
        pygame.draw.polygon(surface, fill, pts)
        pygame.draw.polygon(surface, edge, pts, 3)


def _rotated(kind: str, radius: float, angle: float, cx: float, cy: float):
    ca, sa = math.cos(angle), math.sin(angle)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in shape_vertices(kind, radius)]


def draw_preview(
    surface: pygame.Surface,
    kind: str,
    pos: tuple[float, float],
    radius: float,
    hue: int,
    t: float,
    angle: float = 0.0,
    ring: bool = True,
) -> None:
    """Pulsing translucent shape plus a size ring (off while stretching), while a pinch is held."""
    color = PALETTE[hue]
    pulse = 0.5 + 0.5 * abs(((t * 2.0) % 2.0) - 1.0)  # triangle wave, 1 Hz
    size = round(radius * 2 + 8)
    layer = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    alpha = round(90 + 80 * pulse)
    if kind == "circle":
        pygame.draw.circle(layer, (*color, alpha), (c, c), radius)
    else:
        pygame.draw.polygon(layer, (*color, alpha), _rotated(kind, radius, angle, c, c))
    surface.blit(layer, (round(pos[0] - c), round(pos[1] - c)))
    if ring:
        center = (round(pos[0]), round(pos[1]))
        pygame.draw.circle(surface, lighten(color, 0.6), center, round(radius + 6), 2)


def draw_stretch_line(
    surface: pygame.Surface, a: tuple[float, float], b: tuple[float, float], hue: int
) -> None:
    """The glowing link between two pinch points while a shape is stretched."""
    color = PALETTE[hue]
    pygame.draw.line(surface, fade_to(color, (10, 12, 18), 0.4), a, b, 9)
    pygame.draw.line(surface, lighten(color, 0.7), a, b, 3)
    for p in (a, b):
        pygame.draw.circle(surface, lighten(color, 0.7), (round(p[0]), round(p[1])), 7)


def draw_palette(surface: pygame.Surface, selected: str, chosen: bool) -> None:
    """Shape palette on the right edge: finger count -> shape, with the current selection lit.

    `chosen` means the other hand picked it; otherwise it's the automatic cycle's "next".
    """
    cell, icon = 58, 17
    w, h = 82, cell * len(KINDS) + 34
    x = surface.get_width() - w - 8
    y = (surface.get_height() - h) // 2
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 120))
    surface.blit(panel, (x, y))
    label = "chosen" if chosen else "next"
    text.draw(surface, label, (x + w // 2, y + 8), (90, 220, 255), 13, anchor="midtop")
    for i, kind in enumerate(KINDS):
        cx, cy = x + w // 2 + 8, y + 34 + cell * i + cell // 2
        lit = kind == selected
        color = (255, 255, 255) if lit else (120, 120, 130)
        if lit:
            pygame.draw.rect(
                surface, (90, 220, 255), (x + 4, cy - cell // 2 + 3, w - 8, cell - 6), 2, 6
            )
        text.draw(surface, str(i + 1), (x + 12, cy), color, 16, bold=True, anchor="midleft")
        if kind == "circle":
            pygame.draw.circle(surface, color, (cx, cy), icon)
        else:
            r = icon * (1.0 if kind == "bar" else 1.1)
            pygame.draw.polygon(surface, color, _rotated(kind, r, 0.0, cx, cy))


class Sparkles:
    """Short expanding rings, e.g. on release. Replaced by the particle system in A.4."""

    def __init__(self, life: float = 0.35) -> None:
        self.life = life
        self._items: list[list] = []  # [x, y, hue, age]

    def add(self, pos: tuple[float, float], hue: int) -> None:
        self._items.append([pos[0], pos[1], hue, 0.0])

    def update(self, dt: float) -> None:
        for it in self._items:
            it[3] += dt
        self._items = [it for it in self._items if it[3] < self.life]

    def draw(self, surface: pygame.Surface) -> None:
        for x, y, hue, age in self._items:
            k = age / self.life
            color = fade_to(lighten(PALETTE[hue], 0.6), (10, 12, 18), 1.0 - k)
            pygame.draw.circle(surface, color, (round(x), round(y)), round(8 + 40 * k), 2)
