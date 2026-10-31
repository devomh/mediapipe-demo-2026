"""Shape palette, shape drawing, and hand cursors. Glow and particles: render/particles.py."""

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


def draw_piece(
    surface: pygame.Surface, piece: Piece, bg=(10, 12, 18), highlight: bool = False
) -> None:
    color = PALETTE[piece.hue]
    a = piece.alpha
    fill, edge = fade_to(color, bg, a), fade_to(lighten(color, 0.8 if highlight else 0.45), bg, a)
    width = 5 if highlight else 3
    pos = piece.body.position
    if piece.kind == "circle":
        c = (round(pos.x), round(pos.y))
        r = round(piece.radius)
        pygame.draw.circle(surface, fill, c, r)
        pygame.draw.circle(surface, edge, c, r, width)
        # A spoke, so rolling is visible.
        tip = piece.body.local_to_world((piece.radius * 0.7, 0))
        pygame.draw.line(surface, edge, c, (round(tip.x), round(tip.y)), 3)
    else:
        pts = piece_points(piece)
        pygame.draw.polygon(surface, fill, pts)
        pygame.draw.polygon(surface, edge, pts, width)


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


# Each tracked hand gets its own cursor colour (by track id), distinct from the shape palette.
HAND_COLORS = ((140, 225, 255), (255, 170, 225), (190, 255, 160), (255, 215, 140))


def hand_color(hand_id: int) -> tuple[int, int, int]:
    return HAND_COLORS[hand_id % len(HAND_COLORS)]


def draw_paddle(
    surface: pygame.Surface,
    pos: tuple[float, float],
    radius: float,
    strength: float = 1.0,
    color: tuple[int, int, int] = HAND_COLORS[0],
) -> None:
    """Soft ring where the open hand pushes shapes; fades in (`strength` < 1) while arming."""
    size = round(radius * 2 + 6)
    layer = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    k = max(0.0, min(1.0, strength))
    pygame.draw.circle(layer, (*color, round(28 * k)), (c, c), radius)
    pygame.draw.circle(layer, (*color, round(120 * k)), (c, c), radius, 2)
    surface.blit(layer, (round(pos[0] - c), round(pos[1] - c)))


def draw_grab_ring(
    surface: pygame.Surface,
    pos: tuple[float, float],
    t: float,
    color: tuple[int, int, int] = HAND_COLORS[0],
) -> None:
    """A hand holding a shape: a small ring that turns."""
    center = (round(pos[0]), round(pos[1]))
    r = 16
    start = (t * 4.0) % (2 * math.pi)
    rect = pygame.Rect(center[0] - r, center[1] - r, 2 * r, 2 * r)
    pygame.draw.circle(surface, (40, 44, 60), center, r + 3)
    pygame.draw.arc(surface, color, rect, start, start + 4.2, 4)


def draw_pinch_dot(
    surface: pygame.Surface, pos: tuple[float, float], color: tuple[int, int, int]
) -> None:
    """Where a pinch is (making or about to make a shape)."""
    center = (round(pos[0]), round(pos[1]))
    pygame.draw.circle(surface, (20, 22, 30), center, 9)
    pygame.draw.circle(surface, color, center, 6)


def draw_hand_marker(
    surface: pygame.Surface, pos: tuple[float, float], color: tuple[int, int, int]
) -> None:
    """A hand the computer sees but that isn't doing anything yet: a faint small ring."""
    center = (round(pos[0]), round(pos[1]))
    pygame.draw.circle(surface, fade_to(color, (10, 12, 18), 0.55), center, 12, 2)


def draw_progress_ring(surface: pygame.Surface, pos: tuple[float, float], frac: float) -> None:
    """Fills clockwise from the top while both fists are held (two-fist clear)."""
    center = (round(pos[0]), round(pos[1]))
    r = 44
    rect = pygame.Rect(center[0] - r, center[1] - r, 2 * r, 2 * r)
    pygame.draw.circle(surface, (60, 64, 80), center, r, 4)
    if frac > 0:
        end = math.pi / 2
        pygame.draw.arc(surface, (255, 120, 90), rect, end - 2 * math.pi * frac, end, 7)
