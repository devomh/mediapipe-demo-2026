"""Air Writing's reading screen (spec §6): the result panel on the right, the text line at the
top, and the ink shrinking into the 28x28 grid the network sees."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pygame

from ai_playground.ml.charnet import Guess
from ai_playground.render import text
from ai_playground.render.ink import draw_pixels
from ai_playground.render.ui import px

ACCENT = (90, 220, 255)
SHRINK_S = 0.4  # the ink flies into the grid
REVEAL_S = 0.3  # then the answer and the bars come in
LOOKALIKE_NOTE = "These look alike to the AI, and to people!"


@dataclass(frozen=True, slots=True)
class Layout:
    panel: pygame.Rect
    title_y: int
    answer: pygame.Rect
    note_y: int  # subtitle under a look-alike pair; 0 without one
    bars_y: int
    grid_label_y: int
    grid: pygame.Rect
    caption_y: int


def layout(surface: pygame.Surface, pair: bool) -> Layout:
    """Where everything goes; laid out at 720 px tall and scaled (render/ui)."""
    w, pad = px(256), px(14)
    grid_side = px(224)  # 8 px per network pixel at 720
    heights = [
        px(22),
        px(104),
        px(34) if pair else 0,
        3 * px(26) + px(10),
        px(22),
        grid_side,
        px(22),
    ]
    h = 2 * pad + sum(heights)
    x = surface.get_width() - w - px(16)
    y = max(px(80), (surface.get_height() - h) // 2)
    cy = y + pad
    title_y = cy
    cy += heights[0]
    answer = pygame.Rect(x, cy, w, heights[1])
    cy += heights[1]
    note_y = cy if pair else 0
    cy += heights[2]
    bars_y = cy
    cy += heights[3]
    grid_label_y = cy
    cy += heights[4]
    grid = pygame.Rect(x + (w - grid_side) // 2, cy, grid_side, grid_side)
    cy += grid_side
    return Layout(pygame.Rect(x, y, w, h), title_y, answer, note_y, bars_y, grid_label_y, grid, cy)


def ease(k: float) -> float:
    k = min(1.0, max(0.0, k))
    return k * k * (3 - 2 * k)


def draw_result(
    surface: pygame.Surface,
    guesses: Sequence[Guess],
    pair: tuple[str, str] | None,
    image: np.ndarray,
    age: float,
) -> Layout:
    """The panel: what the AI read, how sure it is, and the 784 numbers it read it from.
    `age` is seconds since the read: the grid fills in while the ink shrinks into it, then the
    answer and the bars appear."""
    lay = layout(surface, pair is not None)
    x, w = lay.panel.x, lay.panel.width
    cx = x + w // 2
    panel = pygame.Surface(lay.panel.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, 160), panel.get_rect(), 0, px(14))
    surface.blit(panel, lay.panel.topleft)
    text.draw(surface, "The AI reads", (cx, lay.title_y), ACCENT, 14, anchor="midtop")

    reveal = ease((age - SHRINK_S) / REVEAL_S)
    if reveal > 0:
        if pair is None:
            img = text.font(84, mono=False, bold=True).render(guesses[0].label, True, (255,) * 3)
        else:
            img = text.font(54, mono=False, bold=True).render(
                f"{pair[0]} or {pair[1]}?", True, (255,) * 3
            )
        img.set_alpha(round(255 * reveal))
        surface.blit(img, img.get_rect(center=lay.answer.center))
        if pair is not None:
            for i, line in enumerate(text.wrap(LOOKALIKE_NOTE, 13, w - px(28))):
                text.draw(
                    surface,
                    line,
                    (cx, lay.note_y + i * px(16)),
                    (200, 200, 210),
                    13,
                    mono=False,
                    anchor="midtop",
                )
        _bars(surface, guesses, x + px(14), lay.bars_y, w - px(28), reveal)

    text.draw(surface, "What the AI sees", (cx, lay.grid_label_y), ACCENT, 14, anchor="midtop")
    fill_in = ease(age / SHRINK_S)
    if fill_in > 0:
        draw_pixels(surface, image * fill_in, lay.grid)
        _grid_lines(surface, lay.grid)
    else:
        pygame.draw.rect(surface, (70, 75, 90), lay.grid, 1)
    text.draw(
        surface,
        f"28 x 28 = {28 * 28} numbers",
        (cx, lay.caption_y + px(4)),
        (160, 160, 175),
        13,
        anchor="midtop",
    )
    return lay


def _bars(surface, guesses: Sequence[Guess], x: int, y: int, w: int, reveal: float) -> None:
    label_w, pct_w, row = px(26), px(50), px(26)
    bar_w = w - label_w - pct_w
    for i, g in enumerate(guesses[:3]):
        ry = y + i * row
        color = ACCENT if i == 0 else (150, 150, 165)
        text.draw(
            surface, g.label, (x, ry + row // 2), (255, 255, 255), 18, bold=True, anchor="midleft"
        )
        bg = pygame.Rect(x + label_w, ry + row // 2 - px(7), bar_w, px(14))
        pygame.draw.rect(surface, (45, 48, 60), bg, 0, px(4))
        filled = bg.copy()
        filled.width = round(bar_w * g.p * reveal)
        if filled.width > 0:
            pygame.draw.rect(surface, color, filled, 0, px(4))
        pct = "<1 %" if g.p < 0.005 else f"{g.p * 100:.0f} %"
        text.draw(surface, pct, (x + w, ry + row // 2), (220, 220, 230), 15, anchor="midright")


def _grid_lines(surface: pygame.Surface, rect: pygame.Rect) -> None:
    """Faint lines between the network's pixels, so the grid reads as 784 separate numbers."""
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    for i in range(1, 28):
        k = round(i * rect.width / 28)
        pygame.draw.line(layer, (255, 255, 255, 22), (k, 0), (k, rect.height))
        pygame.draw.line(layer, (255, 255, 255, 22), (0, k), (rect.width, k))
    surface.blit(layer, rect.topleft)
    pygame.draw.rect(surface, (70, 75, 90), rect, 1)


def shrink(
    points: np.ndarray,
    bounds: tuple[float, float, float, float],
    grid: pygame.Rect,
    fill: float,
    k: float,
) -> tuple[np.ndarray, float]:
    """Ink points `k` (0..1) of the way into the grid, framed as the network frames them.
    Returns the points and the scale reached (for the line width)."""
    x0, y0, x1, y1 = bounds
    side = max(x1 - x0, y1 - y0, 1.0)
    s = grid.width * fill / side
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    target = np.array(grid.center, np.float32) + (points - c) * s
    k = ease(k)
    return points + (target - points) * k, 1.0 + (s - 1.0) * k


def draw_text_line(surface: pygame.Surface, line: str, newest_age: float) -> None:
    """The characters read so far, top centre; the newest one fades in."""
    if not line:
        return
    f = text.font(40, bold=True)
    cw, ch = f.size("M")
    gap = px(6)
    w = len(line) * (cw + gap) - gap + 2 * px(18)
    h = ch + 2 * px(8)
    x, y = (surface.get_width() - w) // 2, px(14)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, 150), panel.get_rect(), 0, h // 2)
    surface.blit(panel, (x, y))
    for i, c in enumerate(line):
        img = f.render(c, True, (255, 255, 255))
        if i == len(line) - 1:
            img.set_alpha(round(255 * ease(newest_age / 0.4)))
        cx = x + px(18) + i * (cw + gap) + cw // 2
        surface.blit(img, img.get_rect(center=(cx, y + h // 2)))


def lookalike(guesses: Sequence[Guess], pairs: str, p_min: float) -> tuple[str, str] | None:
    """(first, second) if the top two guesses are a known look-alike pair and the second is
    likely enough to be worth showing (spec §5.6)."""
    if len(guesses) < 2 or guesses[1].p < p_min:
        return None
    a, b = guesses[0].label, guesses[1].label
    known = {frozenset(p) for p in pairs.split() if len(p) == 2}
    return (a, b) if frozenset((a, b)) in known else None
