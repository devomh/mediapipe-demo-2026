"""Cached fonts and simple text drawing."""

from __future__ import annotations

from functools import cache

import pygame

MONO = ["menlo", "dejavusansmono", "ubuntumono", "couriernew", "monospace"]
SANS = ["helveticaneue", "helvetica", "dejavusans", "ubuntu", "arial"]


@cache
def font(size: int, mono: bool = True, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont(MONO if mono else SANS, size, bold=bold)


def reset() -> None:
    """Drop cached fonts; call after (re)initializing pygame, as old Font objects are invalid."""
    font.cache_clear()


def draw(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    color=(235, 235, 235),
    size: int = 14,
    mono: bool = True,
    bold: bool = False,
    anchor: str = "topleft",
    shadow: bool = False,
) -> pygame.Rect:
    f = font(size, mono, bold)
    img = f.render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    if shadow:
        surface.blit(f.render(text, True, (0, 0, 0)), rect.move(1, 1))
    surface.blit(img, rect)
    return rect


def wrap(text: str, size: int, width: int, mono: bool = False) -> list[str]:
    f = font(size, mono)
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = f"{line} {word}".strip()
            if f.size(trial)[0] <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines
