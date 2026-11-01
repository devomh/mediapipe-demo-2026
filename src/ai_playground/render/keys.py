"""Semi-transparent overlay listing the operator keys and their current state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame

from ai_playground.render import text
from ai_playground.render.ui import px

ON, OFF = (110, 230, 140), (150, 150, 160)


@dataclass(frozen=True, slots=True)
class Hotkey:
    keys: tuple[int, ...]  # pygame key codes
    label: str  # as shown, e.g. "V" or "1 / 2"
    description: str
    action: Callable[[int], None]  # gets the key code that was pressed
    state: Callable[[], str] | None = None  # e.g. "on", "off", "dim"
    available: Callable[[], bool] | None = None  # hidden and ignored when this returns False

    def is_available(self) -> bool:
        return self.available is None or self.available()


class KeysOverlay:
    def __init__(self, visible: bool = True) -> None:
        self.visible = visible

    def draw(self, surface: pygame.Surface, hotkeys: list[Hotkey]) -> None:
        if not self.visible:
            return
        rows = [h for h in hotkeys if h.is_available()]
        size, row_h, pad, title_h = 14, px(20), px(10), px(22)
        f = text.font(size)
        label_w = max(f.size(h.label)[0] for h in rows) + px(14)
        desc_w = max(f.size(h.description)[0] for h in rows) + px(14)
        state_w = max((f.size(h.state())[0] for h in rows if h.state), default=0)
        w = pad * 2 + label_w + desc_w + state_w
        h = pad * 2 + title_h + row_h * len(rows)
        # bottom-left, clear of the HUD and the table
        x, y = px(8), surface.get_height() - h - px(8)
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        surface.blit(panel, (x, y))
        text.draw(surface, "Keys", (x + pad, y + pad), (90, 220, 255), size, bold=True)
        for i, hk in enumerate(rows):
            ry = y + pad + title_h + row_h * i
            text.draw(surface, hk.label, (x + pad, ry), (255, 255, 255), size, bold=True)
            text.draw(surface, hk.description, (x + pad + label_w, ry), (210, 210, 210), size)
            if hk.state:
                s = hk.state()
                color = OFF if s in ("off", "hidden", "-") else ON
                text.draw(surface, s, (x + pad + label_w + desc_w, ry), color, size)
