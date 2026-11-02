"""Booth shell drawing: the hint line at the bottom and the hold-Esc-to-quit indicator."""

from __future__ import annotations

import pygame

from ai_playground.booth import HoldKey
from ai_playground.render import text
from ai_playground.render.ui import px

HINT_SIZE = 22


class HintLine:
    """One line of onboarding text, bottom centre. A new hint waits until it has been wanted for
    `settle_s` (so a brief state, like a quick pinch, doesn't flash a hint) and cross-fades."""

    def __init__(self, enabled: bool = True, fade_s: float = 0.25, settle_s: float = 0.3) -> None:
        self.enabled = enabled
        self.fade_s, self.settle_s = fade_s, settle_s
        self.shown: str | None = None
        self.alpha = 0.0
        self._target: str | None = None
        self._since = 0.0

    def update(self, dt: float, target: str | None) -> None:
        if target != self._target:
            self._target, self._since = target, 0.0
        else:
            self._since += dt
        step = dt / self.fade_s if self.fade_s > 0 else 1.0
        if self.shown == target:
            self.alpha = min(1.0, self.alpha + step) if target is not None else 0.0
            return
        self.alpha = max(0.0, self.alpha - step)
        if self.alpha == 0.0 and self._since >= self.settle_s:
            self.shown = target

    def draw(self, surface: pygame.Surface) -> None:
        if not self.enabled or self.shown is None or self.alpha <= 0.0:
            return
        f = text.font(HINT_SIZE, mono=False)
        img = f.render(self.shown, True, (255, 255, 255))
        pad_x, pad_y = px(18), px(8)
        w, h = img.get_width() + 2 * pad_x, img.get_height() + 2 * pad_y
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (0, 0, 0, round(150 * self.alpha)), panel.get_rect(), 0, h // 2)
        img.set_alpha(round(255 * self.alpha))
        panel.blit(img, (pad_x, pad_y))
        sw, sh = surface.get_size()
        surface.blit(panel, panel.get_rect(midbottom=(sw // 2, sh - px(20))))


def draw_quit_hold(surface: pygame.Surface, hold: HoldKey) -> None:
    """While Esc is held: a filling bar at the top. After a short tap: a fading reminder."""
    sw = surface.get_width()
    if hold.held_s is not None:
        msg, alpha = "Keep holding Esc to quit", 1.0
    elif hold.reminder_s > 0:
        msg = f"Hold Esc for {hold.hold_s:g} s to quit"
        alpha = min(1.0, hold.reminder_s / (0.5 * HoldKey.REMINDER_S))
    else:
        return
    f = text.font(18, mono=False, bold=True)
    img = f.render(msg, True, (255, 255, 255))
    pad = px(12)
    bar_h = px(6) if hold.held_s is not None else 0
    w = max(img.get_width(), px(260)) + 2 * pad
    h = img.get_height() + 2 * pad + (bar_h + pad // 2 if bar_h else 0)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (40, 12, 16, round(210 * alpha)), panel.get_rect(), 0, px(10))
    img.set_alpha(round(255 * alpha))
    panel.blit(img, img.get_rect(midtop=(w // 2, pad)))
    if bar_h:
        bar = pygame.Rect(pad, h - pad - bar_h, w - 2 * pad, bar_h)
        pygame.draw.rect(panel, (90, 60, 64, 255), bar, 0, bar_h // 2)
        filled = bar.copy()
        filled.width = round(bar.width * hold.progress)
        if filled.width > 0:
            pygame.draw.rect(panel, (255, 120, 110, 255), filled, 0, bar_h // 2)
    surface.blit(panel, panel.get_rect(midtop=(sw // 2, px(16))))
