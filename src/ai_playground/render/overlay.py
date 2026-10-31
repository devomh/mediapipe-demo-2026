"""Vision overlay: what the computer sees. Skeleton, per-hand panel, numbers table, event flashes.

Drawn by the app on top of any scene when toggled (`V`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pygame

from ai_playground.config import PinchConfig
from ai_playground.gestures import features as F
from ai_playground.gestures.engine import HandState
from ai_playground.render import text
from ai_playground.render.viewport import Viewport
from ai_playground.types import GestureEvent

FINGER_COLORS = (
    (255, 150, 60),  # thumb
    (255, 220, 70),  # index
    (110, 230, 120),  # middle
    (80, 200, 255),  # ring
    (230, 110, 255),  # pinky
)
PALM_COLOR = (220, 220, 220)
FLASH_S = 0.5
PANEL_BG = (0, 0, 0, 160)
ACCENT = (90, 220, 255)


def _finger_of(i: int) -> int:
    return 0 if i == 0 else (i - 1) // 4


@dataclass(slots=True)
class _Flash:
    label: str
    pos: tuple[float, float]
    age: float = 0.0


class VisionOverlay:
    def __init__(self, pinch_cfg: PinchConfig) -> None:
        self.pinch_cfg = pinch_cfg
        self.visible = True
        self.show_indices = False
        self._flashes: list[_Flash] = []
        self._states: list[HandState] = []

    def handle(self, events: Sequence[GestureEvent], states: Sequence[HandState]) -> None:
        self._states = [s for s in states if s.visible]
        for e in events:
            if not e.type.endswith("_move"):
                self._flashes.append(_Flash(e.type, e.pos))

    def update(self, dt: float) -> None:
        for f in self._flashes:
            f.age += dt
        self._flashes = [f for f in self._flashes if f.age < FLASH_S]

    def draw(self, surface: pygame.Surface, vp: Viewport) -> None:
        if not self.visible:
            return
        for s in self._states:
            self._draw_skeleton(surface, vp, s)
        for s in self._states:
            self._draw_hand_panel(surface, vp, s)
        if self._states:
            self._draw_table(surface, self._states[0])
        for f in self._flashes:
            alpha = int(255 * (1.0 - f.age / FLASH_S))
            x, y = vp.to_px(*f.pos)
            img = text.font(16, bold=True).render(f.label, True, (255, 255, 255))
            img.set_alpha(alpha)
            surface.blit(img, img.get_rect(midbottom=(x, y - 14 - int(20 * f.age / FLASH_S))))

    def _draw_skeleton(self, surface: pygame.Surface, vp: Viewport, s: HandState) -> None:
        lm = s.tracked.hand.landmarks
        pts = [vp.to_px(float(lm[i, 0]), float(lm[i, 1])) for i in range(len(lm))]
        width = max(2, round(vp.scale * 2))
        for a, b in F.PALM_EDGES:
            pygame.draw.line(surface, PALM_COLOR, pts[a], pts[b], width)
        for finger, chain in enumerate(F.FINGER_CHAINS):
            color = FINGER_COLORS[finger]
            for a, b in zip(chain, chain[1:], strict=False):
                pygame.draw.line(surface, color, pts[a], pts[b], width)
        tips = set(F.FINGERTIPS)
        for i, p in enumerate(pts):
            r = round((7 if i in tips else 4) * max(1.0, vp.scale * 0.6))
            pygame.draw.circle(surface, (20, 20, 20), p, r + 2)
            pygame.draw.circle(surface, FINGER_COLORS[_finger_of(i)] if i else PALM_COLOR, p, r)
            if self.show_indices:
                text.draw(surface, str(i), (p[0] + r + 2, p[1] - r - 2), size=12, shadow=True)
        if s.pinching:
            pygame.draw.circle(surface, ACCENT, vp.to_px(*s.features.pinch_point), 18, 3)

    def _draw_hand_panel(self, surface: pygame.Surface, vp: Viewport, s: HandState) -> None:
        hand, f, cfg = s.tracked.hand, s.features, self.pinch_cfg
        lines = [
            f"#{s.id} {hand.handedness} {hand.handedness_score:.2f}",
            f"{hand.gesture} {hand.gesture_score:.2f}",
            f"fingers {f.finger_count}",
        ]
        pw, ph = 190, 18 * len(lines) + 46
        wx, wy = vp.to_px(float(hand.landmarks[F.WRIST, 0]), float(hand.landmarks[F.WRIST, 1]))
        sw, sh = surface.get_size()
        # Toward the screen middle (mirror view: right hand on the right), clear of the table.
        inward = wx - 30 - pw if hand.handedness == "Right" else wx + 30
        x = min(max(8, inward), sw - pw - 8)
        y = min(max(8, wy - ph // 2), sh - ph - 8)
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        surface.blit(panel, (x, y))
        for i, line in enumerate(lines):
            text.draw(surface, line, (x + 8, y + 6 + 18 * i), size=14)

        # Pinch-ratio bar: 0 .. 1.2 palm lengths, enter/exit thresholds marked.
        bx, by, bw, bh = x + 8, y + 12 + 18 * len(lines), pw - 16, 12
        full = 1.2
        lit = s.pinching
        pygame.draw.rect(surface, (60, 60, 70), (bx, by, bw, bh), border_radius=3)
        fill = min(f.pinch_ratio, full) / full
        pygame.draw.rect(
            surface,
            ACCENT if lit else (150, 150, 160),
            (bx, by, round(bw * fill), bh),
            border_radius=3,
        )
        for thr, color in ((cfg.enter, (90, 255, 140)), (cfg.exit, (255, 120, 90))):
            tx = bx + round(bw * thr / full)
            pygame.draw.line(surface, color, (tx, by - 3), (tx, by + bh + 3), 2)
        label = "PINCH" if lit else f"pinch {f.pinch_ratio:.2f}"
        text.draw(surface, label, (bx, by + bh + 3), ACCENT if lit else (200, 200, 200), size=12)

    def _draw_table(self, surface: pygame.Surface, s: HandState) -> None:
        lm = s.tracked.hand.landmarks
        sw, sh = surface.get_size()
        row_h, size = 16, 13
        pw, ph = 250, 28 + row_h * (len(lm) + 1) + 8
        x, y = sw - pw - 8, max(8, (sh - ph) // 2)
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        surface.blit(panel, (x, y))
        text.draw(surface, "What the computer sees", (x + 8, y + 6), ACCENT, size=14, bold=True)
        y0 = y + 28
        text.draw(surface, " #      x       y       z", (x + 8, y0), (150, 150, 160), size=size)
        tips = set(F.FINGERTIPS)
        for i in range(len(lm)):
            ry = y0 + row_h * (i + 1)
            color = (200, 200, 200)
            if i in tips:
                color = FINGER_COLORS[_finger_of(i)]
                pygame.draw.rect(surface, (*color, 40), (x + 4, ry - 1, pw - 8, row_h), 1)
            row = f"{i:2d}  {lm[i, 0]:6.3f}  {lm[i, 1]:6.3f}  {lm[i, 2]:6.3f}"
            text.draw(surface, row, (x + 8, ry), color, size=size)
