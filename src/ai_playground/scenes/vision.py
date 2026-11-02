"""Vision scene: the mirrored camera with what the computer sees on top.

The skeleton, panels, and table come from the app-level Vision overlay, so the same "discover"
layer works on top of every later scene.
"""

from __future__ import annotations

from collections.abc import Sequence

import pygame

from ai_playground.booth import Hints
from ai_playground.gestures.engine import HandState
from ai_playground.scenes.base import AppContext
from ai_playground.types import GestureEvent


class VisionScene:
    name = "vision"
    show_vision = True

    def __init__(self) -> None:
        self.ctx: AppContext | None = None
        self.hints = Hints(
            {
                "raise": "Raise your hand: the computer finds 21 points on it",
                "gesture": "Make a fist, open your hand, or pinch, and watch the table",
            }
        )

    def enter(self, ctx: AppContext) -> None:
        self.ctx = ctx
        ctx.background.mode = "full"
        ctx.background.empty = "grid"

    def exit(self) -> None:
        self.ctx = None

    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None:
        if any(h.visible for h in hands):
            self.hints.mark("raise")
        if any(e.type in ("fist_start", "palm_start", "pinch_start") for e in events):
            self.hints.mark("gesture")

    def hint(self) -> str | None:
        return self.hints.first(self.hints.texts)

    def idle_reset(self) -> None:
        self.hints.reset()

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        assert self.ctx is not None
        frame = self.ctx.frame
        self.ctx.background.draw(surface, self.ctx.viewport, frame.image if frame else None)
