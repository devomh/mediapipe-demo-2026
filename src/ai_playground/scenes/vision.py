"""Vision scene: the mirrored camera with what the computer sees on top.

The skeleton, panels, and table come from the app-level Vision overlay, so the same "discover"
layer works on top of every later scene.
"""

from __future__ import annotations

from collections.abc import Sequence

import pygame

from ai_playground.gestures.engine import HandState
from ai_playground.scenes.base import AppContext
from ai_playground.types import GestureEvent


class VisionScene:
    name = "vision"

    def __init__(self) -> None:
        self.ctx: AppContext | None = None

    def enter(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def exit(self) -> None:
        self.ctx = None

    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        assert self.ctx is not None
        frame = self.ctx.frame
        self.ctx.background.draw(surface, self.ctx.viewport, frame.image if frame else None)
