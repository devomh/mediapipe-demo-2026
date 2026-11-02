"""Scene protocol and the context every scene gets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pygame

from ai_playground.audio.sfx import Sfx
from ai_playground.config import Config
from ai_playground.gestures.engine import HandState
from ai_playground.render.background import Background
from ai_playground.render.viewport import Viewport
from ai_playground.types import GestureEvent, HandFrame


@dataclass
class AppContext:
    config: Config
    viewport: Viewport
    background: Background
    switch_scene: Callable[[str], None]
    frame: HandFrame | None = None  # newest frame from the source
    sfx: Sfx = field(default_factory=Sfx)  # sound effects; silent unless the app opened them
    custom_model: Any = None  # trained custom-gesture model (parked)
    audio: Any = None  # audio engine (Phase B)


class Scene(Protocol):
    name: str

    def enter(self, ctx: AppContext) -> None: ...
    def exit(self) -> None: ...
    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None: ...
    def update(self, dt: float) -> None: ...
    def draw(self, surface: pygame.Surface) -> None: ...
    def hint(self) -> str | None:
        """The onboarding line to show now (booth shell), or None."""
        ...

    def idle_reset(self) -> None:
        """No hands for a while: tidy up for the next visitor and bring the hints back."""
        ...
