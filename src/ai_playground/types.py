"""Core data types shared by every layer (stdlib + NumPy only)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Handedness = Literal["Left", "Right"]

NUM_LANDMARKS = 21


@dataclass(frozen=True, slots=True)
class Hand:
    handedness: Handedness  # from the visitor's point of view (frame is mirrored before inference)
    handedness_score: float
    landmarks: np.ndarray  # (21, 3) float32; x, y normalized [0, 1] in the mirrored image
    world: np.ndarray  # (21, 3) float32; metres, origin near the hand's centre
    gesture: str  # MediaPipe built-in category, "None" if no gesture
    gesture_score: float


@dataclass(frozen=True, slots=True)
class HandFrame:
    seq: int  # increments per frame produced by the source
    t_capture: float  # time.monotonic() when the camera frame was read
    t_result: float  # time.monotonic() when inference finished
    size: tuple[int, int]  # (width, height) of the inference image
    hands: tuple[Hand, ...]
    image: np.ndarray | None  # RGB, mirrored; None in replay mode


@dataclass(frozen=True, slots=True)
class TrackedHand:
    """Produced by HandTracker on the main thread."""

    id: int  # stable while the hand stays visible
    hand: Hand  # with smoothed landmarks
    age: int  # frames since first seen
    missed: int = 0  # consecutive frames without a detection; > 0 means held during grace


@dataclass(frozen=True, slots=True)
class GestureEvent:
    type: str  # e.g. "pinch_start", "pinch_move", "pinch_end", "fist_start", "custom"
    hand_id: int
    handedness: Handedness
    t: float
    pos: tuple[float, float]  # normalized screen coordinates [0, 1], mirrored
    strength: float = 0.0  # 0..1, meaning depends on type
    vel: tuple[float, float] = (0.0, 0.0)  # normalized units per second
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """The event shape proposed in the brainstorm, plus hand identity and time."""
        return {
            "gesture": self.type,
            "hand": self.hand_id,
            "handedness": self.handedness,
            "t": round(self.t, 4),
            "position": [round(self.pos[0], 4), round(self.pos[1], 4)],
            "strength": round(self.strength, 4),
            "velocity": [round(self.vel[0], 4), round(self.vel[1], 4)],
            "data": dict(self.data),
        }
