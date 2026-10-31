"""Per-hand gesture state machines (portable core: stdlib + NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass

from ai_playground.config import BuiltinConfig, PinchConfig
from ai_playground.gestures.features import HandFeatures
from ai_playground.types import GestureEvent, Handedness


@dataclass(frozen=True, slots=True)
class DetectorInput:
    """Everything a detector sees about one hand on one frame."""

    hand_id: int
    handedness: Handedness
    t: float
    features: HandFeatures
    gesture: str  # MediaPipe built-in category this frame
    gesture_score: float
    pinch_vel: tuple[float, float]
    palm_vel: tuple[float, float]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class PinchDetector:
    """Thumb + index pinch with hysteresis (enter < exit) and a consecutive-frame debounce."""

    def __init__(self, cfg: PinchConfig) -> None:
        self.cfg = cfg
        self.active = False
        self._count = 0  # consecutive frames arguing for a state change

    def update(self, x: DetectorInput) -> list[GestureEvent]:
        cfg, ratio = self.cfg, x.features.pinch_ratio
        if not self.active:
            blocked = cfg.block_on_fist and x.gesture == "Closed_Fist"
            self._count = self._count + 1 if ratio < cfg.enter and not blocked else 0
            if self._count >= cfg.min_frames:
                self.active, self._count = True, 0
                return [self._event("pinch_start", x)]
            return []
        self._count = self._count + 1 if ratio > cfg.exit else 0
        if self._count >= cfg.min_frames:
            self.active, self._count = False, 0
            return [self._event("pinch_end", x, data={"reason": "released"})]
        return [self._event("pinch_move", x)]

    def end(self, x: DetectorInput, reason: str) -> list[GestureEvent]:
        if not self.active:
            return []
        self.active, self._count = False, 0
        return [self._event("pinch_end", x, data={"reason": reason})]

    def _event(self, type_: str, x: DetectorInput, data: dict | None = None) -> GestureEvent:
        return GestureEvent(
            type=type_,
            hand_id=x.hand_id,
            handedness=x.handedness,
            t=x.t,
            pos=x.features.pinch_point,
            strength=_clamp01(1.0 - x.features.pinch_ratio / self.cfg.enter),
            vel=x.pinch_vel,
            data=data or {},
        )


class BuiltinGestureDetector:
    """Debounced start/end events for one MediaPipe built-in category."""

    def __init__(self, category: str, name: str, cfg: BuiltinConfig) -> None:
        self.category, self.name, self.cfg = category, name, cfg
        self.active = False
        self._count = 0

    def update(self, x: DetectorInput, suppressed: bool = False) -> list[GestureEvent]:
        cfg = self.cfg
        if not self.active:
            seen = x.gesture == self.category and x.gesture_score >= cfg.min_score
            self._count = self._count + 1 if seen and not suppressed else 0
            if self._count >= cfg.start_frames:
                self.active, self._count = True, 0
                return [self._event(f"{self.name}_start", x)]
            return []
        # Ending only needs the category to be absent (any score), which adds hysteresis.
        self._count = self._count + 1 if x.gesture != self.category else 0
        if self._count >= cfg.end_frames:
            self.active, self._count = False, 0
            return [self._event(f"{self.name}_end", x, data={"reason": "released"})]
        return []

    def end(self, x: DetectorInput, reason: str) -> list[GestureEvent]:
        if not self.active:
            return []
        self.active, self._count = False, 0
        return [self._event(f"{self.name}_end", x, data={"reason": reason})]

    def _event(self, type_: str, x: DetectorInput, data: dict | None = None) -> GestureEvent:
        return GestureEvent(
            type=type_,
            hand_id=x.hand_id,
            handedness=x.handedness,
            t=x.t,
            pos=x.features.palm_center,
            strength=x.gesture_score if x.gesture == self.category else 0.0,
            vel=x.palm_vel,
            data=data or {},
        )
