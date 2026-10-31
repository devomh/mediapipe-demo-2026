"""GestureEngine: tracked hands -> events + per-hand state (portable core: stdlib + NumPy only)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ai_playground.config import GesturesConfig
from ai_playground.gestures import features
from ai_playground.gestures.detectors import BuiltinGestureDetector, DetectorInput, PinchDetector
from ai_playground.gestures.features import HandFeatures, VelocityTracker
from ai_playground.types import GestureEvent, Handedness, TrackedHand


@dataclass(frozen=True, slots=True)
class HandState:
    """A tracked hand plus everything computed about it this frame, for continuous control."""

    tracked: TrackedHand
    features: HandFeatures
    velocity: tuple[float, float]  # palm centre, normalized units/s
    pinching: bool
    fist: bool
    palm: bool

    @property
    def id(self) -> int:
        return self.tracked.id

    @property
    def handedness(self) -> Handedness:
        return self.tracked.hand.handedness

    @property
    def visible(self) -> bool:
        return self.tracked.missed == 0


class _HandSlot:
    def __init__(self, cfg: GesturesConfig) -> None:
        self.pinch = PinchDetector(cfg.pinch)
        self.fist = BuiltinGestureDetector("Closed_Fist", "fist", cfg.builtin)
        self.palm = BuiltinGestureDetector("Open_Palm", "palm", cfg.builtin)
        self.pinch_vel = VelocityTracker(cfg.velocity.window_s)
        self.palm_vel = VelocityTracker(cfg.velocity.window_s)
        self.last: DetectorInput | None = None
        self.state: HandState | None = None


class GestureEngine:
    def __init__(self, cfg: GesturesConfig, size: tuple[int, int] = (640, 480)) -> None:
        self.cfg = cfg
        self.size = size  # inference image size, for pixel-aspect features
        self._slots: dict[int, _HandSlot] = {}

    def reset(self) -> None:
        self._slots.clear()

    def update(
        self, tracked: Sequence[TrackedHand], t: float
    ) -> tuple[list[GestureEvent], list[HandState]]:
        events: list[GestureEvent] = []
        present = {th.id for th in tracked}

        for hid in [h for h in self._slots if h not in present]:
            slot = self._slots.pop(hid)
            if slot.last is None:
                continue
            lost = replace(slot.last, t=t)
            for det in (slot.pinch, slot.fist, slot.palm):
                events += det.end(lost, "lost")
            events.append(_lifecycle("hand_exit", lost, {"reason": "lost"}))

        states: list[HandState] = []
        for th in tracked:
            slot = self._slots.get(th.id)
            if slot is None:
                slot = self._slots[th.id] = _HandSlot(self.cfg)
            if th.missed > 0 and slot.state is not None:
                # Held during grace: keep the last features, no detector updates, no events.
                slot.state = replace(slot.state, tracked=th)
                states.append(slot.state)
                continue
            hand = th.hand
            f = features.compute(hand.landmarks, hand.world, self.size, self.cfg.fingers)
            x = DetectorInput(
                hand_id=th.id,
                handedness=hand.handedness,
                t=t,
                features=f,
                gesture=hand.gesture,
                gesture_score=hand.gesture_score,
                pinch_vel=slot.pinch_vel.update(t, f.pinch_point),
                palm_vel=slot.palm_vel.update(t, f.palm_center),
            )
            if slot.last is None:
                events.append(_lifecycle("hand_enter", x))
            slot.last = x
            events += slot.pinch.update(x)
            events += slot.fist.update(x, suppressed=slot.pinch.active)
            events += slot.palm.update(x)
            slot.state = HandState(
                tracked=th,
                features=f,
                velocity=x.palm_vel,
                pinching=slot.pinch.active,
                fist=slot.fist.active,
                palm=slot.palm.active,
            )
            states.append(slot.state)
        return events, states


def _lifecycle(type_: str, x: DetectorInput, data: dict | None = None) -> GestureEvent:
    return GestureEvent(
        type=type_,
        hand_id=x.hand_id,
        handedness=x.handedness,
        t=x.t,
        pos=x.features.palm_center,
        vel=x.palm_vel,
        data=data or {},
    )
