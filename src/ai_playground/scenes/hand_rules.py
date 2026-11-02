"""Hand rules shared by scenes: an open palm that arms before it acts, and hold-to-trigger.

Geometry's paddle and Air Writing's eraser arm the same way (Geometry spec §2, "Changed in
A.3"; Air Writing spec §2): MediaPipe's debounced Open_Palm, held for `hold_s`, and not within
`after_pinch_s` of the same hand ending a pinch. Without the hold, a hand repositioning between
pinches would bat shapes away or erase the stroke just written.

Times are the scene's own clock (the sum of `update` dts), not capture timestamps.
"""

from __future__ import annotations


class PalmArming:
    def __init__(self, hold_s: float, after_pinch_s: float) -> None:
        self.hold_s, self.after_pinch_s = hold_s, after_pinch_s
        self._since: dict[int, float] = {}  # hand -> when its open palm started arming
        self._pinch_end: dict[int, float] = {}  # hand -> when it last ended a pinch

    def pinch_ended(self, hand_id: int, t: float) -> None:
        self._pinch_end[hand_id] = t

    def update(self, hand_id: int, t: float) -> float | None:
        """Call each frame while the hand is an open palm free to act. Returns how far it has
        armed (>= 1: armed), or None while it is too soon after a pinch."""
        end = self._pinch_end.get(hand_id)
        if end is not None and t - end < self.after_pinch_s:
            self.stop(hand_id)
            return None
        since = self._since.setdefault(hand_id, t)
        return (t - since) / self.hold_s if self.hold_s > 0 else 1.0

    def stop(self, hand_id: int) -> None:
        """The hand is not an open palm now: arming starts over next time."""
        self._since.pop(hand_id, None)

    def forget(self, hand_id: int) -> None:
        """The hand left."""
        self._since.pop(hand_id, None)
        self._pinch_end.pop(hand_id, None)

    def reset(self) -> None:
        self._since.clear()
        self._pinch_end.clear()


class HoldTrigger:
    """Fills while a condition holds and fires once when full. It stays full (and quiet) until the
    condition drops, so holding on doesn't fire again: open a hand to re-arm."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self.progress = 0.0  # 0..1
        self._latched = False

    def update(self, dt: float, active: bool) -> bool:
        if not active:
            self.reset()
            return False
        if self._latched:
            return False
        self.progress = min(1.0, self.progress + dt / self.hold_s) if self.hold_s > 0 else 1.0
        if self.progress >= 1.0:
            self._latched = True
            return True
        return False

    def reset(self) -> None:
        self.progress, self._latched = 0.0, False
