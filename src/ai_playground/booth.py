"""Booth-minimum shell logic (Phase A spec §6.0): idle reset, onboarding hints, hold-to-quit.

Pure state machines driven by the frame's `dt`, so tests run them without a window. Drawing is
in `render/booth.py`; the app wires them together.
"""

from __future__ import annotations

from collections.abc import Iterable


class IdleTimer:
    """Time with no hands in view. `update` returns True once per idle spell, at the timeout."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s  # 0 or less: never
        self.idle_s = 0.0
        self._fired = False

    def update(self, dt: float, hands_present: bool) -> bool:
        if hands_present:
            self.idle_s, self._fired = 0.0, False
            return False
        self.idle_s += dt
        if self._fired or self.timeout_s <= 0 or self.idle_s < self.timeout_s:
            return False
        self._fired = True
        return True


class HoldKey:
    """A key that acts only after being held `hold_s`. A short tap shows a reminder instead."""

    REMINDER_S = 1.5

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self.held_s: float | None = None  # None while the key is up
        self.reminder_s = 0.0  # counts down after a tap that was too short

    def press(self) -> None:
        if self.held_s is None:
            self.held_s = 0.0
            self.reminder_s = 0.0

    def release(self) -> None:
        """Key up, or the window lost focus (the key-up may never arrive)."""
        if self.held_s is not None:
            self.reminder_s = self.REMINDER_S
        self.held_s = None

    def update(self, dt: float) -> bool:
        """True when the hold is complete."""
        self.reminder_s = max(0.0, self.reminder_s - dt)
        if self.held_s is None:
            return False
        self.held_s += dt
        return self.held_s >= self.hold_s

    @property
    def progress(self) -> float:
        if self.held_s is None:
            return 0.0
        return 1.0 if self.hold_s <= 0 else min(1.0, self.held_s / self.hold_s)


class Hints:
    """A scene's onboarding hints in teaching order. Each one goes away for good once its action
    has been done; `reset` (an idle reset) brings them all back for the next visitor."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts  # key -> text, in order
        self.done: set[str] = set()

    def mark(self, key: str) -> None:
        if key not in self.texts:
            raise KeyError(key)
        self.done.add(key)

    def reset(self) -> None:
        self.done.clear()

    def first(self, applicable: Iterable[str]) -> str | None:
        """The text of the first hint, in order, that applies right now and hasn't been done."""
        ok = set(applicable)
        for key, text in self.texts.items():
            if key in ok and key not in self.done:
                return text
        return None
