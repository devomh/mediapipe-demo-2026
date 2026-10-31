"""Frame pacing for the main loop.

`pygame.time.Clock.tick(60)` sleeps with OS timers that oversleep, giving ~59 FPS even when a
frame needs only 5 ms of work (A.4 measurement). A pure busy-wait is exact but spins the main
thread, competing for the GIL with the camera/MediaPipe worker. This sleeps until shortly
before the deadline (releasing the GIL), then spins the last `spin_s`.
"""

from __future__ import annotations

import time
from collections import deque


class FramePacer:
    def __init__(self, fps: float, spin_s: float = 0.0015) -> None:
        self.period = 1.0 / fps
        self.spin_s = spin_s
        self._next = time.perf_counter() + self.period
        self._last = time.perf_counter()
        self._stamps: deque[float] = deque(maxlen=61)

    def tick(self) -> float:
        """Wait for the next frame; returns seconds since the previous tick."""
        now = time.perf_counter()
        if self._next - now > self.spin_s:
            time.sleep(self._next - now - self.spin_s)
        while time.perf_counter() < self._next:
            pass
        now = time.perf_counter()
        # A slow frame doesn't make the next ones rush to catch up: schedule from whichever is
        # later, the deadline or now.
        self._next = max(self._next, now) + self.period
        dt, self._last = now - self._last, now
        self._stamps.append(now)
        return dt

    @property
    def fps(self) -> float:
        """Frames per second over roughly the last second."""
        if len(self._stamps) < 2:
            return 0.0
        return (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])
