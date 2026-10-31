"""LatestSlot: newest-value-wins handoff from the source worker to the main thread."""

from __future__ import annotations

import threading


class LatestSlot[T]:
    """One lock-protected value plus a sequence number. No queue, so latency can't build up."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None
        self._seq = 0

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._seq += 1

    def get(self) -> tuple[int, T | None]:
        with self._lock:
            return self._seq, self._value

    def get_newer(self, seen_seq: int) -> tuple[int, T | None]:
        """Return (seq, value) if something newer than `seen_seq` arrived, else (seen_seq, None)."""
        with self._lock:
            if self._seq > seen_seq:
                return self._seq, self._value
            return seen_seq, None

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
