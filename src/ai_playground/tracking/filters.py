"""Vectorized One Euro filter (Casiez et al., CHI 2012).

Low cutoff when the signal is slow (removes jitter), higher cutoff when it moves fast (removes lag).
"""

from __future__ import annotations

import math

import numpy as np

from ai_playground.config import OneEuroConfig


def _alpha(dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, cfg: OneEuroConfig) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = cfg.min_cutoff, cfg.beta, cfg.d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t = 0.0

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 1e-6:
            return self._x.copy()
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(dt, self.d_cutoff) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        self._x = (self._x + _alpha(dt, cutoff) * (x - self._x)).astype(np.float32)
        self._t = t
        return self._x.copy()

    def reset(self) -> None:
        self._x = self._dx = None
