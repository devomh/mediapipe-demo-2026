"""Timing statistics and the HUD (camera FPS, inference, render FPS, pipeline latency)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from ai_playground.render import text
from ai_playground.render.ui import px


class RollingWindow:
    """(t, value) samples kept for the last `window_s` seconds."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._s: deque[tuple[float, float]] = deque()

    def add(self, t: float, value: float) -> None:
        self._s.append((t, value))
        while self._s and t - self._s[0][0] > self.window_s:
            self._s.popleft()

    def values(self) -> np.ndarray:
        return np.fromiter((v for _, v in self._s), dtype=np.float64, count=len(self._s))

    def mean(self) -> float:
        v = self.values()
        return float(v.mean()) if len(v) else float("nan")

    def p95(self) -> float:
        v = self.values()
        return float(np.percentile(v, 95)) if len(v) else float("nan")

    def rate(self) -> float:
        """Samples per second, from the timestamps of the first and last sample."""
        if len(self._s) < 2:
            return 0.0
        span = self._s[-1][0] - self._s[0][0]
        return (len(self._s) - 1) / span if span > 0 else 0.0


class Stats:
    def __init__(self, window_s: float) -> None:
        self.camera = RollingWindow(window_s)  # t_capture of each new frame
        self.inference = RollingWindow(window_s)  # seconds
        self.latency = RollingWindow(window_s)  # seconds, display time - t_capture
        self.render_fps = 0.0


def _ms(x: float) -> str:
    return "  --" if x != x else f"{x * 1000:4.0f}"  # NaN check


class Hud:
    def __init__(self, stats: Stats, visible: bool = True) -> None:
        self.stats = stats
        self.visible = visible

    def draw(
        self,
        surface: pygame.Surface,
        source_label: str,
        hands: int,
        recording_s: float | None,
        status: str = "",
    ) -> None:
        if not self.visible:
            return
        s = self.stats
        lines = [
            f"{source_label}   hands {hands}",
            f"camera {s.camera.rate():5.1f} fps   infer {_ms(s.inference.mean())} ms",
            f"render {s.render_fps:5.1f} fps",
            f"latency {_ms(s.latency.mean())} ms mean {_ms(s.latency.p95())} ms p95",
        ]
        if status:
            lines.append(status)
        row = px(18)
        w = max(text.font(14).size(line)[0] for line in lines) + px(16)
        h = row * len(lines) + px(10) + (px(20) if recording_s is not None else 0)
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        surface.blit(panel, (px(8), px(8)))
        for i, line in enumerate(lines):
            text.draw(surface, line, (px(16), px(13) + row * i), size=14)
        if recording_s is not None:
            y = px(13) + row * len(lines) + px(9)
            pygame.draw.circle(surface, (235, 50, 50), (px(22), y), px(6))
            text.draw(
                surface, f"REC {recording_s:5.1f}s", (px(34), y), (255, 120, 120), anchor="midleft"
            )
