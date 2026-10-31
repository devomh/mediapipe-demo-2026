"""Normalized image coordinates <-> screen pixels, with *cover* scaling (image fills the window)."""

from __future__ import annotations


class Viewport:
    def __init__(self, window_size: tuple[int, int], image_size: tuple[int, int]) -> None:
        self.window_size = window_size
        self.image_size = image_size
        self._recompute()

    def update(
        self, window_size: tuple[int, int] | None = None, image_size: tuple[int, int] | None = None
    ) -> None:
        changed = False
        if window_size is not None and window_size != self.window_size:
            self.window_size, changed = window_size, True
        if image_size is not None and image_size != self.image_size:
            self.image_size, changed = image_size, True
        if changed:
            self._recompute()

    def _recompute(self) -> None:
        (ww, wh), (iw, ih) = self.window_size, self.image_size
        self.scale = max(ww / iw, wh / ih)
        self.dest_size = (round(iw * self.scale), round(ih * self.scale))
        self.offset = ((ww - self.dest_size[0]) / 2.0, (wh - self.dest_size[1]) / 2.0)

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        return (
            round(self.offset[0] + x * self.dest_size[0]),
            round(self.offset[1] + y * self.dest_size[1]),
        )

    def to_norm(self, px: float, py: float) -> tuple[float, float]:
        return (
            (px - self.offset[0]) / self.dest_size[0],
            (py - self.offset[1]) / self.dest_size[1],
        )

    def length_px(self, normalized_height: float) -> float:
        """A length given in image heights, in pixels."""
        return normalized_height * self.dest_size[1]
