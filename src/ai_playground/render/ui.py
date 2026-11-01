"""One scale for everything drawn in fixed pixels: text, panels, cursors, line widths.

Overlays are laid out for a 720 px tall window (the default 1280x720). A taller window scales
them by its height, so a maximized or fullscreen window on a high-resolution monitor looks the
same as the default window, only bigger. Shapes and the camera image already scale on their own
(they are sized in window heights). The app calls `set_window_height` whenever the window changes.
"""

from __future__ import annotations

DESIGN_HEIGHT = 720
MIN_SCALE = 0.75  # keeps 1-2 px line widths >= 1 (pygame fills a shape drawn with width 0)
MAX_SCALE = 6.0

_factor = 1.0
_scale = 1.0


def set_factor(factor: float) -> None:
    """The operator's extra multiplier on top of the window scale (`[display] ui_scale`)."""
    global _factor
    _factor = factor


def set_window_height(height: int) -> None:
    global _scale
    _scale = min(MAX_SCALE, max(MIN_SCALE, _factor * height / DESIGN_HEIGHT))


def scale() -> float:
    return _scale


def px(v: float) -> int:
    """A length laid out at 720 px tall, in pixels at the current window size."""
    return round(v * _scale)
