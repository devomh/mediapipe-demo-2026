"""Air-written strokes (Air Writing spec §3), portable core: stdlib + NumPy only.

A stroke is an (N, 2) float32 array of screen points plus a capture timestamp per point; a
character is a list of strokes. Points are resampled to a fixed spacing, so ink density doesn't
depend on drawing speed and the eraser can work point by point.

W.2 adds `render_strokes` and `frame_image` here, shared by training and the app.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3  # shorter strokes and erase fragments are dropped


@dataclass(frozen=True, slots=True)
class Stroke:
    points: np.ndarray  # (N, 2) float32, screen px
    times: np.ndarray  # (N,) float64, capture time of each point

    def __len__(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        """Path length in px."""
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


class StrokeBuilder:
    """Collects one stroke while the pinch is held, resampling as points arrive.

    Between camera frames the path is taken as straight, and a point is placed every `step` px
    along it. The time of each placed point is interpolated the same way.
    """

    def __init__(self, step: float) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        self.step = step
        self._pts: list[tuple[float, float]] = []
        self._ts: list[float] = []
        self._raw: tuple[float, float, float] | None = None  # newest input point and time

    def add(self, p: tuple[float, float], t: float) -> None:
        x, y = float(p[0]), float(p[1])
        self._raw = (x, y, t)
        if not self._pts:
            self._pts.append((x, y))
            self._ts.append(t)
            return
        (lx, ly), lt = self._pts[-1], self._ts[-1]
        d = math.hypot(x - lx, y - ly)
        n = int(d // self.step)
        for k in range(1, n + 1):
            f = k * self.step / d
            self._pts.append((lx + f * (x - lx), ly + f * (y - ly)))
            self._ts.append(lt + f * (t - lt))

    @property
    def points(self) -> np.ndarray:
        """The resampled points so far, for drawing the stroke while it is written."""
        return np.asarray(self._pts, np.float32).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self._pts)

    def finish(self) -> Stroke | None:
        """The finished stroke, ending at the last input point; None if it is too short."""
        if self._raw is not None and self._pts:
            x, y, t = self._raw
            (lx, ly) = self._pts[-1]
            if math.hypot(x - lx, y - ly) > 0.5 * self.step:
                self._pts.append((x, y))
                self._ts.append(t)
        if len(self._pts) < MIN_POINTS:
            return None
        return Stroke(np.asarray(self._pts, np.float32), np.asarray(self._ts, np.float64))


def resample(points: Sequence[tuple[float, float]], times: Sequence[float], step: float):
    """A whole stroke at once (tests, and strokes loaded from files)."""
    b = StrokeBuilder(step)
    for p, t in zip(points, times, strict=True):
        b.add(p, t)
    return b.finish()


def erase(strokes: Sequence[Stroke], center: tuple[float, float], radius: float) -> list[Stroke]:
    """Remove the points inside the circle. A stroke cut in the middle splits in two; pieces
    under MIN_POINTS are dropped. Strokes the circle doesn't touch are returned as they are."""
    cx, cy = center
    out: list[Stroke] = []
    for s in strokes:
        d2 = (s.points[:, 0] - cx) ** 2 + (s.points[:, 1] - cy) ** 2
        keep = d2 > radius * radius
        if keep.all():
            out.append(s)
            continue
        # Runs of kept points: edges where `keep` switches on and off.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            if stop - start >= MIN_POINTS:
                out.append(Stroke(s.points[start:stop], s.times[start:stop]))
    return out


def trim_tail(stroke: Stroke, seconds: float) -> Stroke | None:
    """Drop the last `seconds` of the stroke: the hook left by a hand curling into a fist.
    None if too little is left."""
    keep = stroke.times <= stroke.times[-1] - seconds
    n = int(keep.sum())  # times are non-decreasing, so the kept points are a prefix
    if n < MIN_POINTS:
        return None
    return Stroke(stroke.points[:n], stroke.times[:n])


def cut_after(stroke: Stroke, t: float) -> Stroke | None:
    """Only the points up to time `t` (inclusive). None if too little is left."""
    n = int(np.searchsorted(stroke.times, t, side="right"))
    if n < MIN_POINTS:
        return None
    if n == len(stroke):
        return stroke
    return Stroke(stroke.points[:n], stroke.times[:n])


def bounds(strokes: Sequence[Stroke]) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) of all points, or None without ink."""
    pts = [s.points for s in strokes if len(s)]
    if not pts:
        return None
    allp = np.concatenate(pts)
    (x0, y0), (x1, y1) = allp.min(axis=0), allp.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def char_size(strokes: Sequence[Stroke]) -> float:
    """The larger side of the ink's bounding box, in px (0 without ink)."""
    b = bounds(strokes)
    return 0.0 if b is None else max(b[2] - b[0], b[3] - b[1])
