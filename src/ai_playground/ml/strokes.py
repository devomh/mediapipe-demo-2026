"""Air-written strokes (Air Writing spec §3), portable core: stdlib + NumPy only.

A stroke is an (N, 2) float32 array of screen points plus a capture timestamp per point; a
character is a list of strokes. Points are resampled to a fixed spacing, so ink density doesn't
depend on drawing speed and the eraser can work point by point.

From strokes to the network's input (spec §4), shared by training and the app so the framing
can't drift: `render_strokes` draws the ink at 112x112, `frame_image` crops, centres, pads and
area-averages any high-resolution ink image (ours, or an upsampled EMNIST glyph) to 28x28.
Collected samples are stored as JSON lines (`write_samples`, `read_samples`).
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MIN_POINTS = 3  # shorter strokes and erase fragments are dropped
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # the network's 36 classes (spec §5.1)
RENDER_SIZE = 112  # render_strokes output, 4x the network input
INPUT_SIZE = 28
SCHEMA = 1


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


# --- framing (spec §4) --------------------------------------------------------------------------
def render_strokes(
    strokes: Sequence[np.ndarray | Stroke], raster_width: float, size: int = RENDER_SIZE
) -> np.ndarray:
    """Ink in [0, 1] on a size x size float32 canvas. The character keeps its aspect ratio and is
    centred; the line is `raster_width` x the character's larger side wide, anti-aliased over 1
    pixel: each pixel's ink comes from its distance to the nearest segment (NumPy only).

    Strokes are resampled to <= 1 px segments on the canvas, so each segment only touches a small
    window of pixels around it; those windows are computed at once and max-ed into the image.
    """
    pts = [np.asarray(s.points if isinstance(s, Stroke) else s, np.float64)[:, :2] for s in strokes]
    pts = [p for p in pts if len(p)]
    img = np.zeros(size * size, np.float32)
    if not pts:
        return img.reshape(size, size)
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    side = max(float((hi - lo).max()), 1e-6)
    # Leave room for the line's half width (it scales with the character) plus 1 px of anti-alias.
    scale = (size - 2.0) / (side * (1.0 + raster_width))
    half = 0.5 * raster_width * side * scale
    centre = (lo + hi) / 2
    seg_a, seg_b = [], []
    for p in pts:
        q = _densify((p - centre) * scale + size / 2, 1.0)
        seg_a.append(q if len(q) == 1 else q[:-1])
        seg_b.append(q if len(q) == 1 else q[1:])
    a, b = np.concatenate(seg_a), np.concatenate(seg_b)
    k = int(math.ceil(2 * half + 4))  # window side: the line's width, 1 px segment, anti-alias
    off = np.stack(np.mgrid[0:k, 0:k], axis=-1).reshape(-1, 2)[:, ::-1]  # (k*k, 2) as x, y
    origin = np.floor((a + b) / 2).astype(np.int64) - k // 2  # (S, 2)
    cells = origin[:, None, :] + off[None, :, :]  # (S, k*k, 2) integer pixel coordinates
    pix = cells + 0.5
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)[:, None]
    ap = pix - a[:, None, :]
    t = np.clip((ap * ab[:, None, :]).sum(axis=2) / denom, 0.0, 1.0)
    d = ap - t[..., None] * ab[:, None, :]
    ink = np.clip(half + 0.5 - np.sqrt((d * d).sum(axis=2)), 0.0, 1.0).ravel()
    x, y = cells[..., 0].ravel(), cells[..., 1].ravel()
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size) & (ink > 0)
    np.maximum.at(img, y[inside] * size + x[inside], ink[inside].astype(np.float32))
    return img.reshape(size, size)


def _densify(q: np.ndarray, step: float) -> np.ndarray:
    """The polyline with extra points so no segment is longer than `step`."""
    if len(q) < 2:
        return q
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    n = np.maximum(1, np.ceil(seg / step).astype(np.int64))
    f = np.concatenate([np.arange(m) / m for m in n])  # fractions along each segment
    i = np.repeat(np.arange(len(seg)), n)
    out = q[i] + f[:, None] * (q[i + 1] - q[i])
    return np.concatenate([out, q[-1:]])


def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) weights that average input pixels over each output pixel's span."""
    edges = np.linspace(0.0, n_in, n_out + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    j = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
    return overlap / overlap.sum(axis=1, keepdims=True)


def frame_image(
    img_hi: np.ndarray, ink_threshold: float, fill: float, size: int = INPUT_SIZE
) -> np.ndarray:
    """Any ink image (H, W) in [0, 1] -> (size, size) float32, framed like EMNIST:
    crop to the ink, fit into a square keeping the aspect ratio (centred on the bounding box),
    pad so the character spans `fill` of the frame, area-average down, brightest pixel = 1."""
    img = np.asarray(img_hi, np.float32)
    ys, xs = np.nonzero(img > ink_threshold)
    if len(ys) == 0:
        return np.zeros((size, size), np.float32)
    crop = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    total = max(side, int(round(side / fill)))
    canvas = np.zeros((total, total), np.float32)
    y0, x0 = (total - h) // 2, (total - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = crop
    a = _area_matrix(total, size)
    out = (a @ canvas @ a.T).astype(np.float32)
    peak = float(out.max())
    return out / peak if peak > 0 else out


def strokes_to_input(
    strokes: Sequence[np.ndarray | Stroke], raster_width: float, ink_threshold: float, fill: float
) -> np.ndarray:
    """What the network sees: render_strokes, then frame_image."""
    return frame_image(render_strokes(strokes, raster_width), ink_threshold, fill)


# --- collected samples (spec §8) ----------------------------------------------------------------
@dataclass(slots=True)
class Sample:
    """One written character. Points are in window heights (resolution-independent), times in
    seconds from the sample's first point."""

    label: str
    strokes: list[np.ndarray]  # each (N, 3): x, y, t
    style: str = ""
    writer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> list[np.ndarray]:
        return [s[:, :2] for s in self.strokes]


def sample_from_strokes(
    label: str, strokes: Sequence[Stroke], height_px: float, style: str = "", writer: str = ""
) -> Sample:
    t0 = min(float(s.times[0]) for s in strokes)
    out = []
    for s in strokes:
        xyt = np.empty((len(s), 3), np.float64)
        xyt[:, :2] = s.points / height_px
        xyt[:, 2] = s.times - t0
        out.append(xyt)
    return Sample(label, out, style, writer)


def write_samples(path: Path | str, samples: Iterable[Sample], header: dict | None = None) -> None:
    """JSON lines: a header, then one sample per line. `.gz` paths are compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "schema": SCHEMA, **(header or {})}) + "\n")
        for smp in samples:
            obj = {
                "type": "sample",
                "label": smp.label,
                "style": smp.style,
                "writer": smp.writer,
                "strokes": [
                    [[round(float(x), 4), round(float(y), 4), round(float(t), 3)] for x, y, t in s]
                    for s in smp.strokes
                ],
            }
            if smp.meta:
                obj["meta"] = smp.meta
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")


def read_samples(path: Path | str) -> tuple[dict, list[Sample]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    header: dict | None = None
    samples: list[Sample] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj
            elif obj.get("type") == "sample":
                strokes = [np.asarray(s, np.float64).reshape(-1, 3) for s in obj["strokes"]]
                samples.append(
                    Sample(
                        obj["label"],
                        strokes,
                        obj.get("style", ""),
                        obj.get("writer", ""),
                        obj.get("meta", {}),
                    )
                )
    if header is None:
        raise ValueError(f"{path}: no header")
    return header, samples
