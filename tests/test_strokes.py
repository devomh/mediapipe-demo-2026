"""Air Writing strokes (spec §3, §11): resampling, erasing, tail trimming."""

import numpy as np
import pytest

from ai_playground.ml.strokes import (
    MIN_POINTS,
    Stroke,
    StrokeBuilder,
    bounds,
    char_size,
    erase,
    resample,
    trim_tail,
)


def line(n, x0=0.0, x1=100.0, y=50.0, t1=1.0):
    xs = np.linspace(x0, x1, n)
    return [(x, y) for x in xs], list(np.linspace(0.0, t1, n))


def spacing(s: Stroke) -> np.ndarray:
    return np.linalg.norm(np.diff(s.points, axis=0), axis=1)


@pytest.mark.parametrize("frames", [4, 15, 60])  # fast, medium, slow drawing
def test_resampling_spacing_does_not_depend_on_speed(frames):
    s = resample(*line(frames), step=3.0)
    d = spacing(s)
    assert np.allclose(d[:-1], 3.0, atol=1e-3)  # every gap but the end point's
    assert 0 < d[-1] <= 3.0
    assert s.points[0] == pytest.approx((0, 50))
    assert np.hypot(*(s.points[-1] - (100, 50))) <= 1.5  # ends within half a step of the input
    assert len(s) == pytest.approx(100 / 3 + 1, abs=1.5)


def test_resampled_times_are_interpolated_and_increasing():
    s = resample(*line(5, t1=2.0), step=1.0)
    assert np.all(np.diff(s.times) >= 0)
    mid = np.argmin(np.abs(s.points[:, 0] - 50))
    assert s.times[mid] == pytest.approx(1.0, abs=0.03)
    assert s.points.dtype == np.float32 and s.times.dtype == np.float64


def test_a_tap_is_not_a_stroke():
    b = StrokeBuilder(3.0)
    b.add((10, 10), 0.0)
    b.add((11, 10), 0.03)
    assert len(b) == 1 and b.finish() is None
    with pytest.raises(ValueError):
        StrokeBuilder(0)


def test_erasing_the_middle_splits_a_stroke_in_two():
    s = resample(*line(10), step=2.0)
    a, b = erase([s], (50, 50), 10)
    assert a.points[:, 0].max() < 40 and b.points[:, 0].min() > 60
    assert len(a) + len(b) < len(s)
    assert np.all(a.times < b.times[0])


def test_erasing_keeps_untouched_strokes_and_drops_crumbs():
    s = resample(*line(10), step=2.0)
    other = resample(*line(10, y=200), step=2.0)
    (left, kept) = erase([s, other], (100 - 2.0 * MIN_POINTS + 3, 50), 2.0 * MIN_POINTS)
    assert kept is other
    assert left.points[:, 0].max() < 100 - 2.0 * (MIN_POINTS - 1)  # the short tail is gone
    assert erase([s], (50, 50), 500) == []


def test_tail_trim_removes_exactly_the_trailing_window():
    s = resample(*line(30, t1=1.0), step=1.0)
    t = trim_tail(s, 0.15)
    assert t.times[-1] <= 0.85 + 1e-9
    assert np.all(s.times[len(t) :] > 0.85 - 1e-9)
    assert np.array_equal(t.points, s.points[: len(t)])
    assert trim_tail(s, 2.0) is None


def test_bounds_and_size():
    assert bounds([]) is None and char_size([]) == 0.0
    a = resample(*line(5, 10, 40, y=20), step=1.0)
    b = resample([(15, 0), (15, 90)], [0.0, 1.0], step=1.0)
    assert bounds([a, b]) == pytest.approx((10, 0, 40, 90))
    assert char_size([a, b]) == pytest.approx(90)
