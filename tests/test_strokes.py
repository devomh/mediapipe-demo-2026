"""Air Writing strokes (spec §3, §11): resampling, erasing, tail trimming."""

import time

import numpy as np
import pytest

from ai_playground.ml.strokes import (
    CLASSES,
    INPUT_SIZE,
    MIN_POINTS,
    RENDER_SIZE,
    Sample,
    Stroke,
    StrokeBuilder,
    _area_matrix,
    bounds,
    char_size,
    erase,
    frame_image,
    read_samples,
    render_strokes,
    resample,
    sample_from_strokes,
    strokes_to_input,
    trim_tail,
    write_samples,
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


# --- framing (spec §4, §11) ---------------------------------------------------------------------
W, THR, FILL = 0.1, 0.1, 24 / 28


def seven(dx=0.0, dy=0.0, k=1.0):
    """A "7" with a crossbar, as two strokes in screen px."""

    def tr(pts):
        return np.array([(dx + k * x, dy + k * y) for x, y in pts], np.float32)

    return [tr([(0, 0), (60, 0), (20, 100)]), tr([(15, 50), (55, 50)])]


def test_render_is_ink_in_unit_range_and_empty_without_strokes():
    img = render_strokes(seven(), W)
    assert img.shape == (RENDER_SIZE, RENDER_SIZE) and img.dtype == np.float32
    assert img.min() == 0.0 and img.max() == 1.0
    assert not render_strokes([], W).any()
    assert render_strokes([np.array([[5.0, 5.0]])], W).max() == 1.0  # a dot is still ink


def test_framing_ignores_where_and_how_big_the_ink_was():
    """Spec §11: the same output under translation and scaling."""
    ref = strokes_to_input(seven(), W, THR, FILL)
    for dx, dy, k in [(300, 120, 1.0), (0, 0, 2.5), (-50, 900, 0.4)]:
        x = strokes_to_input(seven(dx, dy, k), W, THR, FILL)
        assert np.abs(x - ref).max() < 0.05, (dx, dy, k)


def test_framing_keeps_the_aspect_ratio():
    x = strokes_to_input([np.array([[0, 0], [0, 100]], np.float32)], W, THR, FILL)
    assert x.shape == (INPUT_SIZE, INPUT_SIZE) and x.max() == 1.0
    cols = np.nonzero(x.sum(axis=0) > 0.05)[0]
    rows = np.nonzero(x.sum(axis=1) > 0.05)[0]
    assert len(cols) <= 4 and abs(cols.mean() - 13.5) <= 0.5  # a "1" stays thin and centred
    assert 23 <= len(rows) <= 26  # spans `fill` of the frame (24 px) plus anti-aliasing


def test_frame_image_on_any_ink_image():
    img = np.zeros((200, 120), np.float32)
    img[30:150, 50:70] = 1.0  # an upsampled pen glyph: a tall bar off-centre
    x = frame_image(img, THR, FILL)
    rows = np.nonzero(x.sum(axis=1) > 0.05)[0]
    assert rows[0] == 2 and rows[-1] == 25  # 2-pixel border, like EMNIST
    assert not frame_image(np.zeros((50, 50)), THR, FILL).any()


def test_area_averaging_conserves_ink():
    a = _area_matrix(97, 28)
    assert np.allclose(a.sum(axis=1), 1.0)
    assert np.allclose(a.sum(axis=0).sum(), 28.0)


def test_render_is_fast_enough_for_a_read():
    long = [resample(*line(400, 0, 300, t1=4.0), step=0.5).points] * 3
    render_strokes(long, W)
    t0 = time.perf_counter()
    render_strokes(long, W)
    assert time.perf_counter() - t0 < 0.05  # measured 4-7 ms; generous for CI runners


# --- sample files -------------------------------------------------------------------------------
def test_samples_round_trip(tmp_path):
    a = resample(*line(10), step=2.0)
    b = resample(*line(10, y=80, t1=2.0), step=2.0)
    smp = sample_from_strokes("Z", [a, b], height_px=720, style="fast", writer="w1")
    assert smp.strokes[0][0, 2] == 0.0 and smp.strokes[0][:, :2].max() < 1.0  # window heights
    for name in ("s.jsonl", "s.jsonl.gz"):
        write_samples(tmp_path / name, [smp, Sample("0", smp.strokes)], {"writer": "w1"})
        header, back = read_samples(tmp_path / name)
        assert header["schema"] == 1 and header["writer"] == "w1"
        assert [s.label for s in back] == ["Z", "0"] and back[0].style == "fast"
        assert np.allclose(back[0].strokes[1], smp.strokes[1], atol=1e-3)
    assert len(CLASSES) == 36
