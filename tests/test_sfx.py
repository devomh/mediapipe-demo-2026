"""Sound effects (Phase A spec §4.1): synthesis, impact limiter, and the mixer wrapper.

The mixer runs on SDL's dummy audio driver (conftest), so nothing is heard and no device is needed.
"""

import time

import numpy as np
import pygame
import pytest

from ai_playground.audio import sfx
from ai_playground.audio.sfx import (
    BUCKET_NOTES,
    SIZE_BUCKETS,
    TIMBRES,
    Hit,
    ImpactLimiter,
    Sfx,
    impact_name,
    midi_hz,
    size_bucket,
    synthesize_bank,
)

RATE = 48_000


@pytest.fixture(scope="module")
def bank():
    return synthesize_bank(RATE, inflate_s=1.5)


def test_bank_has_every_sound(bank):
    impacts = {impact_name(k, b) for k in TIMBRES for b in range(SIZE_BUCKETS)}
    assert set(bank) == impacts | {"spawn", "grab", "throw", "clear", "inflate"}
    assert len(impacts) == 25


def test_every_sound_is_finite_bounded_and_click_free(bank):
    for name, x in bank.items():
        assert x.dtype == np.float32 and x.ndim == 1 and len(x) > 0, name
        assert np.all(np.isfinite(x)), name
        assert np.max(np.abs(x)) <= 1.0, name
        assert abs(float(x[0])) < 1e-3 and abs(float(x[-1])) < 1e-3, name  # faded edges


def test_lengths(bank):
    for kind in TIMBRES:
        assert len(bank[impact_name(kind, 0)]) <= 0.8 * RATE
    assert len(bank["grab"]) < 0.05 * RATE
    assert len(bank["inflate"]) == pytest.approx((1.5 + 0.15) * RATE, abs=2)


def test_bank_is_deterministic_and_quick():
    t0 = time.perf_counter()
    a = synthesize_bank(RATE, 1.5)
    elapsed = time.perf_counter() - t0
    b = synthesize_bank(RATE, 1.5)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert elapsed < 0.5  # spec: < 100 ms; generous here for slow CI runners


def _peak_hz(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / RATE)[np.argmax(spec)])


@pytest.mark.parametrize("bucket", range(SIZE_BUCKETS))
def test_circle_impacts_ring_at_their_note(bank, bucket):
    expected = midi_hz(BUCKET_NOTES[bucket])
    assert _peak_hz(bank[impact_name("circle", bucket)]) == pytest.approx(expected, rel=0.02)


def test_bigger_shapes_sound_lower(bank):
    for kind in TIMBRES:
        peaks = [_peak_hz(bank[impact_name(kind, b)]) for b in range(SIZE_BUCKETS)]
        assert peaks == sorted(peaks, reverse=True), kind


def test_size_buckets_are_monotonic_and_clamped():
    sizes = np.geomspace(0.01, 1.0, 50)
    buckets = [size_bucket(s, 0.03, 0.3) for s in sizes]
    assert buckets == sorted(buckets)
    assert buckets[0] == 0 and buckets[-1] == SIZE_BUCKETS - 1
    assert set(buckets) == set(range(SIZE_BUCKETS))


# --- limiter ------------------------------------------------------------------------------------
def hit(key, volume):
    return Hit(key, volume, 0.5, "impact.circle.0")


def test_limiter_keeps_the_loudest_per_frame():
    lim = ImpactLimiter(max_per_frame=3, cooldown_s=0.08)
    chosen = lim.select([hit(i, v) for i, v in enumerate([0.1, 0.9, 0.5, 0.7, 0.3])], t=0.0)
    assert [h.volume for h in chosen] == [0.9, 0.7, 0.5]


def test_limiter_counts_a_piece_once_per_frame():
    lim = ImpactLimiter(3, 0.08)
    chosen = lim.select([hit(1, 0.2), hit(1, 0.8), hit(2, 0.5)], t=0.0)
    assert [(h.key, h.volume) for h in chosen] == [(1, 0.8), (2, 0.5)]


def test_limiter_cooldown_per_piece():
    lim = ImpactLimiter(3, 0.08)
    assert lim.select([hit(1, 0.5)], 0.0)
    assert lim.select([hit(1, 0.5)], 0.05) == []  # still cooling down
    assert lim.select([hit(2, 0.5)], 0.05)  # another piece is fine
    assert lim.select([hit(1, 0.5)], 0.09)


def test_limiter_skips_cooling_pieces_without_using_up_the_frame():
    lim = ImpactLimiter(1, 0.08)
    lim.select([hit(1, 0.9)], 0.0)
    chosen = lim.select([hit(1, 0.9), hit(2, 0.1)], 0.01)
    assert [h.key for h in chosen] == [2]


# --- mixer --------------------------------------------------------------------------------------
@pytest.fixture
def mixer():
    pygame.init()
    yield
    pygame.quit()


def test_open_plays_everything_on_the_dummy_driver(mixer):
    s = Sfx.open(enabled=True, volume=0.7, inflate_s=1.5)
    assert s.available and pygame.mixer.get_num_channels() == sfx.NUM_CHANNELS
    for name in ("spawn", "grab", "throw", "clear", impact_name("bar", 4)):
        s.play(name, 0.5, x=0.2)
    s.play("no-such-sound")  # ignored
    s.start_inflate(0.8)
    s.stop_inflate()
    s.toggle_mute()
    assert s.muted
    s.play("spawn")  # muted: nothing, no crash
    s.close()


def test_disabled_sound_is_silent(mixer):
    s = Sfx.open(enabled=False, volume=0.7, inflate_s=1.5)
    assert not s.available
    s.play("spawn")
    s.start_inflate()
    s.stop_inflate()
    s.close()


def test_a_mixer_that_fails_to_open_leaves_sound_off(mixer, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise pygame.error("no audio device")

    monkeypatch.setattr(pygame.mixer, "init", broken)
    s = Sfx.open(enabled=True, volume=0.7, inflate_s=1.5)
    assert not s.available and "no audio device" in caplog.text
    s.play("spawn")
    s.start_inflate()
    s.stop_inflate()
    s.toggle_mute()
    s.close()


def test_unopened_sfx_is_a_silent_no_op():
    s = Sfx()
    assert not s.available
    s.play("spawn")
    s.start_inflate()
    s.stop_inflate()
    s.close()


def test_pan_is_constant_power_and_never_one_sided():
    for x in np.linspace(0, 1, 11):
        left, right = sfx._pan(1.0, x)
        assert left**2 + right**2 == pytest.approx(1.0)
        assert min(left, right) > 0.2
    assert sfx._pan(1.0, 0.0)[0] > sfx._pan(1.0, 0.0)[1]
    assert sfx._pan(1.0, 1.0)[1] > sfx._pan(1.0, 1.0)[0]


def test_conversion_to_mixer_formats():
    x = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    a = sfx._to_mixer(x, -16, 2)
    assert a.dtype == np.int16 and a.shape == (4, 2) and a[1, 0] == 32767 and a[2, 1] == -32767
    b = sfx._to_mixer(x, 32, 1)
    assert b.dtype == np.float32 and b.shape == (4, 1)
    with pytest.raises(ValueError):
        sfx._to_mixer(x, 8, 2)
