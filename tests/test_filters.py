import numpy as np

from ai_playground.config import OneEuroConfig
from ai_playground.tracking.filters import OneEuroFilter

FPS = 30.0


def run(cfg, signal):
    f = OneEuroFilter(cfg)
    return np.array([f(np.array([x]), i / FPS)[0] for i, x in enumerate(signal)])


def test_first_sample_passes_through(cfg):
    f = OneEuroFilter(cfg.tracking.one_euro.image)
    x = np.arange(63, dtype=np.float32).reshape(21, 3)
    assert np.array_equal(f(x, 0.0), x)


def test_non_increasing_time_returns_previous(cfg):
    f = OneEuroFilter(cfg.tracking.one_euro.image)
    a = f(np.zeros(3), 1.0)
    assert np.array_equal(f(np.ones(3), 1.0), a)


def test_reduces_jitter_on_a_still_hand(cfg):
    rng = np.random.default_rng(1)
    noisy = 0.5 + rng.normal(0.0, 0.002, 300)  # MediaPipe-like jitter, normalized units
    out = run(cfg.tracking.one_euro.image, noisy)[30:]
    assert out.std() < noisy[30:].std() / 2.5


def test_step_response_is_fast(cfg):
    """A sudden 0.2 jump (fast hand move) must be 90 % followed within 3 frames."""
    sig = np.r_[np.full(30, 0.3), np.full(30, 0.5)]
    out = run(cfg.tracking.one_euro.image, sig)
    after = out[30:]
    first = int(np.argmax(after >= 0.3 + 0.9 * 0.2))
    assert after[first] >= 0.48 and first <= 3


def test_pure_low_pass_without_beta_is_slow():
    """Sanity check of the speed term: with beta = 0 the same step takes much longer."""
    sig = np.r_[np.full(30, 0.3), np.full(30, 0.5)]
    out = run(OneEuroConfig(min_cutoff=1.0, beta=0.0, d_cutoff=1.0), sig)[30:]
    assert int(np.argmax(out >= 0.48)) > 8


def test_vectorized_over_landmarks(cfg):
    f = OneEuroFilter(cfg.tracking.one_euro.world)
    x = np.zeros((21, 3), dtype=np.float32)
    f(x, 0.0)
    y = f(x + 0.01, 1 / FPS)
    assert y.shape == (21, 3) and y.dtype == np.float32
    assert np.allclose(y, y[0, 0])
