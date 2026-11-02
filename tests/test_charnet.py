"""CharNet NumPy inference (Air Writing spec §5.5, §11): parity with PyTorch, probabilities,
speed, framing agreement with the config, and an accuracy floor on the air-written set."""

import time
from pathlib import Path

import numpy as np
import pytest

from ai_playground import config
from ai_playground.ml.charnet import CharNet
from ai_playground.ml.strokes import CLASSES, read_samples, strokes_to_input

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets" / "charnet" / "charnet_v1.npz"
PARITY = ROOT / "tests" / "fixtures" / "charnet_parity.npz"
EVAL = ROOT / "tests" / "fixtures" / "strokes" / "eval_v1.jsonl.gz"

# A little under charnet_v1's measured 90.7 % top-1 (98/108) and 100 % top-3 (W.3; see
# assets/charnet/charnet_v1.json): a framing or weights change that silently breaks
# recognition fails here. Six of the ten misses are I->1 and O->0, look-alikes by design.
FLOOR_TOP1 = 0.85
FLOOR_TOP3 = 0.97


@pytest.fixture(scope="module")
def net():
    return CharNet(MODEL)


def test_numpy_matches_pytorch(net):
    z = np.load(PARITY)
    assert np.abs(net.logits(z["inputs"]) - z["logits"]).max() < 1e-4


def test_probabilities(net):
    x = np.load(PARITY)["inputs"]
    p = net.probs(x)
    assert p.shape == (len(x), 36) and np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()
    top = net.top(x[0])
    assert len(top) == 3 and top[0].p >= top[1].p >= top[2].p
    assert net.classes == CLASSES and net.temperature > 0


def test_one_read_is_fast(net):
    x = np.load(PARITY)["inputs"][0]
    net.top(x)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        net.top(x)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 0.015  # spec budget 5 ms; generous for CI runners


def test_the_app_frames_like_the_model_was_trained(net):
    w = config.load().writing
    assert (w.raster_width, w.ink_threshold, w.fill) == pytest.approx(
        (net.raster_width, net.ink_threshold, net.fill), abs=1e-6
    )


def test_accuracy_floor_on_the_air_written_set(net):
    _, samples = read_samples(EVAL)
    x = np.stack(
        [strokes_to_input(s.points, net.raster_width, net.ink_threshold, net.fill) for s in samples]
    )
    p = net.probs(x)
    y = np.array([CLASSES.index(s.label) for s in samples])
    top3 = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top1_acc = (top3[:, 0] == y).mean()
    top3_acc = (top3 == y[:, None]).any(axis=1).mean()
    assert top1_acc >= FLOOR_TOP1 and top3_acc >= FLOOR_TOP3, (top1_acc, top3_acc)
