"""CharNet inference in NumPy (Air Writing spec §5.5), portable core: stdlib + NumPy only.

The network is trained in PyTorch (training/charnet/) and exported with each BatchNorm folded
into the convolution before it, so inference is: 3 x [conv 3x3 + ReLU, conv 3x3 + ReLU,
max-pool 2], flatten (channel-major, as PyTorch), dense + ReLU, dense, softmax with temperature.
A convolution is a sliding-window view of the padded input times the kernel matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

N_CONV = 6


@dataclass(frozen=True, slots=True)
class Guess:
    label: str
    p: float


class CharNet:
    def __init__(self, path: Path | str) -> None:
        with np.load(path, allow_pickle=False) as z:
            self.convs = [
                (z[f"conv{i}_w"].astype(np.float32), z[f"conv{i}_b"].astype(np.float32))
                for i in range(N_CONV)
            ]
            self.fc = [
                (z[f"fc{i}_w"].astype(np.float32), z[f"fc{i}_b"].astype(np.float32)) for i in (1, 2)
            ]
            self.classes = str(z["classes"])
            self.temperature = float(z["temperature"])
            # The framing the network was trained with (spec §4); the app must use the same.
            self.raster_width = float(z["raster_width"])
            self.ink_threshold = float(z["ink_threshold"])
            self.fill = float(z["fill"])
            self.version = str(z["version"])
        # Kernels as matrices: (out, in*3*3), matching the window layout below.
        self._kmats = [w.reshape(w.shape[0], -1) for w, _ in self.convs]

    @staticmethod
    def _conv(h: np.ndarray, kmat: np.ndarray, b: np.ndarray) -> np.ndarray:
        """3x3 convolution, padding 1, then ReLU. h: (B, C, H, W) -> (B, O, H, W)."""
        n, c, hh, ww = h.shape
        hp = np.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)))
        win = sliding_window_view(hp, (3, 3), axis=(2, 3))  # (B, C, H, W, 3, 3)
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * hh * ww, c * 9)
        out = cols @ kmat.T + b  # (B*H*W, O)
        out = out.reshape(n, hh, ww, -1).transpose(0, 3, 1, 2)
        return np.maximum(out, 0.0)

    @staticmethod
    def _pool(h: np.ndarray) -> np.ndarray:
        n, c, hh, ww = h.shape
        h = h[:, :, : hh // 2 * 2, : ww // 2 * 2]  # floor, as PyTorch
        return h.reshape(n, c, hh // 2, 2, ww // 2, 2).max(axis=(3, 5))

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x: (28, 28) or (B, 28, 28) in [0, 1] -> (B, n_classes) logits."""
        h = np.asarray(x, np.float32)
        if h.ndim == 2:
            h = h[None]
        h = h[:, None]
        for i, ((_, b), kmat) in enumerate(zip(self.convs, self._kmats, strict=True)):
            h = self._conv(h, kmat, b)
            if i % 2 == 1:
                h = self._pool(h)
        h = h.reshape(len(h), -1)
        (w1, b1), (w2, b2) = self.fc
        h = np.maximum(h @ w1.T + b1, 0.0)
        return h @ w2.T + b2

    def probs(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def top(self, x: np.ndarray, k: int = 3) -> list[Guess]:
        """The k most likely characters for one 28x28 input."""
        p = self.probs(x)[0]
        idx = np.argsort(p)[::-1][:k]
        return [Guess(self.classes[i], float(p[i])) for i in idx]
