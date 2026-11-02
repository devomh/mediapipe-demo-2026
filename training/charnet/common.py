"""Shared by the CharNet training scripts (Air Writing spec §5): model, batched framing, and
augmentation, all on the GPU.

`strokes.py` is the app's own `src/ai_playground/ml/strokes.py` (NumPy only). In Colab it is
uploaded next to these scripts; locally it is imported from the package. Framing must be
identical in training and in the app, so `frame_batch` is a batched port of
`strokes.frame_image`, checked against it by `compare_framing.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    import strokes  # Colab: uploaded next to this file
except ImportError:  # local runs from the repo
    from ai_playground.ml import strokes

CLASSES = strokes.CLASSES
UP = 112  # EMNIST 28x28 -> 112x112, the size render_strokes draws ours at
CANVAS = 144  # room around the glyph so rotation and shear don't clip it


# --- model --------------------------------------------------------------------------------------
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CharNet(nn.Module):
    """VGG-style, ~0.6 M parameters: 28 -> 14 -> 7 -> 3 (spec §5.4)."""

    def __init__(self, n_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# --- framing ------------------------------------------------------------------------------------
def frame_batch(img: torch.Tensor, ink_threshold: float, fill: float, size: int = 28):
    """`strokes.frame_image` for a batch (B, 1, H, W) -> (B, 1, size, size).

    Crop + square padding + area averaging is a linear map per image: out = R @ img @ C^T,
    where R[i, j] is the overlap of image row j (shifted into the padded square) with output
    row i, divided by the output row's span. Same for columns.
    """
    b, _, h, w = img.shape
    x = img[:, 0]
    mask = x > ink_threshold
    rows, cols = mask.any(dim=2), mask.any(dim=1)  # (B, H), (B, W)
    empty = ~rows.any(dim=1)
    ar_h = torch.arange(h, device=img.device)
    ar_w = torch.arange(w, device=img.device)
    big = torch.iinfo(torch.int64).max
    ymin = torch.where(rows, ar_h, big).min(dim=1).values
    ymax = torch.where(rows, ar_h, -1).max(dim=1).values
    xmin = torch.where(cols, ar_w, big).min(dim=1).values
    xmax = torch.where(cols, ar_w, -1).max(dim=1).values
    ymin, ymax = torch.where(empty, 0, ymin), torch.where(empty, 0, ymax)
    xmin, xmax = torch.where(empty, 0, xmin), torch.where(empty, 0, xmax)
    hh, ww = ymax - ymin + 1, xmax - xmin + 1
    side = torch.maximum(hh, ww)
    total = torch.maximum(side, torch.round(side.double() / fill).long())  # half to even
    y0, x0 = (total - hh) // 2, (total - ww) // 2

    def weights(lo_px, hi_px, off, n):
        ar = torch.arange(n, device=img.device, dtype=torch.float64)
        c = ar[None, :] - lo_px[:, None] + off[:, None]  # canvas coordinate of each pixel
        valid = (ar[None, :] >= lo_px[:, None]) & (ar[None, :] <= hi_px[:, None])
        span = total.double()[:, None] / size  # (B, 1)
        i = torch.arange(size, device=img.device, dtype=torch.float64)[None, :]
        bin_lo, bin_hi = (i * span)[:, :, None], ((i + 1) * span)[:, :, None]  # (B, size, 1)
        cc = c[:, None, :]  # (B, 1, n)
        ov = (torch.minimum(bin_hi, cc + 1) - torch.maximum(bin_lo, cc)).clamp(min=0)
        return ov * valid[:, None, :] / span[:, :, None]  # (B, size, n)

    r = weights(ymin, ymax, y0, h)
    c = weights(xmin, xmax, x0, w)
    out = (r @ x.double() @ c.transpose(1, 2)).float()
    peak = out.amax(dim=(1, 2), keepdim=True)
    out = torch.where(peak > 0, out / peak.clamp(min=1e-12), out)
    out = torch.where(empty[:, None, None], 0.0, out)
    return out[:, None]


def upsample(x28: torch.Tensor) -> torch.Tensor:
    """(B, 28, 28) uint8 EMNIST -> (B, 1, CANVAS, CANVAS) float in [0, 1], glyph in the middle."""
    x = x28[:, None].float() / 255.0
    x = F.interpolate(x, size=(UP, UP), mode="bilinear", align_corners=False)
    p = (CANVAS - UP) // 2
    return F.pad(x, (p, p, p, p))


def emnist_input(x28: torch.Tensor, ink_threshold: float, fill: float) -> torch.Tensor:
    """EMNIST glyphs as the network sees them, without augmentation."""
    return frame_batch(upsample(x28), ink_threshold, fill)


# --- augmentation (spec §5.3) -------------------------------------------------------------------
def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def augment(x: torch.Tensor, g: torch.Generator | None = None) -> torch.Tensor:
    """(B, 1, S, S) EMNIST ink -> the same, looking more like air-writing: thicker or thinner
    lines, crisper edges, tilted, slanted, squashed, wobbly, and sometimes with a gap."""
    b, _, s, _ = x.shape
    dev = x.device

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=dev, generator=g)

    # Line width: our ink has one width per character, EMNIST's varies by writer and pen.
    # 112 px glyphs have ~10 px lines; +-2-4 px is roughly 0.6-1.6x.
    op = torch.randint(0, 5, (b,), device=dev, generator=g)
    variants = torch.stack([x, _dilate(x, 3), _dilate(x, 5), _erode(x, 3), x], dim=0)
    x = variants[op, torch.arange(b, device=dev)]

    # Crisp edges (p = 0.5): EMNIST glyphs are blurred scans, our ink is drawn with a 1-pixel
    # anti-aliased edge (compare_framing grid).
    crisp = (torch.rand(b, device=dev, generator=g) < 0.5)[:, None, None, None]
    x = torch.where(crisp, ((x - 0.3) * 2.5).clamp(0.0, 1.0), x)

    # Affine: rotation +-12 deg, shear +-0.3, scale 0.8-1.15 per axis.
    ang = u(-12, 12, b) * math.pi / 180
    shear = u(-0.3, 0.3, b)
    sx, sy = u(0.8, 1.15, b), u(0.8, 1.15, b)
    cos, sin = torch.cos(ang), torch.sin(ang)
    # theta maps output coords to input coords, so divide by the scale.
    a11, a12 = cos / sx, (-sin + shear * cos) / sx
    a21, a22 = sin / sy, (cos + shear * sin) / sy
    zero = torch.zeros(b, device=dev)
    theta = torch.stack([torch.stack([a11, a12, zero], 1), torch.stack([a21, a22, zero], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)

    # Elastic wobble: a smooth random displacement field (hand tremor).
    alpha = u(0.0, 0.035, b)[:, None, None, None]
    noise = torch.randn(b, 2, 5, 5, device=dev, generator=g) * alpha
    disp = F.interpolate(noise, size=(s, s), mode="bicubic", align_corners=False)
    grid = grid + disp.permute(0, 2, 3, 1)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Gaps (p = 0.15): one or two discs of ink removed, like an early pinch release.
    gap = torch.rand(b, device=dev, generator=g) < 0.15
    if gap.any():
        yy, xx = torch.meshgrid(
            torch.arange(s, device=dev), torch.arange(s, device=dev), indexing="ij"
        )
        for _ in range(2):
            w = x[:, 0].flatten(1) + 1e-6
            idx = torch.multinomial(w, 1, generator=g)[:, 0]
            cy, cx = (idx // s).float(), (idx % s).float()
            r = u(5.0, 9.0, b)
            disc = ((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2) < (
                r[:, None, None] ** 2
            )
            keep = gap & (torch.rand(b, device=dev, generator=g) < 0.7)
            x = torch.where((disc & keep[:, None, None])[:, None], 0.0, x)
    return x.clamp(0.0, 1.0)


# --- metrics ------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a proportion k/n (Wilson score)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
