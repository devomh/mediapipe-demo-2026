"""Framing check (Air Writing spec §5.3): do our air-written characters reach the network
looking like EMNIST's?

    python compare_framing.py --data /content/data --eval eval_v1.jsonl.gz --out framing.png

1. Parity: the batched training framing (`common.frame_batch`) against the app's
   `strokes.frame_image` on real EMNIST glyphs.
2. Line width: the mean ink of a framed image (0..1 per pixel) for EMNIST, and for our samples
   at several `raster_width` values. The closest match is the `raster_width` to use. This
   matches a statistic, not accuracy, so it doesn't tune anything to the evaluation set.
3. A grid per class: EMNIST, EMNIST augmented, ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import CLASSES, augment, emnist_input, frame_batch, strokes, upsample
from PIL import Image, ImageDraw

THR, FILL = 0.1, 0.857
WIDTHS = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/content/data"))
    ap.add_argument("--eval", type=Path, default=Path("/content/charnet/eval_v1.jsonl.gz"))
    ap.add_argument("--out", type=Path, default=Path("/content/charnet/framing.png"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data / "emnist36.npz")
    xt, yt = d["x_test"], d["y_test"]
    rng = np.random.default_rng(0)
    pick = np.concatenate(
        [rng.choice(np.flatnonzero(yt == c), 50, replace=False) for c in range(36)]
    )
    x28 = torch.tensor(xt[pick], device=dev)

    # 1. parity on real glyphs
    up = upsample(x28)
    ours = frame_batch(up, THR, FILL)[:, 0].cpu().numpy()
    ref = np.stack([strokes.frame_image(im, THR, FILL) for im in up[:, 0].cpu().numpy()])
    parity = float(np.abs(ours - ref).max())
    print(f"framing parity on {len(pick)} EMNIST glyphs: max |torch - numpy| = {parity:.2e}")
    assert parity < 1e-4

    # 2. line width
    em = emnist_input(x28, THR, FILL)[:, 0].cpu().numpy()
    emnist_ink = float(em.mean())
    _, samples = strokes.read_samples(a.eval)
    ink = {}
    for w in WIDTHS:
        ink[w] = float(
            np.mean([strokes.strokes_to_input(s.points, w, THR, FILL).mean() for s in samples])
        )
    best = min(WIDTHS, key=lambda w: abs(ink[w] - emnist_ink))
    print(
        f"mean ink per pixel: EMNIST {emnist_ink:.3f}; ours "
        + ", ".join(f"{w}: {v:.3f}" for w, v in ink.items())
        + f" -> raster_width {best}"
    )

    # 3. grid: per class 4 EMNIST, 2 augmented, then our samples at the best width
    g = torch.Generator(device=dev).manual_seed(0)
    aug = frame_batch(augment(up, g), THR, FILL)[:, 0].cpu().numpy()
    by_class = {c: [] for c in CLASSES}
    for s in samples:
        by_class[s.label].append(strokes.strokes_to_input(s.points, best, THR, FILL))
    cell, gap = 28 * 2, 3
    ncol = 4 + 2 + 3
    sheet = Image.new("L", (ncol * (cell + gap) + 40 + 6 * gap, 36 * (cell + gap)), 40)
    draw = ImageDraw.Draw(sheet)
    for ci, c in enumerate(CLASSES):
        rows = list(em[ci * 50 : ci * 50 + 4]) + list(aug[ci * 50 : ci * 50 + 2]) + by_class[c][:3]
        y = ci * (cell + gap)
        draw.text((10, y + cell // 3), c, fill=255)
        for k, im in enumerate(rows):
            x = 40 + k * (cell + gap) + (gap * 3 if k >= 4 else 0) + (gap * 3 if k >= 6 else 0)
            tile = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).resize((cell, cell))
            sheet.paste(tile, (x, y))
    sheet.save(a.out)
    print(f"wrote {a.out}")
    summary = {"parity": parity, "emnist_ink": emnist_ink, "ours_ink": ink, "raster_width": best}
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
