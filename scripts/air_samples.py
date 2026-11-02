"""Air Writing samples (collect mode, T): merge sessions into a set, and preview the framing.

    uv run python scripts/air_samples.py merge OUT IN [IN ...] [--note TEXT]
    uv run python scripts/air_samples.py grid IN [IN ...] -o grid.png

`merge` concatenates sample files, reports how many samples each class and style has, and
writes OUT (`.gz` compresses), e.g. tests/fixtures/strokes/eval_v1.jsonl.gz.

`grid` draws every sample as the network sees it (28x28, framed with the [writing] settings):
one row per class, in class order. Use it to check the framing looks like EMNIST.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ai_playground import config
from ai_playground.ml.strokes import (
    CLASSES,
    INPUT_SIZE,
    read_samples,
    strokes_to_input,
    write_samples,
)


def merge(out: Path, inputs: list[Path], note: str) -> int:
    samples, writers = [], set()
    for p in inputs:
        _, s = read_samples(p)
        samples += s
        writers |= {x.writer for x in s}
    bad = sorted({s.label for s in samples} - set(CLASSES))
    if bad:
        print(f"unknown labels: {bad}", file=sys.stderr)
        return 1
    per_class = Counter(s.label for s in samples)
    per_style = Counter(s.style for s in samples)
    missing = [c for c in CLASSES if per_class[c] == 0]
    print(f"{len(samples)} samples, writers {sorted(writers)}, styles {dict(per_style)}")
    print("per class: " + " ".join(f"{c}:{per_class[c]}" for c in CLASSES))
    if missing:
        print(f"missing classes: {''.join(missing)}", file=sys.stderr)
    header = {
        "kind": "air_writing_samples",
        "classes": CLASSES,
        "writers": sorted(writers),
        "sources": [p.name for p in inputs],
        "note": note,
    }
    samples.sort(key=lambda s: (CLASSES.index(s.label), s.style))
    write_samples(out, samples, header)
    print(f"wrote {out}")
    return 0


def grid(inputs: list[Path], out: Path, cell: int = 3) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    w = config.load().writing
    rows: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for p in inputs:
        for s in read_samples(p)[1]:
            rows[s.label].append(
                strokes_to_input(s.points, w.raster_width, w.ink_threshold, w.fill)
            )
    ncol = max(1, max(len(v) for v in rows.values()))
    side = INPUT_SIZE * cell
    gap = 2
    img = np.full((len(CLASSES) * (side + gap), ncol * (side + gap) + side), 40, np.uint8)
    pygame.init()
    for r, c in enumerate(CLASSES):
        y = r * (side + gap)
        for k, x28 in enumerate(rows[c]):
            big = np.kron(x28, np.ones((cell, cell)))
            x = side + k * (side + gap)
            img[y : y + side, x : x + side] = (big * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.repeat(img.T[:, :, None], 3, axis=2))
    font = pygame.font.SysFont(["dejavusans", "helvetica", "arial"], side // 2, bold=True)
    for r, c in enumerate(CLASSES):
        label = font.render(c, True, (255, 200, 90))
        surf.blit(label, label.get_rect(center=(side // 2, r * (side + gap) + side // 2)))
    pygame.image.save(surf, str(out))
    pygame.quit()
    print(f"wrote {out}: {sum(len(v) for v in rows.values())} samples")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("out", type=Path)
    m.add_argument("inputs", type=Path, nargs="+")
    m.add_argument("--note", default="")
    g = sub.add_parser("grid")
    g.add_argument("inputs", type=Path, nargs="+")
    g.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        return merge(a.out, a.inputs, a.note)
    return grid(a.inputs, a.out)


if __name__ == "__main__":
    sys.exit(main())
