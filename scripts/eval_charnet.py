"""CharNet accuracy on air-written samples (Air Writing spec §8).

    uv run python scripts/eval_charnet.py [SAMPLES ...] [--model assets/charnet/charnet_v1.npz]

Default samples: tests/fixtures/strokes/eval_v1.jsonl.gz. Frames each sample with the model's
own framing parameters, and reports top-1 and top-3 accuracy with 95 % confidence intervals
(Wilson), accuracy per style, per-class accuracy, and the confusions.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ai_playground.ml.charnet import CharNet
from ai_playground.ml.strokes import read_samples, strokes_to_input

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(net: CharNet, paths: list[Path]) -> dict:
    samples = [s for p in paths for s in read_samples(p)[1]]
    x = np.stack(
        [strokes_to_input(s.points, net.raster_width, net.ink_threshold, net.fill) for s in samples]
    )
    p = net.probs(x)
    top3 = np.argsort(p, axis=1)[:, ::-1][:, :3]
    y = np.array([net.classes.index(s.label) for s in samples])
    hit1, hit3 = top3[:, 0] == y, (top3 == y[:, None]).any(axis=1)
    return {"samples": samples, "top3": top3, "y": y, "hit1": hit1, "hit3": hit3, "probs": p}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "samples", type=Path, nargs="*", default=[ROOT / "tests/fixtures/strokes/eval_v1.jsonl.gz"]
    )
    ap.add_argument("--model", type=Path, default=ROOT / "assets/charnet/charnet_v1.npz")
    a = ap.parse_args(argv)
    net = CharNet(a.model)
    r = evaluate(net, a.samples)
    n = len(r["y"])
    for name, hits in (("top-1", r["hit1"]), ("top-3", r["hit3"])):
        lo, hi = wilson(int(hits.sum()), n)
        print(f"{name}: {hits.mean():.1%} ({hits.sum()}/{n}), 95% CI {lo:.1%}-{hi:.1%}")
    styles = sorted({s.style for s in r["samples"]})
    for st in styles:
        m = np.array([s.style == st for s in r["samples"]])
        print(f"  style {st or '-'}: top-1 {r['hit1'][m].mean():.1%} ({m.sum()})")
    per = {c: r["hit1"][r["y"] == i] for i, c in enumerate(net.classes)}
    weak = [f"{c} {h.mean():.0%}" for c, h in per.items() if len(h) and h.mean() < 1]
    print("classes below 100%: " + (", ".join(weak) or "none"))
    conf = Counter(
        (net.classes[t], net.classes[p])
        for t, p in zip(r["y"], r["top3"][:, 0], strict=True)
        if t != p
    )
    print(
        "confusions (true -> read): "
        + (", ".join(f"{t}->{p} x{k}" for (t, p), k in conf.most_common()) or "none")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
