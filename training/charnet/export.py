"""Export CharNet for the app (Air Writing spec §5.5).

    python export.py --run /content/charnet/run --data /content/data --eval eval_v1.jsonl.gz \
        --raster-width 0.1 --commit <git sha> --out /content/charnet/export

Folds each BatchNorm into the convolution before it and writes:
- `charnet_v1.npz`: weights (float32), classes, temperature, the framing parameters, version;
- `charnet_v1.json`: the model card (data, training settings, accuracies, top confusions);
- `charnet_parity.npz`: 64 inputs and PyTorch's logits, for the app's parity test.
Then checks the app's NumPy inference (`charnet.py`, uploaded next to this file) against
PyTorch on those inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from common import CLASSES, CharNet, emnist_input, strokes

THR, FILL = 0.1, 0.857
VERSION = "charnet_v1"


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    k = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    w = conv.weight.detach() * k[:, None, None, None]
    b = bn.bias.detach() - bn.running_mean * k
    return w.cpu().numpy().astype(np.float32), b.cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("/content/charnet/run"))
    ap.add_argument("--data", type=Path, default=Path("/content/data"))
    ap.add_argument("--eval", type=Path, default=Path("/content/charnet/eval_v1.jsonl.gz"))
    ap.add_argument("--out", type=Path, default=Path("/content/charnet/export"))
    ap.add_argument("--raster-width", type=float, required=True)
    ap.add_argument("--commit", default="unknown")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    model = CharNet()
    model.load_state_dict(torch.load(a.run / "best.pt", map_location="cpu"))
    model.eval()
    metrics = json.loads((a.run / "metrics.json").read_text())

    arrays: dict[str, np.ndarray] = {}
    convs = []
    for block in model.features:
        convs += [(block[0], block[1]), (block[3], block[4])]
    for i, (conv, bn) in enumerate(convs):
        arrays[f"conv{i}_w"], arrays[f"conv{i}_b"] = fold(conv, bn)
    for name, lin in (("fc1", model.head[1]), ("fc2", model.head[4])):
        arrays[f"{name}_w"] = lin.weight.detach().numpy().astype(np.float32)
        arrays[f"{name}_b"] = lin.bias.detach().numpy().astype(np.float32)
    arrays.update(
        classes=np.array(CLASSES),
        temperature=np.array(metrics["temperature"], np.float32),
        raster_width=np.array(a.raster_width, np.float32),
        ink_threshold=np.array(THR, np.float32),
        fill=np.array(FILL, np.float32),
        version=np.array(VERSION),
    )
    np.savez_compressed(a.out / f"{VERSION}.npz", **arrays)

    # Parity inputs: half our air-written samples, half EMNIST test glyphs.
    _, samples = strokes.read_samples(a.eval)
    ours = np.stack(
        [strokes.strokes_to_input(s.points, a.raster_width, THR, FILL) for s in samples[::3][:32]]
    )
    d = np.load(a.data / "emnist36.npz")
    rng = np.random.default_rng(1)
    pick = rng.choice(len(d["y_test"]), 32, replace=False)
    em = emnist_input(torch.tensor(d["x_test"][pick]), THR, FILL)[:, 0].numpy()
    inputs = np.concatenate([ours, em]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(inputs)[:, None]).numpy()
    np.savez_compressed(a.out / "charnet_parity.npz", inputs=inputs, logits=logits)

    import charnet  # the app's NumPy inference, uploaded next to this script

    net = charnet.CharNet(a.out / f"{VERSION}.npz")
    diff = float(np.abs(net.logits(inputs) - logits).max())
    print(f"NumPy vs PyTorch logits: max |diff| = {diff:.2e}")
    assert diff < 1e-4, diff

    data_meta = json.loads((a.data / "emnist36.json").read_text())
    card = {
        "model": VERSION,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": a.commit,
        "classes": CLASSES,
        "architecture": "VGG-style CNN (Air Writing spec §5.4), BatchNorm folded, "
        f"{sum(p.numel() for p in model.parameters())} parameters",
        "data": {
            **data_meta,
            "note": "EMNIST ByClass (Cohen et al., 2017, from NIST SD 19), "
            "lowercase dropped; class-balanced sampling; 5 % of train held out",
        },
        "framing": {"raster_width": a.raster_width, "ink_threshold": THR, "fill": FILL},
        "training": metrics["args"],
        "train_minutes": metrics["train_minutes"],
        "temperature": metrics["temperature"],
        "emnist": {
            "val_top1": metrics["val_top1"],
            "test_top1": metrics["emnist_test_top1"],
            "test_top3": metrics["emnist_test_top3"],
            "test_n": metrics["emnist_test_n"],
            "per_class_test_top1": metrics["emnist_per_class"],
            "top_confusions": metrics["emnist_top_confusions"],
        },
        "air_written": {
            "set": "tests/fixtures/strokes/eval_v1.jsonl.gz (one writer, 3 styles)",
            "n": metrics["air_n"],
            "top1": metrics["air_top1"],
            "top1_ci95": metrics["air_top1_ci95"],
            "top3": metrics["air_top3"],
            "top3_ci95": metrics["air_top3_ci95"],
            "top1_by_style": metrics["air_top1_by_style"],
            "mistakes": metrics["air_mistakes"],
        },
        "parity_max_abs_diff": diff,
        "history": metrics["history"],
    }
    (a.out / f"{VERSION}.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
