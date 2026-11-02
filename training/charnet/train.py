"""Train CharNet on EMNIST digits + uppercase (Air Writing spec §5.4).

    python train.py --data /content/data --eval eval_v1.jsonl.gz --out /content/charnet/run \
        --raster-width 0.1

Everything runs on the GPU: the whole dataset sits there as uint8, each batch is drawn
class-balanced (every class equally likely; augmentation makes the repeats differ), augmented,
framed by `frame_batch` (the app's framing), and trained with AdamW, a one-cycle schedule, label
smoothing and mixed precision. The epoch with the best accuracy on a held-out 5 % of EMNIST
train is kept, and a softmax temperature is fitted on that split.

Our air-written evaluation set is scored every epoch for information only. It never chooses
anything: not the epoch, not the temperature.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from common import CLASSES, CharNet, augment, emnist_input, frame_batch, strokes, upsample, wilson

THR, FILL = 0.1, 0.857


def log(*a) -> None:
    print(*a, flush=True)


def framed(x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([emnist_input(x[i : i + chunk], THR, FILL) for i in range(0, len(x), chunk)])


@torch.no_grad()
def logits_of(model, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + chunk]).float() for i in range(0, len(x), chunk)])


def topk(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    top3 = logits.topk(3, dim=1).indices
    return (top3[:, 0] == y).float().mean().item(), (top3 == y[:, None]).any(
        1
    ).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/content/data"))
    ap.add_argument("--eval", type=Path, default=Path("/content/charnet/eval_v1.jsonl.gz"))
    ap.add_argument("--out", type=Path, default=Path("/content/charnet/run"))
    ap.add_argument("--raster-width", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=1500, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU: smoke tests only
    amp = dev == "cuda"
    t_start = time.time()

    d = np.load(a.data / "emnist36.npz")
    x_all, y_all = torch.tensor(d["x_train"], device=dev), torch.tensor(d["y_train"], device=dev)
    # Hold out 5 % of each class for validation.
    g = torch.Generator(device=dev).manual_seed(a.seed)
    val_mask = torch.zeros(len(y_all), dtype=torch.bool, device=dev)
    for c in range(36):
        idx = torch.nonzero(y_all == c)[:, 0]
        n = max(1, int(0.05 * len(idx)))
        val_mask[idx[torch.randperm(len(idx), device=dev, generator=g)[:n]]] = True
    x_tr, y_tr = x_all[~val_mask], y_all[~val_mask]
    x_val, y_val = framed(x_all[val_mask]), y_all[val_mask]
    order = torch.argsort(y_tr)
    x_tr, y_tr = x_tr[order], y_tr[order]
    counts = torch.bincount(y_tr, minlength=36)
    starts = torch.cumsum(counts, 0) - counts
    lo, hi = counts.min().item(), counts.max().item()
    log(f"train {len(y_tr)}, val {len(y_val)}; per class {lo}-{hi}")

    _, samples = strokes.read_samples(a.eval)
    air_x = torch.tensor(
        np.stack([strokes.strokes_to_input(s.points, a.raster_width, THR, FILL) for s in samples]),
        device=dev,
    )[:, None]
    air_y = torch.tensor([CLASSES.index(s.label) for s in samples], device=dev)

    model = CharNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=amp)
    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for _ in range(a.steps):
            cls = torch.randint(0, 36, (a.batch,), device=dev, generator=g)
            off = (torch.rand(a.batch, device=dev, generator=g) * counts[cls]).long()
            idx = starts[cls] + off
            with torch.no_grad():
                xb = frame_batch(augment(upsample(x_tr[idx]), g), THR, FILL)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), y_tr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
        val1, _ = topk(logits_of(model, x_val), y_val)
        air1, air3 = topk(logits_of(model, air_x), air_y)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / a.steps,
                "val_top1": val1,
                "air_top1": air1,
                "air_top3": air3,
                "s": time.time() - t0,
            }
        )
        mark = ""
        if val1 > best:
            best, mark = val1, " *"
            torch.save(model.state_dict(), a.out / "best.pt")
        log(
            f"epoch {epoch + 1:2d}  loss {loss_sum / a.steps:.3f}  val {val1:.4f}  "
            f"air {air1:.3f}/{air3:.3f}  {time.time() - t0:.0f} s{mark}"
        )

    model.load_state_dict(torch.load(a.out / "best.pt"))
    # Temperature on the validation split, so the confidence bars mean something.
    lv = logits_of(model, x_val)
    temps = torch.linspace(0.5, 3.0, 51, device=dev)
    nll = torch.stack([F.cross_entropy(lv / t, y_val) for t in temps])
    temp = float(temps[nll.argmin()])

    x_te, y_te = (
        framed(torch.tensor(d["x_test"], device=dev)),
        torch.tensor(d["y_test"], device=dev),
    )
    lt = logits_of(model, x_te)
    te1, te3 = topk(lt, y_te)
    pred = lt.argmax(1)
    per_class = {c: (pred[y_te == i] == i).float().mean().item() for i, c in enumerate(CLASSES)}
    conf = torch.zeros(36, 36, dtype=torch.long, device=dev)
    conf.index_put_((y_te, pred), torch.ones_like(y_te), accumulate=True)
    off_diag = [
        (int(conf[i, j]), CLASSES[i], CLASSES[j]) for i in range(36) for j in range(36) if i != j
    ]
    top_conf = [{"true": t, "pred": p, "n": n} for n, t, p in sorted(off_diag, reverse=True)[:15]]

    la = logits_of(model, air_x)
    air1, air3 = topk(la, air_y)
    n = len(air_y)
    k1 = round(air1 * n)
    k3 = round(air3 * n)
    probs = F.softmax(la / temp, dim=1)
    top3 = probs.topk(3, dim=1)
    mistakes = [
        {
            "label": s.label,
            "style": s.style,
            "top3": [
                [CLASSES[j], round(float(p), 3)]
                for j, p in zip(top3.indices[i], top3.values[i], strict=True)
            ],
        }
        for i, s in enumerate(samples)
        if top3.indices[i, 0] != air_y[i]
    ]
    by_style = {}
    for st in sorted({s.style for s in samples}):
        m = torch.tensor([s.style == st for s in samples], device=dev)
        by_style[st] = (la[m].argmax(1) == air_y[m]).float().mean().item()
    metrics = {
        "val_top1": best,
        "temperature": temp,
        "emnist_test_top1": te1,
        "emnist_test_top3": te3,
        "emnist_test_n": len(y_te),
        "emnist_per_class": per_class,
        "emnist_top_confusions": top_conf,
        "air_top1": air1,
        "air_top3": air3,
        "air_n": n,
        "air_top1_ci95": wilson(k1, n),
        "air_top3_ci95": wilson(k3, n),
        "air_top1_by_style": by_style,
        "air_mistakes": mistakes,
        "history": history,
        "args": {k: str(v) for k, v in vars(a).items()},
        "train_minutes": (time.time() - t_start) / 60,
    }
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"temperature {temp:.2f}; EMNIST test top-1 {te1:.4f} top-3 {te3:.4f}")
    log(
        f"air-written top-1 {air1:.3f} (95% CI {wilson(k1, n)[0]:.2f}-{wilson(k1, n)[1]:.2f}), "
        f"top-3 {air3:.3f}; by style {by_style}"
    )
    log(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
