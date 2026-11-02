# CharNet training (Air Writing spec §5)

Trains the character network on a Colab GPU and exports it for the app's NumPy inference.
Driven from this machine with the Colab CLI (the `colab-operator` skill); nothing here enters
`uv.lock`.

| File | What |
|------|------|
| `common.py` | Model, `frame_batch` (batched port of the app's `frame_image`), augmentation, Wilson CI |
| `prepare_data.py` | EMNIST ByClass → 36 classes (0–9, A–Z), orientation fixed, pinned source |
| `compare_framing.py` | Framing parity on real EMNIST, line-width match (`raster_width`), comparison grid |
| `train.py` | Class-balanced, augmented training on the GPU; temperature; metrics |
| `export.py` | BatchNorm folding → `charnet_v1.npz`, model card, parity fixture; NumPy parity check |

`strokes.py` and `charnet.py` are the app's own `src/ai_playground/ml/` modules, uploaded next
to these scripts, so framing and inference can't drift between training and the app.

## Run

NIST's server refuses scripted downloads (HTTP 403), so download EMNIST's `gzip.zip` ("Binary
format as the original MNIST dataset", 562 MB) from
https://www.nist.gov/itl/products-and-services/emnist-dataset in a browser. `prepare_data.py`
checks its MD5 and needs only NumPy, so run it locally and upload the 159 MB result.

```bash
uv run python training/charnet/prepare_data.py --zip ~/Downloads/gzip.zip --out /tmp/emnist

C="colab --auth=adc"   # this CLI version defaults to its own oauth2 flow
S=charnet
$C new -s $S --gpu T4
$C exec -s $S <<< "import os; [os.makedirs(d, exist_ok=True) for d in ('/content/charnet', '/content/data')]"
for f in training/charnet/*.py src/ai_playground/ml/strokes.py src/ai_playground/ml/charnet.py \
         tests/fixtures/strokes/eval_v1.jsonl.gz; do
  $C upload -s $S "$f" "/content/charnet/$(basename "$f")"
done
# Uploads over ~100 MB are refused (HTTP 400): send the dataset in parts and join them.
(cd /tmp/emnist && split -b 40M emnist36.npz part_ && for p in part_*; do
   $C upload -s $S $p /content/data/$p; done)
$C upload -s $S /tmp/emnist/emnist36.json /content/data/emnist36.json
$C exec -s $S <<< "import glob, os; ps = sorted(glob.glob('/content/data/part_*')); open('/content/data/emnist36.npz', 'wb').write(b''.join(open(p, 'rb').read() for p in ps)); [os.remove(p) for p in ps]"

run() { $C exec -s $S <<< "import os, sys, runpy; os.chdir('/content/charnet'); sys.path.insert(0, '.'); [sys.modules.pop(m, None) for m in ('common', 'strokes', 'charnet')]; sys.argv = $1; runpy.run_path(sys.argv[0], run_name='__main__')"; }
run "['compare_framing.py']"            # framing.json: the raster_width to use
run "['train.py', '--raster-width', '0.12']"
run "['export.py', '--raster-width', '0.12', '--commit', '$(git rev-parse --short HEAD)']"
for f in charnet_v1.npz charnet_v1.json charnet_parity.npz; do
  $C download -s $S "/content/charnet/export/$f" "$f"
done
$C stop -s $S
```

Then move `charnet_v1.npz` and `charnet_v1.json` to `assets/charnet/`, `charnet_parity.npz` to
`tests/fixtures/`, set `[writing] raster_width` to the value the model was trained with, and
run `uv run python scripts/eval_charnet.py`.

## Rules

- **The air-written evaluation set never chooses anything.** It is scored each epoch for
  information; the epoch is chosen on held-out EMNIST, the temperature too. `raster_width` is
  matched on mean ink, a statistic of the images, not on accuracy.
- **It is never augmented and never trained on** (spec §8). Fine-tuning, if needed, uses a
  separate collection.
