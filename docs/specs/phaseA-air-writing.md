# Phase A — Air Writing

**Outcome:** a visitor pinches, writes a digit or a capital letter in the air, makes a fist, and a
neural network reads it. The screen shows the 28×28 image the network actually sees and its top
three guesses. This is Phase A's machine-learning mode for the booth (PLAN D11).

Status: **detailed spec**, written 2026-11-01. Decisions: PLAN D11 (booth format), D12 (CNN trained
offline, NumPy inference), D8 (strokes only, never video).
Builds on: [Phase 0](phase0-foundation.md), [Phase A Geometry](phaseA-magic-geometry.md) (gesture
vocabulary, paddle arming, two-fist clear, booth shell), [architecture](architecture.md).

---

## 1. The visitor's 15 seconds

1. A hint reads: *"Pinch your fingers and write a number or a capital letter in the air."*
2. The visitor pinches and draws a **7**. Glowing ink follows the pinch point.
3. *"Make a fist to let the AI read it."* A ring fills around the fist for 0.3 s.
4. The drawing shrinks into a coarse 28×28 pixel grid (*"This is what the AI sees: 784
   numbers"*), and a big **7** appears with confidence bars: `7 94 %`, `1 3 %`, `2 1 %`.
5. The character is added to a line of text at the top. They write the next one, or walk on.

Design principles from Geometry §1 apply: one hand is enough, few reliable gestures, instant
feedback, hints that disappear once an action has been done.

## 2. Interaction vocabulary

| Gesture | Action | Feedback |
|---------|--------|----------|
| **Pinch & hold** (thumb + index) | **Draw:** ink follows the pinch point | Glowing ink in the hand's track color; pinch-dot cursor |
| **Release pinch** | **Pen up.** The character is *not* finished, so multi-stroke characters (A, E, F, H, K, T, 4, 5…) work | Ink stays |
| **Open palm, armed** | **Eraser:** removes the ink under a circle at the palm (radius ∝ hand size) | Eraser ring, fading in while it arms |
| **One fist, held `read_hold_s` (0.3 s)** | **Read:** recognize the current ink (§6) | Progress ring around the fist |
| **Both fists held `clear_hold_s` (1.5 s)** | **Start over:** clear ink, result, and text line | Progress ring around both fists, as in Geometry |
| *(no hands for `idle_timeout`)* | Same as start over (booth shell, Geometry §6.0) | – |

Any pinching hand draws, so two hands may write at once. Each has its own active stroke.

**Rules carried over from Geometry, and why** (Geometry §2, "Changed in A.3"):

- **The eraser uses the paddle's arming rule:** MediaPipe's debounced `Open_Palm` held for
  `paddle_hold_s` (0.4 s), and not within `paddle_after_pinch_s` (0.4 s) of the same hand ending a
  pinch. Without it, the half-open hand between the strokes of an "A" would erase the first
  stroke. A visitor who wants to erase opens the hand deliberately; the ring shows when it is live.
- **Read requires a held fist, and not two fists.** The read ring only fills while exactly one
  hand is a fist. If the second hand also closes, the clear ring takes over. A read that happened
  just before a clear is harmless, because the clear wipes it.
- **Pinch is already blocked during a fist** (`block_on_fist`, `min_index_extension` in
  `[gestures.pinch]`), so making a fist doesn't start a new stroke.

**New rule: drop the release drag at every pen-up** *(revised in W.1, measured on
`air_write_digits`)*. Letting go of a pinch drags the pinch point: as thumb and index open, the
pinch ratio climbs from ~0.2 past 0.5 over 2–4 frames (60–130 ms) before `pinch_end` fires, and
the ink point slides 0.03–0.08 window heights. That leaves a hook at the end of **every** stroke,
not only before a fist (the first spec's assumption): the "1" in the fixture had one with the fist
0.67 s later. So a pen-up drops the ink drawn after the pinch ratio last rose above
`release_ratio` (0.35, the pinch's own enter threshold), at most `release_trim_max_s` (0.25 s) so
a loosely held pinch keeps its ink. Curling into a fist opens the pinch too, so this also removes
the hook of a fist made right after writing (the 7 and the A in the fixture). Thresholds of 0.27
and 0.30 made no visible difference; the remaining 1–3 % ticks are real hand movement while the
pinch is still firmly closed.

A fist that closes **while the pinch is still held** ends the stroke and also trims its last
`tail_trim_s` (0.15 s), since the fist label lags the curling fingers by 70–130 ms (Geometry §3).
The fixture has no such case (its writer always let go first), so 0.15 s is untested on real
hands. The first spec's `tail_window_s` (trim after a fist within 0.4 s of a pinch end) is gone:
on the fixture it only removed good ink once the release drag was handled.

**Fist with no ink:** no read; the hint line says *"Pinch to write first."*

## 3. Strokes (`ml/strokes.py`, portable core)

- **Ink point:** the pinch point (midpoint of thumb tip and index tip), already smoothed by the
  One Euro filter, mapped to screen pixels with the viewport. The display is mirrored, so writing
  is not reversed.
- **Resampling:** points are appended at a fixed spacing of `resample_step` (0.004 window heights),
  interpolating between camera frames. Ink density then doesn't depend on drawing speed, and the
  eraser can work point by point.
- **Data:** a stroke is an `(N, 2) float32` array of screen points plus a timestamp per point; a
  character is a list of strokes.
- **Erase:** delete the points inside the eraser circle and split strokes where points were
  removed. Fragments shorter than 3 points are dropped.
- **Drawing:** round-capped polylines of width `ink_width` (0.012 window heights) with the same
  additive glow as Geometry.
- **Writing area:** a large, faint rounded box in the middle of the screen. It is only a guide to
  keep characters a sensible size; ink outside it still counts.
- **Too small:** if the ink's bounding box is under `min_char_size` (0.08 window heights), don't
  read; hint *"Write it bigger."*

## 4. From strokes to the network's input (`ml/strokes.py`)

The model is trained on EMNIST images, but it reads our strokes. Both must arrive at the network
**framed the same way**, so one function does the framing for both, in training and in the app:

```
frame_image(img_hi: float32 (H, W), ink in [0, 1]) -> float32 (28, 28)
```

1. **Crop** to the bounding box of pixels above `ink_threshold`.
2. **Fit** the box into a square, keeping the aspect ratio (a "1" stays thin), centered by the
   bounding box.
3. **Pad** so the character fills `fill` of the frame (EMNIST: 24 of 28 pixels, i.e. a 2-pixel
   border).
4. **Downsample** to 28×28 by area averaging, then scale so the brightest pixel is 1.

In the app, strokes are first rendered at high resolution (`render_strokes(strokes) -> 112×112`):
NumPy computes each pixel's distance to the nearest segment, and ink intensity falls off from 1
at `raster_width / 2` over 1 pixel (anti-aliasing). `raster_width` is relative to the character's
larger side and is **tuned so our stroke thickness matches EMNIST's** (§5.3 check). In training,
EMNIST images are upsampled to 112×112, augmented (§5.3), and passed to the same `frame_image`.

All of this is NumPy only: no cv2, no pygame, no SciPy.

## 5. Model

### 5.1 Classes

36 classes: digits `0–9` and uppercase `A–Z`. No lowercase: half of the lowercase letters look
like their uppercase forms in EMNIST (c/C, o/O, s/S…), and adding them would cost accuracy for
little gain at a booth.

### 5.2 Data (`training/charnet/prepare_data.py`)

- **EMNIST ByClass** (Cohen et al., 2017, derived from NIST Special Database 19), filtered to the
  36 classes, dropping the 26 lowercase classes. ByClass is the only split that keeps uppercase
  apart from lowercase: Balanced and ByMerge merge confusable upper/lowercase pairs, and Letters
  mixes both cases in each class.
- **Imbalance:** digits have tens of thousands of examples each, some letters only a few thousand.
  Train with a class-balanced sampler (each epoch draws equally per class, with augmentation
  making the repeats differ). Report per-class accuracy, not only the mean.
- **Orientation:** EMNIST's raw files store images transposed. Fix it and **look at a grid of
  samples per class** before training.
- **Source and integrity:** NIST's download has been unreliable at times. Pin whichever source is
  used (URL + SHA-256) in the script and record it in the model card. *W.3:* NIST's server
  refuses scripted downloads (HTTP 403), so `gzip.zip` is downloaded in a browser and passed to
  `prepare_data.py --zip`, which checks the MD5 torchvision publishes
  (`58c8d27c78d21e728a6bc7b3cc06412e`; SHA-256 `fb9bb67e…cda8e`). The 36 classes: 533,993 train
  images (fewest: K, 2,468), 89,264 test.
- **Splits:** EMNIST's own test set measures clean accuracy. The real measure is our own
  air-written set (§8), which is never used for training except through a separate fine-tuning
  split.

### 5.3 Augmentation: making pen scans look like air-writing

Air-written characters are wobbly, evenly thick, slanted, and sometimes not quite closed. Applied
on the GPU per batch, before `frame_image`:

| Augmentation | Range | Imitates |
|--------------|-------|----------|
| Rotation | ±12° | A tilted hand |
| Shear | ±0.3 | Slanted writing |
| Scale, anisotropic | 0.8–1.15 per axis | Squashed or stretched letters |
| Elastic distortion | small α, σ | Hand tremor, wobbly lines |
| **Re-stroke** (p = 0.5) | thickness 0.06–0.14 of the box | Skeletonize the glyph, then redraw it at an even random thickness: our ink has constant width, a pen scan doesn't. *Not in v1:* per-image skeletonizing is CPU work that would slow training a lot. v1 thickens or thins the line instead (max-pool dilation/erosion, 3 or 5 px at 112 px, p = 0.6). Add it in W.5 if line width turns out to matter |
| *Crisp edges* (p = 0.5) *(added in W.3)* | contrast ×2.5 around 0.3 | EMNIST glyphs are blurred scans; our ink has a 1-pixel anti-aliased edge (the framing grid shows the difference) |
| Gaps (p = 0.15) | 1–2 short gaps | Partial erasing; a pinch released early |

**Framing check** (`training/charnet/compare_framing.py`, run in Colab): a grid with our framed
air-written samples next to framed EMNIST samples of the same class, plus the mean ink fraction
per image for both. Thickness and placement should look the same before training starts. This is
also how `raster_width` (§4) is tuned. *W.3:* the batched training framing matches the app's
`frame_image` within 6e-8 on 1,800 EMNIST glyphs; mean ink is 0.177 for EMNIST and 0.158 / 0.183
for ours at `raster_width` 0.10 / 0.12, so **`raster_width` = 0.12**. Placement and size match
in the grid; EMNIST's edges are softer (hence the crisp-edge augmentation).

### 5.4 Architecture and training (`training/charnet/train.py`)

A compact VGG-style CNN, ~0.6 M parameters:

```
input 1×28×28
[conv 3×3, 32 → BN → ReLU] ×2 → max-pool 2      14×14
[conv 3×3, 64 → BN → ReLU] ×2 → max-pool 2       7×7
[conv 3×3, 128 → BN → ReLU] ×2 → max-pool 2      3×3
flatten 1152 → dense 256 → ReLU → dropout 0.4 → dense 36
```

- PyTorch on a Colab GPU, driven with the Colab CLI (the `colab-operator` skill). Training
  dependencies live in `training/charnet/requirements.txt` and **never enter `uv.lock`**.
- AdamW, one-cycle learning-rate schedule, label smoothing 0.1, batch 256, mixed precision,
  ~20 epochs; keep the epoch with the best validation accuracy (a held-out 5 % of EMNIST train).
- **Calibration:** fit a softmax temperature on the validation set, so the confidence bars mean
  something. Stored in the weights file.
- The architecture is a starting point. Changes are judged on the **air-written** set, not on
  EMNIST test accuracy.

### 5.4.1 Results: charnet_v1 *(W.3, 2026-11-01)*

Colab T4, 20 epochs × 1,500 batches of 256, 18 minutes. The epoch was chosen on held-out EMNIST
(epoch 19, 93.1 %); temperature 0.65 (label smoothing leaves the raw softmax underconfident).
Full numbers in `assets/charnet/charnet_v1.json`.

| | top-1 | top-3 |
|---|---|---|
| EMNIST test, 36 classes (89,264) | 93.3 % | 99.9 % |
| Air-written `eval_v1` (108, one writer) | **90.7 %** (98/108; 95 % CI 84–95 %) | **100 %** |

- **Air-written misses:** I→1 ×3 and O→0 ×3 (this writer's I and O have no serifs or shape that
  separates them from 1 and 0), 6→G ×2, 1→7 (a 1 with a flag), 1→9 (fast style). Leaving out
  the look-alike pairs: 98/102. By style: natural 89 %, fast 92 %, other hand 92 %.
- **EMNIST's own look-alikes** dominate its errors too: O→0 1,539, 0→O 1,303, 1↔I ~740 each,
  5→S 326, 2→Z 284. Its weakest classes are O (62 %), I (63 %), 0 (75 %), 1 (88 %).
- **For W.4:** the second guess in the six I/O misses had p = 0.12–0.25, so `confusable_p` = 0.25
  (§5.6) would show "1 or I?" for only one of them. Lower it (≈ 0.1) for the listed pairs.
- Air-written accuracy moved between 87 % and 92 % from epoch to epoch (±1–2 characters):
  differences that small between models can't be told apart on this set.
- NumPy inference matches PyTorch within 2.4e-6 (fixture: 1e-4 tolerance) and takes 1.7 ms per
  read on the Ubuntu desktop.

### 5.5 Export and NumPy inference (`training/charnet/export.py`, `ml/charnet.py`)

- **Export:** fold each BatchNorm into the preceding conv or dense layer, then save
  `assets/charnet/charnet_v1.npz`: weights (float32), class list, temperature, `frame_image`
  parameters, and the source data hash. Next to it, `charnet_v1.json` is the **model card**: data,
  training settings, git commit, EMNIST test accuracy, air-written accuracy, top confusions.
  At ~2.4 MB the weights are **committed to the repo** (`assets/models/` stays for fetched
  third-party models).
- **Inference** (`ml/charnet.py`, portable core): conv via `sliding_window_view` + matmul, ReLU,
  max-pool via reshape, dense, softmax with temperature. Budget: ≤ 5 ms per read. It runs once
  per fist, not every frame.
- **Parity:** export also writes `tests/fixtures/charnet_parity.npz` (64 inputs and PyTorch's
  logits). A test checks NumPy inference matches within 1e-4. This catches layout mistakes
  (NCHW vs NHWC, transposed kernels) that would otherwise only show up as "bad accuracy".

### 5.6 Look-alike characters

Some pairs are ambiguous even to people: **O/0, I/1, Z/2, S/5, B/8, G/6, D/0, Q/O, T/7**.
- If the top two guesses are a pair from `confusable_pairs` and the second has probability
  ≥ `confusable_p` (0.25), show both: **"0 or O?"** with the subtitle *"These look alike to the
  AI — and to people!"* This is a teaching moment, not a failure.
- The text line takes the top guess.

## 6. The screen (`scenes/writing.py`)

- **Center:** the writing area with the ink. Camera background as in Geometry (dim, `B` toggles).
- **Right panel, after a read:**
  - the big recognized character (or the "0 or O?" pair);
  - top-3 confidence bars with percentages;
  - **"What the AI sees":** the 28×28 input drawn as a pixelated 8× grid, labelled *"784
    numbers"*. On a read, the ink animates shrinking into this grid (0.4 s): the moment that
    explains what the model gets.
- **Top:** the text line of read characters (last `text_max` = 12, oldest scroll off).
- **After a read,** the ink fades out over 0.5 s, so the next character starts on a clean canvas.
  The result panel stays until the next read.
- **Bottom:** the hint line (booth shell).
- The Vision overlay (`V`) works on top, as in every scene.

## 7. Configuration `[writing]`

`ink_width`, `resample_step`, `min_char_size`, `read_hold_s`, `release_ratio`,
`release_trim_max_s`, `tail_trim_s`, `eraser_scale`, `area_size`, `background_brightness`
*(W.1)*, `raster_width`, `ink_threshold`, `fill`, `writer` *(W.2)*, then `ink_threshold`, `fill`, `confusable_pairs`, `confusable_p`,
`text_max`, `model` (path to the `.npz`). The eraser reuses `paddle_hold_s` and
`paddle_after_pinch_s`; the clear reuses `clear_hold_s`.

## 8. Air-written data: evaluation and fine-tuning

- **Collect mode:** operator key `T` in the scene. A panel on the left shows a target character
  and a writing style; the writer writes it and makes a fist; the strokes are saved with the
  label to `recordings/strokes/*.jsonl` (gitignored), the whole file after every read, so
  quitting midway keeps what was written. `Backspace` asks for the last character again (a
  stroke went wrong, or the wrong character was written). Points are stored in window heights,
  times from the sample's first point. **Strokes and timestamps only, never images or video**
  (D8). The result panel shows the 28×28 image the network gets, which makes collecting a quick
  framing check too. `scripts/air_samples.py merge` combines sessions into a set and `grid`
  draws every sample as the network sees it.
- **Evaluation set** *(changed in W.2: one writer)*:
  `tests/fixtures/strokes/eval_v1.jsonl.gz`, committed. There wasn't time to find other
  writers, so it has **3 samples per class from one writer, 108 characters**. To make one writer
  behave more like several, each sample of a class is written in a different style, one round
  of all 36 (shuffled) per style: **natural**; **fast and big, without lifting the fingers
  between strokes** (visitors join strokes, as the `air_write_digits` writer did); **with the
  other hand, or slanted**. Collect it in W.2, before the model exists, so it can't be tuned to
  the model.
- **What this set can and can't show.** The model gets its tolerance for different writers from
  EMNIST (thousands of writers, plus augmentation, §5.3). What EMNIST can't show is the gap
  between pen scans and air-writing (even width, wobble, joined strokes); that comes from the
  medium more than the writer, and this set measures it. But 108 samples from one writer are a
  small, narrow sample: a measured 90 % top-1 means roughly 84–96 % (95 % interval), and
  per-class accuracy (3 samples) is only good for spotting confusions. So the evaluation script
  reports a confidence interval with every accuracy, and the PLAN targets stay provisional.
  **The evaluation set is never augmented:** distorted copies of the same samples are
  correlated and would inflate the score. Augmentation is for training only (§5.3).
- **More writers, cheaply:** in the A.8 user tests, log for every read whether it was right
  (staff confirm with a key; strokes and labels only, D8). That gives a second, multi-writer
  accuracy figure at no extra cost.
- **`scripts/eval_charnet.py`:** top-1 and top-3 accuracy, per-class accuracy, and the confusion
  matrix on any stroke file.
- **Fine-tuning** (only if the air-written accuracy misses the target): collect a separate
  training set the same way (one writer: augment it heavily and mix in EMNIST, so the model
  doesn't learn one person's style), fine-tune in Colab at a low learning rate, and re-export.
  The evaluation set is never trained on.

## 9. Files

| Path | What |
|------|------|
| `src/ai_playground/scenes/writing.py` | Scene: gestures → ink, read, UI |
| `src/ai_playground/scenes/hand_rules.py` | `PalmArming` and `HoldTrigger`, shared with Geometry's paddle and clear |
| `src/ai_playground/render/ink.py` | Ink with glow, the writing area, the "You wrote" thumbnail (W.1 stand-in for the result panel) |
| `src/ai_playground/ml/strokes.py` | Resample, erase, `render_strokes`, `frame_image` (portable core) |
| `src/ai_playground/ml/charnet.py` | NumPy CNN inference (portable core) |
| `assets/charnet/charnet_v1.npz`, `.json` | Weights and model card (committed) |
| `training/charnet/` | `prepare_data.py`, `train.py`, `export.py`, `compare_framing.py`, `requirements.txt`, `README.md` (how to run it with the Colab CLI) |
| `scripts/eval_charnet.py` | Accuracy on stroke files, with confidence intervals |
| `scripts/air_samples.py` | Merge collect sessions into a set; draw the framed samples as a grid |
| `src/ai_playground/scenes/collect.py` | Collect mode: the prompt plan (36 characters × 3 styles) and saving |
| `tests/fixtures/strokes/eval_v1.jsonl.gz`, `tests/fixtures/charnet_parity.npz`, `tests/fixtures/air_write_*.jsonl.gz` | Evaluation set, parity check, landmark clips for the interaction |

**`air_write_digits` (W.1):** one right hand writes 1, 4, 7, A, E, each followed by a held fist;
one stroke is erased with an open palm before the E; two fists clear at the end. 54 s; a 15.8 s
held fist was cut out. Note that the writer drew the 4 and the A without lifting the pen; visitors
will too, so W.2's evaluation set should include such connected characters.

`training/` imports `ml/strokes.py` from the repo, so framing can't drift between training and
the app.

## 10. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| W.1 ✓ | Drawing: pinch ink, resampling, armed eraser, tail trim, read ring (no model yet), two-fist clear, idle reset. Landmark fixture `air_write_digits` | Characters look clean; no stray marks or hooks when making a fist (fixture test) |
| W.2 | `render_strokes` + `frame_image`; collect mode; record the evaluation set (one writer, 3 styles, 108 characters; §8) | Evaluation set committed; our framed images look plausible (the side-by-side with EMNIST is W.3's `compare_framing`, which needs the dataset) |
| W.3 ✓ | Colab: data prep, framing check, augmentation, training, export; NumPy inference + parity test | NumPy matches PyTorch within 1e-4; EMNIST test and air-written accuracy reported in the model card; PLAN accuracy targets confirmed or revised |
| W.4 | Reading UI: result, top-3 bars, "what the AI sees" grid and shrink animation, look-alike pairs, text line, hints | A new user gets a character read within 30 s |
| W.5 | Accuracy iteration: framing, augmentation, fine-tuning if needed; accuracy floor test | PLAN §3 Air Writing exit criteria met |
| W.6 | *Stretch:* live guess while writing (top-3 updating every ~200 ms); a small word list that corrects the text line | – |

## 11. Tests

- **Strokes:** resampling spacing is constant across drawing speeds; erasing splits a stroke in
  two; `frame_image` gives the same output within ε under translation and scaling of the ink;
  aspect ratio is kept (a vertical line stays thin and centered); tail trimming removes exactly
  the trailing window.
- **Interaction (fixtures):** replaying `air_write_digits` gives the expected number of strokes
  and reads, and no ink after a fist; the eraser doesn't fire between strokes of a multi-stroke
  character; two fists clear without a stray read surviving.
- **Model:** parity with PyTorch; probabilities sum to 1; inference ≤ 5 ms; the portable-core
  import rule covers `ml/strokes.py` and `ml/charnet.py`.
- **Accuracy floor:** top-1 and top-3 on `eval_v1` stay above the floors set in W.3. A framing or
  weights change that silently breaks recognition fails CI.
- **Scene smoke test** under `SDL_VIDEODRIVER=dummy` with a replayed fixture.

## 12. Open questions

- **Live guess while writing (W.6):** fun ("watch it change its mind") or distracting? Try it with
  users.
- **Restricting the class set** per round (digits only, letters only) would remove the O/0 and
  I/1 confusions. Keep all 36 unless user tests show the look-alike display isn't enough.
- **EMNIST terms:** check the NIST Special Database 19 terms before publishing weights trained
  on it in the public repo.
- **Two hands drawing:** by default both hands draw into the same character (§2). Does anyone
  actually write two-handed, or should only the first pinching hand draw?
