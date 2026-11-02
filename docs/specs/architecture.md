# Architecture

Applies to all phases. Phase specs refine individual modules.

## 1. Runtime overview

```
 ┌──────────────── worker thread ────────────────┐      ┌────────────────────── main thread (60 FPS) ───────────────────────┐
 │                                               │      │                                                                   │
 │  Camera ──► flip ──► GestureRecognizer ──►────┼──►───┤ LatestSlot ──► HandTracker ──► GestureEngine ──► events ──► Scene │
 │ (OpenCV)   (mirror)  (MediaPipe, VIDEO mode)  │ Hand │ (newest     (stable IDs +    (features +      (GestureEvent)  │       │
 │                                               │Frame │  frame wins) One Euro filter) detectors)                      ▼       │
 │    or: ReplaySource (JSONL) ──────────────────┼──►───┤                                               render ◄── Scene.update │
 └───────────────────────────────────────────────┘      │                                                 │                 │
                                                        │ Recorder (optional, JSONL) ◄── HandFrame        ▼                 │
                                                        │                                         Overlay (Vision) + HUD    │
                                                        └───────────────────────────────┬───────────────────────────────────┘
                                                                                        │ AudioParams / NoteEvents (Phase B+)
                                                                                        ▼
                                                                      ┌──── audio thread (sounddevice callback) ────┐
                                                                      │ Synth voices ──► effects ──► output buffer  │
                                                                      └─────────────────────────────────────────────┘
```

**Threads**

| Thread | Owns | Rate |
|--------|------|------|
| Main | pygame window, input, gesture engine, scenes, physics, rendering | 60 FPS (vsync/clock) |
| Source worker | camera capture + MediaPipe inference (or replay timing) | camera rate, ~30 FPS |
| Audio (Phase B+) | sounddevice callback: synthesis only, no allocation-heavy work | every 256–512 samples |

**Handoffs**

- **Source → main:** a `LatestSlot`, i.e. one lock-protected value with a sequence number. The main
  loop takes the newest `HandFrame` and skips frames it missed. There is no queue, so latency
  can't build up.
- **Main → audio:** parameter targets are written as plain attributes (atomic under the GIL) and
  smoothed per-sample inside the callback. Discrete note events go through a bounded
  `collections.deque`, which the callback drains.

We run MediaPipe in **VIDEO mode**, called synchronously in the worker, rather than in
LIVE_STREAM mode. This keeps capture timestamp, image, and result together in one place, so
latency is easy to measure and the code is deterministic. The worker always reads the newest
camera frame. The camera is opened with `CAP_PROP_BUFFERSIZE=1` where the platform supports it.

## 2. Core data types (`ai_playground/types.py`)

```python
Handedness = Literal["Left", "Right"]

@dataclass(frozen=True, slots=True)
class Hand:
    handedness: Handedness      # from the visitor's point of view (frame is mirrored before inference)
    handedness_score: float
    landmarks: np.ndarray       # (21, 3) float32; x, y normalized to [0, 1] in the mirrored image, z relative depth
    world: np.ndarray           # (21, 3) float32; metres, origin near the hand's centre
    gesture: str                # MediaPipe built-in category, "None" if no gesture
    gesture_score: float

@dataclass(frozen=True, slots=True)
class HandFrame:
    seq: int                    # increments per frame produced by the source
    t_capture: float            # time.monotonic() when the camera frame was read
    t_result: float             # time.monotonic() when inference finished
    size: tuple[int, int]       # (width, height) of the inference image
    hands: tuple[Hand, ...]
    image: np.ndarray | None    # RGB, mirrored; None in replay mode

@dataclass(frozen=True, slots=True)
class TrackedHand:              # produced by HandTracker on the main thread
    id: int                     # stable while the hand stays visible
    hand: Hand                  # with smoothed landmarks
    age: int                    # frames since first seen

@dataclass(frozen=True, slots=True)
class GestureEvent:
    type: str                   # e.g. "pinch_start", "pinch_move", "pinch_end", "fist_start", "custom"
    hand_id: int
    handedness: Handedness
    t: float
    pos: tuple[float, float]    # normalized screen coordinates [0, 1], mirrored
    strength: float = 0.0       # 0..1, meaning depends on type
    vel: tuple[float, float] = (0.0, 0.0)  # normalized units per second
    data: Mapping[str, Any] = field(default_factory=dict)
```

`GestureEvent` serializes to the JSON shape proposed in the brainstorm. Scenes consume
`GestureEvent`s and also get one `HandState` per tracked hand (a `TrackedHand` plus computed
features such as pinch ratio, finger count, palm centre, and velocity; see
[Phase 0 §8](phase0-foundation.md#8-detectors-and-engine-gesturesdetectorspy-gesturesenginepy-portable-core)).
These drive continuous control, such as a hand acting as a paddle or theremin pitch.

## 3. Package layout

```
ai-playground/                      (repo root)
├── pyproject.toml   uv.lock   .python-version (3.12)   README.md   LICENSE
├── config/
│   └── default.toml                all tunables; venue overrides via --config
├── assets/
│   ├── models/                     gitignored; filled by scripts/fetch_models.py (pinned URL + SHA-256)
│   ├── charnet/                    our character CNN: weights .npz + model card .json (committed)
│   └── fonts/
├── scripts/
│   ├── fetch_models.py
│   └── eval_charnet.py             Air Writing accuracy on stroke files
├── training/
│   └── charnet/                    Colab training (PyTorch); its deps never enter uv.lock
├── src/ai_playground/
│   ├── __main__.py                 CLI entry: `uv run ai-playground [--replay f] [--record f] [--scene s] ...`
│   ├── app.py                      main loop, scene manager, hotkeys
│   ├── booth.py                    booth shell state: idle timer, hold-to-quit, onboarding hints
│   ├── config.py                   TOML → frozen dataclasses
│   ├── types.py
│   ├── sources/        camera.py  live.py (camera + recognizer worker)  replay.py  recorder.py  slot.py
│   ├── tracking/       recognizer.py (the ONLY module importing mediapipe)  identity.py  filters.py
│   ├── gestures/       features.py  detectors.py  engine.py            ◄ portable core
│   ├── ml/             strokes.py  charnet.py                           ◄ portable (Air Writing)
│   │                   features.py  dataset.py  classifier.py  pca.py   (Train-a-Gesture, parked)
│   ├── render/         viewport.py  ui.py  overlay.py  hud.py  keys.py  booth.py  ink.py  background.py  effects.py  text.py
│   ├── physics/        world.py                                        (Phase A)
│   ├── audio/          sfx.py (Phase A)  engine.py  voices.py  dsp.py  scales.py  (Phase B; scales.py portable)
│   └── scenes/         base.py  hand_rules.py  vision.py  geometry.py  writing.py  ·  later: attract.py  train.py  synth.py  orchestra.py
├── tests/
│   ├── fixtures/                   *.jsonl landmark recordings (small, committed)
│   ├── synthetic.py                build hands programmatically (pinch of distance d, fist, open palm…)
│   └── test_*.py
└── docs/
```

## 4. The portable core rule

`gestures/`, `ml/strokes.py`, `ml/charnet.py`, `ml/features.py`, `ml/pca.py`, and `audio/scales.py`
may import **only the standard library and NumPy**. A test parses their imports with `ast` and fails if this is broken.

Why:

1. They can be unit-tested with synthetic data and fixtures alone.
2. Tuning gesture behavior never touches rendering code.
3. They stay translatable to another platform (e.g. JS) if that is ever wanted.

## 5. Scenes

```python
class Scene(Protocol):
    name: str
    def enter(self, ctx: AppContext) -> None: ...
    def exit(self) -> None: ...
    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None: ...
    def update(self, dt: float) -> None: ...
    def draw(self, surface: pygame.Surface) -> None: ...
```

`AppContext` holds the config, viewport, sound effects (`Sfx`, Phase A), the audio engine (Phase B),
the trained custom-gesture model (parked), and a `switch_scene()` callback. The app draws the
**Vision overlay** on top of any scene when toggled. That is the "AI Vision Mode ↔ Magic Mode"
switch from the brainstorm.

## 6. Coordinates

- **Normalized image coordinates** `(x, y) ∈ [0,1]²`: what MediaPipe returns. They already match
  the mirrored image, because we flip before inference.
- **Screen pixels:** `render/viewport.py` maps normalized → pixels using *cover* scaling (camera
  image fills the window, edges cropped). It also provides the inverse mapping. Every scene uses
  the viewport. Nobody multiplies by window size by hand.
- **UI pixels:** text, panels, cursors and line widths are laid out for a 720 px tall window and
  go through `render/ui.py` (`ui.px(...)`, and `text.font` scales its sizes), which multiplies by
  window height / 720 times `[display] ui_scale`. A maximized or fullscreen window on a 5K
  monitor shows the same layout, only bigger.
- **Physics space:** pymunk uses screen pixels with y pointing down (gravity `+y`) to avoid a third
  coordinate system.
- **Hand-relative features** (for pinch and ML) use **world landmarks** (metres), which don't
  depend on distance to the camera.

## 7. Configuration

`config/default.toml` holds every threshold and tunable, grouped by module (`[camera]`,
`[recognizer]`, `[tracking]`, `[gestures.pinch]`, `[display]`, `[audio]`, …). `--config venue.toml`
deep-merges overrides on top, so venue-specific tuning such as camera index, lighting confidence,
or display never needs a code change. `config.py` validates against frozen dataclasses and fails
fast on unknown keys.

## 8. Testing and CI

- `pytest`, with no camera, display, or audio device needed:
  - `SDL_VIDEODRIVER=dummy` for scene smoke tests;
  - the audio engine runs offline (render N blocks into an array).
- **Unit:** features, filters, detectors, identity, k-NN pipeline, scales, DSP (synthetic inputs).
- **Fixture:** replay committed `.jsonl` clips through tracker + engine and assert the expected
  event sequences.
- **Recognizer smoke test:** load the model and run on a still image in `tests/fixtures/images/`.
  This verifies the MediaPipe wheel works on that platform.
- **CI:** GitHub Actions, matrix `macos-latest` (arm64) × `ubuntu-latest`. Steps: `uv sync --locked`,
  `ruff check`, `ruff format --check`, `pytest`, with a cache for the model download.

## 9. Dependencies

Runtime: `mediapipe` (brings `opencv-contrib-python`, `numpy`, `sounddevice`), `pygame-ce`,
`pymunk` (`scikit-learn` only once Train-a-Gesture returns). Dev: `pytest`, `ruff`.
**Training only** (Colab, `training/charnet/requirements.txt`, never in `uv.lock`): `torch`, and
what the training scripts need for data prep (PLAN D12).
**Linux system package:** `libportaudio2` (the `sounddevice` wheels bundle PortAudio on macOS but
not on Linux).

## 10. Performance budget (per 16.7 ms main-loop frame)

| Stage | Budget |
|-------|--------|
| Tracking + filters + gesture engine | ≤ 1 ms |
| Scene update incl. physics (≤ 60 bodies) | ≤ 3 ms |
| Camera background blit (scaled) | ≤ 3 ms |
| Scene draw + effects + overlay | ≤ 6 ms |
| Headroom | ≥ 3 ms |

Inference runs in the worker thread at about 10–25 ms per frame on CPU and does not count against
this budget. The GIL does mean it competes for time. MediaPipe and OpenCV release the GIL in
native code, but we watch for this in Phase 0 profiling.
