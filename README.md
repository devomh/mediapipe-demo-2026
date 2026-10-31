# AI Playground

*"Your hands are the controller."* A live, self-guided exhibit that uses MediaPipe hand tracking
to get prospective students interested in data science and machine learning.

Plan and specs: [`docs/PLAN.md`](docs/PLAN.md) · [`docs/specs/`](docs/specs/).
**Status:** Phase 0 (foundation) is done: camera → MediaPipe → tracking → gesture events, the
Vision scene, and record/replay. Phase A in progress: Magic Geometry (pinch to make, inflate, drop
and throw shapes).

## Setup

Supported: macOS on Apple Silicon (primary) and Ubuntu x86_64. Needs [uv](https://docs.astral.sh/uv/).

```sh
# Ubuntu only: PortAudio (mediapipe imports sounddevice, whose Linux wheel doesn't bundle it)
# and EGL/GLES (MediaPipe loads them even for CPU inference). Desktop installs usually have them.
sudo apt install libportaudio2 libegl1 libgles2

uv sync --locked
uv run python scripts/fetch_models.py      # gesture_recognizer.task, SHA-256 verified
uv run ai-playground
```

**Linux sound server:** `import mediapipe` initialises PortAudio, and Ubuntu's PortAudio fails
with `PulseAudio_Initialize: Can't connect to server` if no sound server is running. A normal
desktop session has one (PipeWire). Headless machines need `pulseaudio --start`.

**macOS camera permission:** the first run asks for camera access for your *terminal app*
(Terminal, iTerm, VS Code, …). If you denied it: System Settings › Privacy & Security › Camera →
enable the terminal app, then restart it. **Linux:** your user needs to be in the `video` group.

## Running

```sh
uv run ai-playground                          # live camera, Vision scene
uv run ai-playground --scene geometry         # Magic Geometry: pinch, hold, release / throw
uv run ai-playground --list-cameras           # which camera indices work
uv run ai-playground --camera 1 --fullscreen --display 1
uv run ai-playground --config venue.toml      # venue overrides, deep-merged over config/default.toml
uv run ai-playground --replay tests/fixtures/right_pinch_x5.jsonl.gz --loop --speed 0.5
uv run ai-playground --debug                  # logs every gesture event as JSON
```

| Key | Action |
|-----|--------|
| `K` | Show/hide this list of keys on screen, with each toggle's current state |
| `V` | Vision overlay (skeleton, panels, numbers) |
| `H` | HUD (camera FPS, inference, render FPS, latency) |
| `B` | Background: camera → dimmed → hidden |
| `L` | Landmark index numbers |
| `R` | Start/stop recording to `recordings/YYYYmmdd-HHMMSS.jsonl.gz` |
| `Space` | Ground-truth mark (while recording) |
| `F` | Fullscreen |
| `1` / `2` | Vision / Geometry scene |
| `C` | Clear all shapes (Geometry) |
| `Esc` | Quit |

Every threshold lives in [`config/default.toml`](config/default.toml). A venue file only needs
the keys it changes, e.g.:

```toml
[camera]
index = 1
[tracking]
min_hand_size = 0.10   # ignore more of the crowd behind the visitor
```

## Recording fixtures

Recordings hold **landmarks only, never video** (~55 KB/s). Clips in `tests/fixtures/` drive
the fixture tests in `tests/test_fixtures.py`, which skip until a clip exists. To record one:

```sh
uv run ai-playground --record tests/fixtures/right_pinch_x5.jsonl.gz \
    --note "right hand, 5 slow pinches"
# press Space at each pinch, Esc when done (5-15 s)
```

The full list of clips and what each one tests is in
[Phase 0 §9](docs/specs/phase0-foundation.md#9-fixture-set-recorded-in-milestone-03-committed-to-testsfixtures).
When a gesture misbehaves, record it, add it as a fixture, and write a failing test first.

## Development

```sh
uv run pytest            # no camera, display, or audio device needed
uv run ruff check . && uv run ruff format --check .
```

Layout and rules: [architecture](docs/specs/architecture.md). In short:

- `tracking/recognizer.py` is the only module that imports `mediapipe`.
- `gestures/` is the **portable core**: standard library + NumPy only (enforced by a test).
- Import `pygame` before `cv2`/`mediapipe`. On macOS both bundle SDL2 (OpenCV through
  FFmpeg), and the first one loaded wins its Objective-C classes. The runtime prints
  `Class SDL… is implemented in both …` warnings at startup. With pygame loaded first no problems
  have shown up so far, but watch for it in the soak tests.

## License

MIT, see [LICENSE](LICENSE). Sample hand images in `tests/fixtures/images/` are from MediaPipe
(Apache-2.0).
