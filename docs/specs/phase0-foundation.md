# Phase 0 — Foundation

**Outcome:** a window showing the mirrored camera with tracked hands, their landmarks and numbers,
and reliably detected pinch/fist/palm events. Sessions can be recorded and replayed without a
camera. No "fun" yet; this is the engine every later phase stands on.

Read with: [architecture](architecture.md), [plan](../PLAN.md).

---

## 1. Scope

**In:** project scaffold and CI; camera + recognizer worker; record/replay; hand identity
tracking; One Euro smoothing; feature extraction; pinch/fist/palm detectors; gesture engine; the
Vision scene; HUD; config; CLI; startup checks.

**Out:** physics, shapes, audio, ML training, attract screen/onboarding (Phase A).

## 2. Milestones

| # | Milestone | Needs camera | Done when |
|---|-----------|:-:|-----------|
| 0.1 | Scaffold: `uv` project (src layout), deps, ruff, pytest, CI matrix, README, LICENSE, `.gitignore`, `fetch_models.py`, recognizer smoke test | – | CI green on macOS + Ubuntu |
| 0.2 | Camera + recognizer worker + `LatestSlot`; pygame window draws camera + raw landmarks; HUD timings | ✔ | Hands visible at ≥ 25 FPS on MacBook |
| 0.3 | Recorder + replay (+ `.jsonl.gz`); record the fixture set (§9) | ✔ | Replay of a clip looks identical to live |
| 0.4 | Identity tracker + One Euro filter | – (fixtures) | Crossing-hands fixture keeps IDs; jitter visibly reduced |
| 0.5 | Features, detectors, engine, events | – (fixtures) | Fixture event tests pass (§10) |
| 0.6 | Vision scene, config system, CLI, startup checks | ✔ | Exit criteria in PLAN §3 met |
| 0.7 | Verify on the Ubuntu laptop, measure, write up numbers, tag `v0.0` | ✔ | Numbers recorded in `docs/measurements.md` |

Milestone 0.1 can be done on any machine. From 0.2 on, development happens on the MacBook.

## 3. Camera (`sources/camera.py`)

- Open with `cv2.VideoCapture(index, backend)`. Backend: `CAP_AVFOUNDATION` on macOS, `CAP_V4L2`
  on Linux, `CAP_ANY` as fallback.
- Request `width=640, height=480, fps=30` (config). On Linux, also request `MJPG` FOURCC
  (many webcams only reach 30 FPS uncompressed at low resolutions). Log the values actually
  negotiated.
- Set `CAP_PROP_BUFFERSIZE=1` and ignore it if unsupported.
- `--list-cameras` probes indices 0–5 and prints which open, with resolutions.
- If the camera fails to open or returns no frames within 3 s, the app shows a full-screen error
  with a platform hint: *macOS → System Settings › Privacy & Security › Camera → enable your
  terminal app*; *Linux → check `/dev/video*` and the `video` group*.

## 4. Recognizer worker (`tracking/recognizer.py`, `sources/live.py`)

- `GestureRecognizer` from `mediapipe.tasks.python.vision`, `running_mode=VIDEO`,
  `num_hands=2`. Confidence thresholds come from config (defaults 0.5 / 0.5 / 0.5).
- Model: `assets/models/gesture_recognizer.task`, fetched by `scripts/fetch_models.py` from a
  **versioned** URL (not `latest`) and verified against a SHA-256 stored in the script.
- Worker loop: `read()` → `t_capture = monotonic()` → `cv2.flip(frame, 1)` → BGR→RGB →
  `recognize_for_video(mp.Image(...), timestamp_ms)` → build `HandFrame` → `slot.put(frame)`.
  The timestamp must be strictly increasing, so derive it from `t_capture`.
- **Mirroring:** we flip before inference so the screen behaves like a mirror. MediaPipe's docs
  suggest its handedness then matches the visitor's own left/right, but the milestone 0.2 check
  on screen ("raise your right hand") showed the opposite: right hand, mirror view, labelled
  "Left". `recognizer.py` therefore swaps the label, and `Hand.handedness` is always the
  visitor's own hand.
- `recognizer.py` is the only file that imports `mediapipe`. It converts results into `Hand`
  objects (NumPy arrays), so nothing downstream sees MediaPipe types.
- Stop cleanly: worker checks a stop `Event`, releases the camera, closes the recognizer.

## 5. Record and replay (`sources/recorder.py`, `sources/replay.py`)

**Format:** JSON Lines, optionally gzip (`.jsonl.gz`), landmarks only, **never images**.

```jsonc
// line 1: header
{"type":"header","schema":1,"app":"0.0.0","mediapipe":"1.2.0","model_sha256":"…","size":[640,480],
 "mirrored":true,"created":"2026-11-03T14:05:00Z","note":"right hand, 5 slow pinches"}
// frame lines (t relative to first frame, seconds; lat = t_result - t_capture)
{"type":"frame","seq":17,"t":0.5667,"lat":0.0141,
 "hands":[{"h":"Right","hs":0.98,"g":"Open_Palm","gs":0.87,"lm":[/*63 floats*/],"w":[/*63 floats*/]}]}
// optional ground-truth marks (operator presses Space while recording)
{"type":"mark","t":1.234,"label":"pinch"}
```

- Floats are rounded to 4 decimals (about 1.8 KB per 2-hand frame, about 55 KB/s).
- `--record PATH` records from startup. Hotkey `R` toggles recording into
  `recordings/YYYYmmdd-HHMMSS.jsonl.gz` (gitignored). A red dot shows in the HUD while recording.
- `--replay PATH [--loop] [--speed X]`: `ReplaySource` runs in the worker thread and publishes
  `HandFrame`s (with `image=None`) on the recorded timing. Scenes draw a dark grid instead of the
  camera image.
- `replay.load(path) -> list[HandFrame]` is also used directly by tests, with no threads involved.

## 6. Hand identity and smoothing (`tracking/identity.py`, `tracking/filters.py`)

**HandTracker** runs on the main thread, once per new `HandFrame`:

1. **Ignore small hands:** drop any hand whose image-space scale (wrist→middle-MCP distance ÷
   image height) is below `min_hand_size` (default 0.08). This filters out bystanders in the
   background.
2. **Match to existing tracks:** greedy nearest-wrist matching in normalized image space, with a
   gate of 0.25. With at most 2 hands, greedy matching is enough.
3. Unmatched detections start new tracks (IDs increase monotonically). An unmatched track
   survives `grace_frames` (default 4) before it is dropped. This avoids end/start flicker when
   a hand briefly drops out.
4. **Stable handedness:** a track's handedness is the majority vote over its last 15 frames.
   MediaPipe sometimes swaps labels for a frame or two.
5. **Smoothing:** each track has a vectorized **One Euro filter** over its (21, 3) image landmarks
   and its (21, 3) world landmarks, timed by `t_capture`. Defaults: `min_cutoff=1.0`,
   `d_cutoff=1.0`, and `beta=10` (image) / `beta=40` (world), set separately because the units
   differ (normalized units/s vs metres/s). These are config values, tuned in milestone 0.4
   against the jitter and lag seen in fixtures.
   *Changed during implementation:* the original `beta=0.02` is a value for pixel units. On our
   units it delayed `pinch_start` by 9 frames (~300 ms) on a synthetic fast pinch, which fails the
   ≤ 2-frame exit criterion; the new defaults give 1 frame (`tests/test_engine.py`).

## 7. Features (`gestures/features.py`, portable core)

Pure functions over landmark arrays. All distances that matter for gestures use **world
landmarks**, so they don't depend on how far the hand is from the camera.

| Feature | Definition |
|---------|------------|
| `palm_length` | ‖w[0] − w[9]‖ (wrist → middle MCP) |
| `pinch_ratio(finger)` | ‖w[4] − w[tip(finger)]‖ / palm_length. Pinched ≈ 0.1–0.2, open ≈ 0.8–1.2 |
| `pinch_point` | midpoint of image landmarks 4 and 8 (normalized screen coordinates) |
| `finger_extended[5]` | non-thumb fingers: ‖tip − wrist‖ / ‖PIP − wrist‖ > 1.2; thumb: ‖tip − pinky MCP‖ / palm_length > 0.85. Thresholds in config. *(Changed from joint angles: on real world landmarks curled fingers measured up to 152° at the PIP and extended ones as low as 161°, and thumb IP angles overlapped the same way (143° open vs 146° folded). The distance rules separate them on the sample images and a live recording: fingers: extended median 1.38 but down to ~1.1 while a hand enters the frame, curled ≤ 1.17 (fist seen edge-on), so the bands touch and fixtures should tune this; thumb ≥ 0.88 vs ≤ 0.79; `tests/test_recognizer.py`.)* |
| `finger_count` | number of extended fingers |
| `palm_center` | mean of image landmarks 0, 5, 9, 13, 17 |
| `palm_roll` | angle of image vector 5 → 17 (radians), used for the AirSynth filter |
| `velocity(point)` | finite difference over a ~100 ms history window (normalized units/s) |
| `hand_scale_image` | image-space wrist→MCP distance; used for the bystander filter |

## 8. Detectors and engine (`gestures/detectors.py`, `gestures/engine.py`, portable core)

Each detector is a small state machine per hand track. Each frame it gets the hand's features and
returns zero or more events.

**PinchDetector (thumb + index)**
- `IDLE → PINCHED` when `pinch_ratio < enter` (0.25) for `min_frames` (2) consecutive frames → emit `pinch_start`.
- `PINCHED → IDLE` when `pinch_ratio > exit` (0.40) for `min_frames` frames → emit `pinch_end`.
  The event carries the release velocity, needed for throwing in Phase A.
- While `PINCHED`, emit `pinch_move` every frame with `pos = pinch_point` and
  `strength = clamp(1 − ratio/enter)`.
- Having two thresholds (hysteresis) is what prevents flicker. Thresholds are configurable and
  tuned with fixtures.

**BuiltinGestureDetector:** fist (`Closed_Fist`) and palm (`Open_Palm`)
- Start when the category is seen with `score ≥ 0.6` for 3 consecutive frames. End when it is
  absent for 3 frames. Emits `fist_start/fist_end` and `palm_start/palm_end`.
- A fist is suppressed while a pinch is active on the same hand, so one hand can't be in both
  states at once.

**Track lifecycle:** emit `hand_enter` and `hand_exit`. When a track is dropped, every active
detector on it emits its `*_end` with `data={"reason": "lost"}`, so scenes never keep stale state.

**GestureEngine.update(tracked_hands, t) -> (events, states)**
- `states: list[HandState]`, where `HandState = TrackedHand + computed features + active detector
  states`. Scenes read these for continuous control (paddles, pitch, …).
- The engine works per hand track and has no knowledge of rendering.

## 9. Fixture set (recorded in milestone 0.3, committed to `tests/fixtures/`)

Each clip is 5–15 s, `.jsonl.gz`, and has a one-line description in its header `note`.

| File | Content | Used to test |
|------|---------|--------------|
| `no_hands` | empty view, person moving in background | no events |
| `right_static_open` | right hand open, still | jitter level, no events after `hand_enter` |
| `right_pinch_x5` | 5 slow pinches, Space marks | exactly 5 start/end pairs, timing vs marks |
| `right_pinch_hold` | one 10 s pinch while moving | 1 start, 1 end, no flicker |
| `both_pinch` | alternate and simultaneous pinches | per-hand independence |
| `fist_palm_cycle` | open → fist ×4 | fist/palm events, no pinch false positives |
| `hands_crossing` | two hands cross paths | stable IDs and handedness |
| `enter_exit` | hand leaves and re-enters the frame 3× | lifecycle events, `reason: lost` |
| `far_bystander` | small hand far in the background | filtered out by `min_hand_size` |

Ideally record most clips twice, with two different people, and in two lighting conditions.

## 10. Tests

- **Synthetic** (`tests/synthetic.py` builds anatomically plausible hands with a given pinch
  distance, finger count, and position): feature values, hysteresis boundaries, `min_frames`
  debounce, identity gating, One Euro step response.
- **Fixtures:** the expectations in the table above.
- **Portable-core import check** ([architecture §4](architecture.md#4-the-portable-core-rule)).
- **Smoke:** recognizer on a still image; Vision scene runs 120 frames on replay with
  `SDL_VIDEODRIVER=dummy`.

## 11. Vision scene and HUD

**Vision scene.** This is also the "discover" layer for later phases, as an overlay.
- Mirrored camera image (with `B` to toggle a dimmed or hidden background).
- Skeleton: bones colored per finger, joints as dots, fingertips larger. `L` toggles landmark
  index numbers.
- Per hand, a panel next to the wrist with:
  - handedness and score;
  - built-in gesture and score;
  - a **pinch-ratio bar** with the enter/exit thresholds marked and the state lit when pinched;
  - finger count.
- A side panel, **"What the computer sees"**: a live 21 × 3 table of numbers for one hand, with
  the row under the fingertip highlighted.
- Events flash as small labels at their position (e.g. "pinch_start") for 0.5 s.

**HUD** (`H` toggles), top-left, small monospace:
- camera FPS, inference ms, render FPS;
- **pipeline latency** = display time − `t_capture`, as mean and p95 over the last 2 s;
- hands tracked, recording indicator.

**Hotkeys (Phase 0):** `V` vision overlay · `H` HUD · `B` background · `L` landmark numbers ·
`R` record · `Space` mark (while recording) · `F` fullscreen · `Esc` quit.

## 12. Configuration and CLI

```
uv run ai-playground [--scene vision] [--config venue.toml] [--camera N] [--list-cameras]
                     [--record PATH] [--replay PATH [--loop] [--speed X]]
                     [--fullscreen] [--display N] [--debug]
```

`config/default.toml` sections for Phase 0: `[camera]`, `[recognizer]`, `[tracking]`
(`min_hand_size`, `grace_frames`, `one_euro.*`), `[gestures.pinch]`, `[gestures.builtin]`,
`[display]` (window size, fullscreen, display index, target FPS), `[hud]`.

**Startup checks** happen before the window opens. Each failure gives a one-line fix:
- model file missing → `uv run python scripts/fetch_models.py`;
- camera cannot be opened → platform hint (§3);
- Python is not 3.12 → note to use `uv run`.

## 13. Measurements to record (`docs/measurements.md`, both machines)

Camera FPS actually negotiated · inference ms (mean, p95) · render FPS · pipeline latency (mean,
p95) · CPU % · pinch-detection delay vs Space marks · false pinch starts per minute in
`right_static_open` and `fist_palm_cycle`.
