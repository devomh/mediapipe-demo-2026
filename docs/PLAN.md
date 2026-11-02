# AI Playground — Project Plan

*"Your hands are the controller."* A live, self-guided exhibit that uses MediaPipe hand tracking
to get prospective students interested in data science and machine learning.

Source brainstorm: [`ideas01.md`](ideas01.md). Specs: [`specs/`](specs/).

---

## 1. Goals and non-goals

**Goals**

- A fun, self-explanatory exhibit for an event. A visitor walks up, raises a hand, and is
  creating something within ~10 seconds, with no instructions from staff.
- The event is a **booth or table with people passing by**, not a room visitors settle into.
  Each demo must pay off within seconds and make sense in under a minute (D11).
- Every mode can show the machine learning behind it (landmarks, numbers, predictions), and one
  mode shows **a neural network reading what the visitor writes in the air**.
- Runs offline on a single laptop: **MacBook (Apple Silicon)** as the primary machine and an
  **Ubuntu laptop (x86_64)** as the secondary, from the same repo and `uv.lock`.

**Non-goals (for now)**

- A web/browser version. The brainstorm's "Phase 5 Web Exhibit" is out of scope. The portable
  core rule ([architecture §4](specs/architecture.md#4-the-portable-core-rule)) keeps that option
  open but costs nothing now.
- Intel Macs and Windows. MediaPipe 1.2.0 ships wheels for Windows, but we neither test nor
  support it.
- GPU inference. CPU inference is fast enough for 2 hands at 640×480.

## 2. Key decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Python 3.12, `uv`, committed `uv.lock`, `uv sync --locked` | Same environment on macOS arm64 and Linux x86_64 |
| D2 | MediaPipe **Tasks API**, **Gesture Recognizer** task (not Hand Landmarker, not legacy `mp.solutions`) | Same 21 landmarks + handedness, plus 7 built-in gestures (`Closed_Fist`, `Open_Palm`, …) for free |
| D3 | Use the `cv2` that MediaPipe brings in (`opencv-contrib-python`); never add `opencv-python` | Two `cv2` distributions in one environment conflict |
| D4 | `pygame-ce` for window, rendering, and input, on the **main thread** | macOS requires UI on the main thread; one rendering stack for all modes |
| D5 | `pymunk` for 2D physics | Collisions, joints, and collision callbacks without writing our own physics |
| D6 | `sounddevice` + NumPy callback synth for audio | Low latency, no external synth; MediaPipe already depends on it |
| D7 | `scikit-learn` (k-NN) for Train-a-Gesture *(parked with Train-a-Gesture, D11)* | Students will meet the same library in class; training is instant |
| D8 | Record **landmarks only, never video** | Privacy and consent at a public event, and a teaching point |
| D9 | Gesture logic is pure NumPy and tested with recorded fixtures | Testable without a camera; CI runs on both OSes |
| D10 | ~~Train-a-Gesture ships in Phase A~~ *Superseded by D11 (2026-11-01)* | It was the part that taught ML; Air Writing now covers that |
| D11 | **Booth format** (2026-11-01): every demo pays off in seconds. Train-a-Gesture is parked; **Air Writing** (pinch to write a character in the air, a CNN reads it) becomes Phase A's ML mode | Passers-by won't stay ~60 s for a guided training flow. "The computer read what I wrote" lands in seconds and is still visibly ML |
| D12 | Character recognizer: a **CNN trained offline in Colab** (PyTorch) on EMNIST digits + uppercase letters (36 classes). Weights ship as `.npz`, **inference is pure NumPy** | No new runtime dependency (no torch or onnxruntime in `uv.lock`), ~1 ms per read, identical on both OSes |
| D13 | Geometry **sound effects** with `pygame.mixer`, sounds synthesized with NumPy at startup | Comes with pygame-ce; effects don't need the Phase B synth engine; no sample files, so no licensing questions |
| D14 | **Booth-minimum shell:** fullscreen, idle reset, one hint line per scene, mute, operator hotkeys. Attract screen and dwell-button navigation are parked | Staff at the table switch scenes; the screen must just never be left in a messy state |

## 3. Phases

```
Phase 0  Foundation ──► Phase A  Magic Geometry + Air Writing ──► later: Phase B  AirSynth ──► Phase C  Physics Orchestra
 (engine, no fun yet)    (the booth exhibit)                             (audio engine)        (A + B combined, exhibit polish)

Parked (D11, D14): Train-a-Gesture, attract screen, dwell-button navigation. Specs kept for later.
```

Every phase ends with something that runs, can be demoed, and is tested on **both** macOS and
Ubuntu.

### Phase 0 — Foundation · [spec](specs/phase0-foundation.md)

Camera → MediaPipe → identity tracking + smoothing → gesture engine → events, plus the
**Vision scene** (landmarks, numbers, detected gestures) and landmark **record and replay**.

**Exit criteria**

- 2 hands tracked at ≥ 25 FPS (target 30) at 640×480 on the MacBook. Render loop at 60 FPS.
- Pinch start/end shows no flicker during a 30 s hold, and fires within ≤ 2 frames of a real pinch.
- Left/right labels are correct for the visitor (raise your right hand → "Right").
- Pipeline latency (capture → displayed) is measured, shown in the HUD, and under 100 ms.
- `--record` / `--replay` work. At least 6 fixture clips are committed. CI is green on macOS and Ubuntu.

### Phase A — Magic Geometry + Air Writing · [Geometry spec](specs/phaseA-magic-geometry.md) · [Air Writing spec](specs/phaseA-air-writing.md)

Pinch to create shapes, throw them, push them with your hand, grab them with a fist. Gravity,
collisions, colors, particles, and **sound effects**. Then **Air Writing**: pinch to write a digit
or capital letter in the air, make a fist, and a neural network reads it, showing the 28×28 image
it sees and its top three guesses. Includes the booth-minimum shell (fullscreen, idle reset, hints,
mute, operator hotkeys).

*Changed 2026-11-01 (D11, D14):* Train-a-Gesture, the attract screen, and dwell-button navigation
are parked. Their spec sections stay in place for later.

**Exit criteria**

- A first-time user (someone not on the team) creates and throws a shape within 10 s, unprompted.
- A first-time user gets a character read within 30 s, with at most a one-line hint.
- Air Writing accuracy on the committed air-written evaluation set (digits + uppercase):
  ≥ 90 % top-1 and ≥ 97 % top-3. *Provisional:* revisit once the first model is measured
  (Air Writing W.3). *Changed in W.2:* the set has one writer in three styles (108 characters,
  [Air Writing §8](specs/phaseA-air-writing.md#8-air-written-data-evaluation-and-fine-tuning)),
  so it is reported with a confidence interval, and the A.8 user tests add a multi-writer
  figure by logging whether each read was right.
- 30-minute soak test with no crash and no frame-rate degradation.

### Phase B — AirSynth · [spec](specs/phaseB-airsynth.md)

A real-time audio engine plus two instruments: a **Theremin** (continuous pitch quantized to a
pentatonic scale, other hand for volume and filter) and an **Invisible Piano** (note columns, pinch
to play). Waveform/spectrum view as the "discover" layer.

**Exit criteria**

- No audible glitches (audio underruns = 0) over a 10-minute session while rendering at 60 FPS.
- Gesture → sound latency < 100 ms (pipeline + audio buffer, measured).

### Phase C — Physics Orchestra · [spec](specs/phaseC-physics-orchestra.md)

Geometry shapes make sounds when they collide. Timbre comes from the shape, pitch from its size,
volume from the impact. Hand "wind". Optional beat quantization. Final exhibit flow and
operations checklist.

**Exit criteria**

- 2-hour soak test on both machines: no crash, no underruns, stable memory.
- Full visitor flow (attract → play → discover → train) rehearsed with ≥ 5 people new to the exhibit.

## 4. Working method

- **Per-phase workflow:** refine that phase's spec → break it into issues → build → test on the
  MacBook → test on the Ubuntu laptop → demo → tag a release (`v0.0`, `v0.A`, …).
- Phase 0 and A specs are detailed. Phase B and C specs set direction and are refined once their
  phase starts, with what we learned in earlier phases.
- **Branching:** `main` is always runnable. Work happens on short feature branches with PRs, and CI
  must pass.
- **Fixtures are a team asset:** whenever a gesture misbehaves, record a clip of it, add it to
  `tests/fixtures/`, and write a failing test before fixing it.

## 5. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Venue lighting or backlight hurts tracking | High | Test in the venue early; ring light in the kit; `min_detection_confidence` in config |
| Bystanders' hands get tracked | High | Ignore hands smaller than `min_hand_size` (far from camera); mark the floor spot |
| Python audio callback underruns under rendering load | Medium | Larger block size; fall back to running audio in a separate process ([Phase B §2](specs/phaseB-airsynth.md#2-audio-engine-audioenginepy-audiovoicespy-audiodsppy)) |
| macOS camera permission or wrong camera index on event day | Medium | Startup check with clear on-screen error; `--list-cameras`; ops checklist |
| Gestures feel unreliable for new users | Medium | Small gesture vocabulary, hysteresis, onboarding hints, test with new users at every phase |
| Air-written characters look unlike EMNIST's pen scans (wobbly, even stroke width, slanted), so accuracy drops | High | One framing function shared by training and the app; augmentation that imitates air-writing; measure on our own air-written set before building the UI polish; fine-tune on it if needed ([Air Writing §5](specs/phaseA-air-writing.md#5-model)) |
| The evaluation set has one writer (W.2), so it may overstate accuracy for visitors | Medium | Three writing styles per character, a confidence interval with every accuracy, and a multi-writer figure from the A.8 user tests ([Air Writing §8](specs/phaseA-air-writing.md#8-air-written-data-evaluation-and-fine-tuning)) |
| Booth noise drowns the sound effects, or they annoy the neighbouring booths | Medium | Volume in config, `M` mutes, test with the event speakers |
| MediaPipe API change in a future release | Low | Version pinned in `uv.lock`; MediaPipe is called from one module only (`tracking/recognizer.py`) |
| Ubuntu laptop missing system libraries (PortAudio) | Medium | `sudo apt install libportaudio2` in README; startup check. Needed from Phase 0: `import mediapipe` imports `sounddevice` |
| Duplicate SDL2 on macOS: the OpenCV wheel bundles its own (via FFmpeg) next to pygame-ce's; the ObjC runtime warns of "mysterious crashes" | Low | Import pygame before cv2/mediapipe everywhere (`__main__.py`, `tests/conftest.py`); watch for it in soak tests. The headless OpenCV wheel bundles SDL2 too, so swapping wheels doesn't help |
