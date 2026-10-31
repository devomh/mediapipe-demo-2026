# Phase B — AirSynth

**Outcome:** hands become a musical instrument, in two modes: **Theremin** (continuous) and
**Invisible Piano** (discrete notes). A real-time audio engine gets built here and is reused by
Phase C.

Status: **direction spec.** Refine at the start of the phase with Phase 0/A measurements.
Builds on: [Phase 0](phase0-foundation.md), [architecture](architecture.md).

---

## 1. Principles

- **Always sounds good:** default to pentatonic scales and snap pitch to the scale, so random
  movement still sounds musical.
- **Audio never waits on vision:** synthesis runs in the audio callback. The main thread only
  sets targets and queues note events.
- **Latency is a feature:** measure it, show it in the HUD, and choose interactions that tolerate it.
  Continuous control (theremin) tolerates ~100 ms. Percussive strikes don't, so **Air Drums are
  out of scope** unless measured latency turns out much lower than expected.

## 2. Audio engine (`audio/engine.py`, `audio/voices.py`, `audio/dsp.py`)

**Stream:** `sounddevice.OutputStream(samplerate=48000, channels=2, dtype="float32",
blocksize=256, latency="low")`. 256 samples is about 5.3 ms per block. Configurable.

**Callback contract:**
- Vectorized NumPy only: no locks, no logging, no allocation-heavy work.
- Must finish well within the block duration.
- Underflow status flags are counted and shown in the HUD.

**Building blocks**

| Component | Implementation |
|-----------|----------------|
| Parameter smoothing | Per block, a linear ramp from the current value to the target (`np.linspace`), so there is no zipper noise |
| Oscillators | Phase accumulator via `np.cumsum` of per-sample increments, so glide is free. Shapes: sine, triangle, **polyBLEP** saw (anti-aliased), noise |
| Envelope | ADSR, evaluated per block, vectorized per segment |
| Voices | Fixed pool of 8 (16 in Phase C), voice stealing (oldest released first, then quietest) |
| Filter | Biquad low-pass via `scipy.signal.lfilter` with persistent `zi` state. Coefficients are updated per block from the smoothed cutoff. SciPy is already installed via scikit-learn |
| Delay/echo | Ring buffer. Delay ≥ block size means a feedback delay can be computed per block, vectorized |
| Reverb | *Stretch:* Schroeder combs (delays > block size, vectorized) + allpasses via `lfilter` |
| Master | Gain → `tanh` soft clip → output; a copy of the last ~2048 samples goes to a ring buffer for visualization |

**Thread interface (main → audio)**
- `engine.set(param, value)` writes targets: plain float attributes, atomic under the GIL.
- `engine.note_on(...)` / `note_off(...)` push to a bounded `deque`, drained at the start of
  each callback.
- `engine.scope()` returns a copy of the visualization buffer.

**Offline mode:** `engine.render(n_frames)` runs the same code without a stream. It is used by
tests and to write WAV files for listening checks.

**Fallback if underruns persist (measured in B.2):**
1. raise `blocksize` to 512;
2. move the engine into a separate process (`multiprocessing`). Parameters go through a
   `shared_memory` float array, note events through a `Queue`. The `AudioEngine` interface stays
   identical, so scenes don't change.

**Platform notes**
- Ubuntu needs `libportaudio2`. With PipeWire, if the default quantum adds latency, try
  `PIPEWIRE_LATENCY=256/48000`.
- macOS uses CoreAudio with no extra setup.
- The **output device** is configurable (`[audio] device`), because the event will probably use
  external speakers.

## 3. Scales (`audio/scales.py`, portable core)

- MIDI ↔ Hz, note names.
- Scale tables: major and minor pentatonic, blues, major, Dorian, Hirajoshi.
- `quantize(u ∈ [0,1], scale, root, octaves) -> midi`.
- `snap(continuous_midi, scale, strength)` for partial snapping.
- Fully unit-tested.

## 4. Mode 1 — Theremin

**Hand roles go by screen side, not by MediaPipe handedness.** The hand on the right half of the
screen is the **pitch hand**; the one on the left is the **expression hand**. This works the same
for left-handed visitors and is immune to handedness label errors.

| Input | Parameter |
|-------|-----------|
| Pitch hand: index-tip height | Pitch over 2 octaves, **snapped to scale** (snap strength configurable: 1 = always on a note, 0 = classic theremin), glide 60 ms |
| Pitch hand: horizontal position | Timbre blend, sine → saw |
| Pitch hand: fist | Mute |
| Expression hand: thumb–index opening | Volume (no expression hand → fixed 0.7) |
| Expression hand: palm roll | Filter cutoff |
| Both hands moving apart (distance) | Echo mix |
| No pitch hand | Release (fade out over 300 ms) |

**Visuals**
- Horizontal note lanes labeled with note names.
- A glowing orb at the current pitch, with a trail, colored by pitch.
- A **live oscilloscope strip** and **spectrum bars** (FFT of `engine.scope()`).

## 5. Mode 2 — Invisible Piano

- 5–8 vertical key columns (one per scale degree) in the lower part of the screen, labeled. The
  index fingertip's column is highlighted.
- **Play:** `pinch_start` inside a column → note on; `pinch_end` → note off. Both hands can play,
  up to 8 voices.
- Timbre: plucked or electric-piano-like (saw + filter envelope, or simple 2-operator FM).
- Experimental config option: **tap mode**, which triggers on a fast downward fingertip movement.
  More piano-like but less reliable; evaluate it with users.
- *Stretch:* the upper zone of each column plays that degree's triad.
- Visual: the key lights up, and notes rise as small glowing particles. This deliberately echoes
  Geometry's look.

## 6. Discover layer (Vision overlay additions)

A **"Your hand → sound"** panel shows each mapping live, as input value → output value:
- `hand height 0.62 → A4 · 440 Hz`
- `finger opening 0.35 → volume 0.40`
- `palm tilt 18° → filter 2.1 kHz`

Teaching point: a program is a set of functions turning measured numbers into new numbers. The
oscilloscope shows that sound itself is a list of numbers ("48,000 numbers per second").

## 7. Scene and navigation

`scenes/synth.py`, with dwell buttons **Theremin / Piano** and **Change scale** added to the top
strip, and a **🎵 Music** button in the main navigation. Operator keys: `4` Synth, `S` cycle scale,
`M` mute all.

## 8. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| B.1 | Engine, offline: oscillators, envelope, voices, filter, delay; WAV render; tests | Rendered WAVs sound clean (no clicks or aliasing at high notes) |
| B.2 | Live stream, underrun counter, latency measurement in HUD; stress test with Geometry rendering | 0 underruns in 10 min, or fallback (§2) implemented |
| B.3 | Scales + Theremin control mapping | New users play a recognizable melody within 1 min |
| B.4 | Theremin visuals: lanes, orb, oscilloscope, spectrum | 60 FPS maintained |
| B.5 | Invisible Piano | Two-hand playing works; no stuck notes on hand loss |
| B.6 | Discover panel, scale switching, polish | – |
| B.7 | Ubuntu laptop check (PortAudio, PipeWire), measurements, tag `v0.B` | PLAN §3 Phase B exit criteria met |

## 9. Tests

- **DSP, offline:** sine frequency via FFT peak; envelope shape; voice stealing; no NaN or Inf;
  output within [−1, 1] after the soft clip.
- **Parameter-ramp continuity:** no discontinuity larger than ε between blocks.
- **Scales:** quantize and snap tables.
- **Hand loss:** a `pinch_end` with `reason: lost` releases its note, so no stuck notes.
- **Fixture-driven:** replay `right_pinch_x5` in Piano mode → 5 note-on/off pairs.
