# Phase B — AirSynth

**Outcome:** hands become a musical instrument, in two scenes: **Theremin** (continuous) and
**Invisible Piano** (discrete notes). A real-time audio engine gets built here and is reused by
Phase C.

Status: **direction spec, refined for the Theremin (2026-11-04).** §2, §4, §8 and §9 now follow
the Phase 0/A measurements; the Piano (§5) is still direction only and gets refined at B.6.
Builds on: [Phase 0](phase0-foundation.md), [architecture](architecture.md).

**Order of work:** the one real unknown is whether a Python audio callback keeps up while
MediaPipe, pymunk and the 60 FPS render compete for the GIL. A throwaway spike (B.0) answers that
on both machines before any DSP is built. The Theremin comes first and needs one voice; the voice
pool and envelopes come with the Piano.

---

## 1. Principles

- **Always sounds good:** default to pentatonic scales and pull pitch towards the scale's notes,
  so random movement still sounds musical.
- **Audio never waits on vision:** synthesis runs in the audio callback. The main thread only
  sets targets and queues note events.
- **Latency is a feature:** measure it, show it in the HUD, and choose interactions that tolerate it.
  Continuous control (theremin) tolerates ~100 ms. Percussive strikes don't, so **Air Drums are
  out of scope** unless measured latency turns out much lower than expected.

## 2. Audio engine (`audio/engine.py`, `audio/voices.py`, `audio/dsp.py`)

**Stream:** `sounddevice.OutputStream(samplerate=48000, channels=2, dtype="float32",
blocksize=512, latency="low")`. 512 samples is about 10.7 ms per block. Configurable; B.0
measures 256, 512 and 1024 and sets the default.

*Latency budget:* camera-to-screen latency is already 34 ms (MacBook) and 46 ms (Ubuntu) mean
([measurements](../measurements.md)). Under the 100 ms exit criterion that leaves room for a
1024-sample block (21 ms) plus the device buffer. The Theremin tolerates that; only the Piano's
note onsets would notice, so prefer the larger block over underruns.

**Stream lifetime:** the app opens the stream once at startup and keeps it open. Opening it on
scene enter can click and takes 100 ms or more. Scenes that don't use audio leave it idle; the
idle callback writes zeros and returns. If the device can't open (no device, CI), the engine is a
no-op, like `Sfx`: sound never stops the exhibit.

**GIL:** the callback is Python, so it must get the GIL every block. Python's default thread
switch interval is 5 ms (`sys.getswitchinterval()`), about one whole 256-sample block, so the
audio thread can wait that long behind the main thread. B.0 measures with and without
`sys.setswitchinterval(0.001)`; if it helps, the app sets it at startup.

**Callback contract:**
- Vectorized NumPy only: no locks, no logging, no allocation-heavy work.
- Must finish well within the block duration.
- Underflow status flags are counted and shown in the HUD, with the callback's time (mean / max)
  and the largest gap between callbacks.

**Building blocks** (the Theremin needs the first three rows plus delay and master; the rest come
with the Piano)

| Component | Implementation |
|-----------|----------------|
| Parameter smoothing | Targets arrive at the camera rate (30 Hz). Pitch is smoothed in **semitones** with a one-pole glide (time constant from config, ~60 ms), so a glide sounds even across the range and the 30 Hz steps don't show. Gain and other parameters: per block, a linear ramp from the current value to the target (`np.linspace`), so there is no zipper noise |
| Oscillators | Phase accumulator via `np.cumsum` of per-sample increments, so glide is free. Shapes: sine, triangle, **polyBLEP** saw (anti-aliased), noise |
| Mono voice | Theremin: one oscillator pair (sine + saw, blended), gain, no envelope; silence is gain 0 |
| Envelope | *Piano:* ADSR, evaluated per block, vectorized per segment |
| Voices | *Piano:* fixed pool of 8 (16 in Phase C), voice stealing (oldest released first, then quietest) |
| Filter | Biquad low-pass via `scipy.signal.lfilter` with persistent `zi` state. Coefficients are updated per block from the smoothed cutoff. A recursive filter can't be vectorized in plain NumPy, so **`scipy` becomes an explicit dependency** in B.1 (it is not installed today). Until then the Theremin's brightness comes from the sine → saw blend |
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

**Fallback if underruns persist (measured in B.0, confirmed in B.2):**
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
- `scale_notes(scale, root, octaves) -> tuple[int, ...]`: the marks, root to root.
- `note_position(u ∈ [0,1], n) -> p` (band height → continuous note position, §4.2).
- `magnet(p, m) -> p'` and `position_to_midi(p, notes) -> midi` (interpolation in semitones).
- `nearest(midi, notes) -> (note, cents)` for the mark highlight and the Discover panel.
- Fully unit-tested.

## 4. Theremin scene (`scenes/theremin.py`)

A dedicated scene (operator key `4`), with nothing else on screen to compete with the instrument.
The Piano gets its own scene later (§5).

### 4.1 Zones and hand roles

**Hand roles go by screen side, not by MediaPipe handedness.** The screen is split into two
zones by a faint vertical divider:

- **Right half: the pitch zone.** The hand here is the **pitch hand**.
- **Left half: the volume zone.** The hand here is the **volume hand**.

This works the same for left-handed visitors and is immune to handedness label errors. With one
hand in view, it is the pitch hand wherever it is, so a visitor who tries one hand first hears
sound at once.

The **active band** is the middle 15–85 % of the screen height in both zones. Tracking is worse
near the edges, and arms get tired at the top. It is drawn with a clear border; the areas above
and below it are dimmed. Outside the band, the value clamps to the nearest edge.

**v1 controls (B.3): three, all discoverable without instructions.** A visitor has about 30
seconds and no manual; six mappings across two hands is too many to find.

| Input | Parameter |
|-------|-----------|
| Pitch hand: index-tip height | Pitch over 2 octaves (§4.2), glide 60 ms |
| Volume hand: palm height | Volume, the way a real theremin works (higher = louder). No volume hand → fixed 0.7 |
| Pitch hand: horizontal position within the pitch zone | Timbre blend: sine at the divider → saw at the outer edge. Never changes the pitch. Shown by the orb's look (§4.3); dropped to a B.5 candidate if the B.3 tests show it confuses visitors |
| No pitch hand | Release (fade out over 300 ms) |

**Candidates after the B.3 user tests (B.5),** added only if visitors have mastered the three
above. The features already exist in `gestures/features.py` (`pinch_ratio`, `palm_roll`).

| Input | Parameter |
|-------|-----------|
| Volume hand: palm roll | Filter cutoff (needs the SciPy filter) |
| Both hands moving apart (distance) | Echo mix |
| Pitch hand: fist | Mute |
| Volume hand: thumb–index opening | Volume, instead of palm height, if tests favour it |

### 4.2 Pitch mapping: equal spacing per note, continuous between notes

The scale's notes over the 2 octaves (11 for a pentatonic scale, root to root two octaves up)
get **equal heights** in the band, like evenly spaced frets. Each note's **mark** sits at the
centre of its zone. The pitch moves **continuously** between marks, so sliding the finger glides
from one note to the next, as on a fretless instrument.

1. **Height → note position.** Index-tip height in the band maps linearly to a continuous note
   position `p ∈ [−0.5, N − 0.5]`, clamped to `[0, N − 1]`. Integer `p` is a note's mark; with
   the half-zone padding, the lowest and highest notes get as much room as the others.
   (`N` = 11, band 70 % of the screen: one note every ~6.4 % of the height, ~46 px at 720p.)
2. **Slow and fast parts.** On the main thread, at the camera rate, `p` is split into a slow part
   (one-pole low-pass, ~150 ms) and a fast part (`p` minus the slow part): the hand's natural
   shake.
3. **Magnet.** The slow part is pulled towards the nearest mark by a continuous curve, never a
   step. With `k = floor(p)` and `x = p − k`:
   `p' = k + x − m · sin(2πx) / 2π`.
   `m` (`magnet`, 0..1, default 0.8) is how sticky the marks are: at 0 the mapping is linear
   (classic theremin); at 1 the curve is flat at each mark (the pitch rests on the note while the
   finger is near it) and steepest halfway between. It is continuous and monotonic for every `m`
   in 0..1, so there are no jumps and no flicker between notes, and it needs no hysteresis.
4. **Vibrato.** The fast part is added back, scaled by `vibrato_keep` (~0.5). Near a mark the
   magnet flattens the slow part, so this is what keeps a shaking hand sounding alive.
5. **Note position → pitch.** Linear interpolation in semitones between the two neighbouring
   scale notes. A 3-semitone step (A → C) covers the same height as a 2-semitone step (C → D),
   so the glide is a little faster across the wider steps; that is the cost of even spacing.
6. The resulting MIDI target goes to the engine, which glides to it in semitones (§2).

`magnet` and `vibrato_keep` are in config; the operator can set `magnet = 0` for a fretless
classic theremin. **Re-entry:** when the pitch hand returns after a release, the pitch **jumps**
to the new target while the gain is still 0, then fades in. Gliding from the last note sounds
like a siren.

### 4.3 Visuals and cues

**Pitch zone**
- **Note marks:** one horizontal line per scale note across the pitch zone, labeled with its
  note name (A3, C4, D4, …) at the outer edge. The **root notes** (A3, A4, A5) are drawn thicker
  and brighter, like the double dots at a guitar's 12th fret.
- **Current note:** the mark nearest the current pitch lights up, more brightly the closer the
  pitch is to it; a full glow means "on the note". The pitch's exact position is the orb's height,
  so a visitor sees where the finger is relative to the marks.
- **Orb:** a glowing orb at the index tip, with a trail, colored by pitch. Its look follows the
  timbre: round and soft for sine, sharper and with a jagged halo for saw.

**Volume zone**
- **Volume bar:** a vertical bar the height of the band that fills from min (bottom) to max (top)
  as the palm rises, with a marker at the palm's height so hand and bar line up. The fill pulses
  slightly with the actual output level, so the bar shows the sound itself.
- No volume hand: the bar sits at the 0.7 default, dimmed.

**Responses and hints**
- A zone lights up faintly while a hand is in it; the divider makes clear that the side decides
  the role.
- A hand outside the band dims its cursor, and the hint line says where to go.
- Onboarding hints, one at a time, each gone once done (booth shell): "Raise or lower your
  finger on the right to change the note" → "Use your other hand on the left for volume" →
  "Move your right hand sideways to change the sound" (only if timbre stays in v1).

**Sound view:** a **live oscilloscope strip** and **spectrum bars** (FFT of `engine.scope()`)
along the bottom edge, below the band.

## 5. Invisible Piano scene (`scenes/piano.py`, direction only)

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
- `finger height 0.62 → A4 · 440 Hz (+12 cents)`
- `other hand height 0.40 → volume 0.40`
- `palm tilt 18° → filter 2.1 kHz` (once palm roll is mapped, B.5)

Teaching point: a program is a set of functions turning measured numbers into new numbers. The
oscilloscope shows that sound itself is a list of numbers ("48,000 numbers per second").

## 7. Scene and navigation

Two scenes, following the booth-minimum shell (PLAN D14: staff switch scenes, dwell buttons are
parked): `scenes/theremin.py` on operator key `4` and, later, `scenes/piano.py` on `5`. Operator
keys inside them: `S` cycle scale (the note marks and labels follow), `M` mute all.

## 8. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| B.0 | **Spike** (throwaway, `--audio-spike`): live sine follows the pitch hand, in any scene; HUD shows underruns, callback time and gaps. 10 min in Geometry with many shapes, on the MacBook and the Ubuntu laptop, at 256 / 512 / 1024 samples, with and without a 1 ms switch interval | Block size and switch interval chosen from measurements; fallback (§2) needed or not. Results in `docs/measurements.md` |
| B.1 | Engine, offline: mono voice (oscillators, semitone glide, gain ramps), delay, master; `scales.py`; WAV render; tests; `scipy` dependency + filter | Rendered WAVs sound clean (no clicks or aliasing at high notes) |
| B.2 | Live stream owned by the app (replaces the spike), underrun counter and latency in HUD; 10-min stress test with Geometry rendering | 0 underruns in 10 min, or fallback (§2) implemented |
| B.3 | Theremin scene (§4): zones, band, note marks, volume bar, v1 controls, pitch mapping (§4.2), hints; user tests | New users play a recognizable melody within 1 min |
| B.4 | Theremin polish: orb with trail and timbre look, mark glow, oscilloscope, spectrum | 60 FPS maintained |
| B.5 | Theremin extra controls from the candidates (§4), only those the B.3 tests justify | – |
| B.6 | Invisible Piano: envelope, voice pool; refine §5 first | Two-hand playing works; no stuck notes on hand loss |
| B.7 | Discover panel, scale switching, polish | – |
| B.8 | Ubuntu laptop check (PortAudio, PipeWire), measurements, tag `v0.B` | PLAN §3 Phase B exit criteria met |

## 9. Tests

- **DSP, offline:** sine frequency via FFT peak; envelope shape; voice stealing; no NaN or Inf;
  output within [−1, 1] after the soft clip.
- **Parameter-ramp continuity:** no discontinuity larger than ε between blocks.
- **Scales:** note tables; `note_position` puts each mark at the centre of an equal zone, with
  the end notes' zones as large as the others; `position_to_midi` hits every note exactly at its
  mark and is linear in semitones between; `magnet` is continuous and monotonic for `m` in 0..1,
  fixes every mark, and is flat at the marks for `m = 1`; the fast part is added back scaled by
  `vibrato_keep`.
- **Pitch hand x:** moving the pitch hand sideways changes the timbre target and never the pitch
  target.
- **Theremin re-entry:** pitch hand lost, then found an octave away → no glide (the pitch jumps
  while the gain is 0).
- **Fixture WAVs:** record `theremin_sweep` (slow up and down) and `theremin_hand_loss`; a test
  replays each through tracker → mapping → offline `render()` and checks for clicks (no sample
  step above ε), NaN and stuck sound. The WAVs are also the listening check.
- **Hand loss:** a `pinch_end` with `reason: lost` releases its note, so no stuck notes.
- **Fixture-driven:** replay `right_pinch_x5` in Piano mode → 5 note-on/off pairs.
