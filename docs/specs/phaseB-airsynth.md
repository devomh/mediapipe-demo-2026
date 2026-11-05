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
blocksize=512, latency="low")`. 512 samples is about 10.7 ms per block. Configurable. *B.0
(Ubuntu desktop):* in its own process both 256 and 512 run with 0 underruns live, and PipeWire
reports 10.7 ms output latency for both, so 512 keeps more headroom for free. The MacBook run
decides whether 256 saves latency on CoreAudio.

*Latency budget:* camera-to-screen latency is already 34 ms (MacBook) and 46 ms (Ubuntu) mean
([measurements](../measurements.md)). Under the 100 ms exit criterion that leaves room for a
1024-sample block (21 ms) plus the device buffer. The Theremin tolerates that; only the Piano's
note onsets would notice, so prefer the larger block over underruns.

**Lifetime: the app owns the engine, scenes use it.** The app starts the audio process once at
startup and keeps it running in every scene; it goes in `AppContext.audio`, next to `sfx`.
Starting it on scene enter would delay the first note and can click: the process takes ~0.1 s to
start (0.46 s cold) and ~80 ms to stop on the Ubuntu desktop, and `spawn` is slower on macOS.
Scenes that don't use audio leave it idle; the idle callback writes zeros and returns (about 94
callbacks a second at 512 samples, negligible). Phase C's Orchestra uses the same engine.

- **Non-blocking start:** the app starts the process and carries on; the window opens at once.
  The proxy becomes ready when the child reports its stream is running; until then `set` is a
  no-op and the HUD shows "audio starting". (`ProcessSpike.open()` in the spike waits instead.)
- **Shutdown:** the app's `finally` stops the child (run flag, join, then terminate), on every
  exit path, including the error screens. The child also stops when its parent goes away.
- If the device can't open (no device), or `[audio] enabled = false` (tests, CI), the engine is a
  no-op, like `Sfx`: sound never stops the exhibit.

**The engine runs in its own process** (decided in B.0, [measurements](../measurements.md)).
The callback is Python, so in the app's process it must win the GIL from the main thread every
block. Live with two hands, the main thread's work (tracking, gestures, Geometry, drawing) made
it crackle: the callback lost the GIL mid-block and waited up to a switch interval (5 ms) to get
it back. MediaPipe is not the cause; it releases the GIL during inference. A shorter switch
interval only narrowed the window. In a spawned child process (`multiprocessing`, `spawn` on both
platforms: never fork a process running MediaPipe's threads) the callback has its own GIL: 0
underruns live, worst callback under 4 % of a block. The child imports only NumPy, sounddevice
and the engine modules, and stops when the parent goes away.

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

**Process interface (app ↔ audio process)**: `AudioEngine` in the app is a thin proxy; scenes
never see the process.
- `engine.set(param, value)` writes targets into a shared float array (`RawArray` or
  `shared_memory`), one slot per parameter. One writer per slot, so no locks; the callback reads
  the targets at the start of each block.
- `engine.note_on(...)` / `note_off(...)` go through a `multiprocessing.Queue`, drained without
  blocking at the start of each callback.
- `engine.scope()` returns a copy of a shared ring buffer of the last ~2048 output samples. The
  child writes it each block; a torn read only affects one drawn frame of the oscilloscope.
- `engine.state()`: what the sound is actually doing, for the visuals (§4.3), written by the
  child each block into the shared array: the **current pitch** after the glide (the orb and the
  glowing mark), the **output level** (RMS of the block, for the volume bar's pulse) and the
  current gain. Scenes read it in `draw()` at 60 Hz; a torn read affects one drawn frame only, so
  there are no locks.
- `engine.stats()`: underflows, callback time, gaps and output latency, written by the child into
  the shared array for the HUD.
- If the child fails to start or dies, the proxy logs once and becomes a no-op (sound never stops
  the exhibit); the HUD shows "audio off".

**Offline mode:** the synthesis core (`render(n_frames)`) runs the same code in-process without
a stream or a child. It is used by tests and to write WAV files for listening checks.

**If underruns still appear (B.2 stress test):** raise `blocksize` to 1024; the latency budget
above allows it.

**Config** (`[audio]`): `enabled`, `device`, `blocksize`, `master_volume`. Like every section,
it has its dataclass in `config.py` (unknown keys are an error).

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
The Piano gets its own scene later (§5). The scene doesn't own the audio: it uses the app's
engine from `AppContext.audio` (§2).

### 4.0 Scene lifecycle and code layout

- **`enter()`:** load the scale and band from `[theremin]`, hide the Vision overlay
  (`show_vision = False`, as in Geometry and Writing; `V` still toggles it) and dim the camera
  background (`[theremin] background_brightness`).
- **`exit()`:** fade the gain to 0 over the release time (300 ms) rather than cutting it, so
  switching scenes mid-note neither clicks nor leaves a drone. Other scenes never touch the
  engine, so it stays silent there.
- **`idle_reset()`:** the sound has already faded (no pitch hand); reset the hints and the scale
  to the defaults.
- **Camera stall:** `handle()` runs only when a camera frame arrives. If none has arrived for
  ~200 ms (camera error, USB dropout), `update(dt)` fades the gain to 0, so the last note doesn't
  hold forever.
- **Code layout:** the pure maths (note positions, magnet, interpolation, slow/fast split) is in
  `audio/scales.py` and a small mapping module with no pygame, so it is unit-tested alone. The
  scene reads `HandState`s in `handle()` and calls `engine.set(...)`. The slow/fast split runs on
  camera time (`t_capture`), not render `dt`, so it is the same at any frame rate.
- **Config** (`[theremin]`): band, `magnet`, `vibrato_keep`, glide, scale, root, octaves,
  default volume, release time, background brightness.

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
parked): `scenes/theremin.py` on operator key `4` (added to `SCENE_KEYS`) and, later,
`scenes/piano.py` on `5`.

- **`S` cycle scale:** the note marks and labels follow. Available only in scenes that have a
  scale, like `T` (collect) is only in Writing.
- **`M` mute all:** today it toggles only the sound effects (`Sfx`); from B.2 it mutes the engine
  too, and the keys overlay shows one state for both.
- **HUD:** the engine's stats line (underruns, callback time, output latency) is app-level and
  shows in every scene, because Phase C uses the engine in other scenes too.
- **Two audio outputs:** Geometry's effects stay on `pygame.mixer`, so the app has two outputs
  open. `[audio] device` picks the engine's output, but the mixer uses the system default, so at
  the venue the **system default output must be the external speakers** (ops checklist).

## 8. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| B.0 | **Spike** (throwaway, `--audio-spike`): live sine follows the pitch hand, in any scene; HUD shows underruns, callback time and gaps. Live in Geometry with two hands, on the Ubuntu desktop and the MacBook | *Ubuntu desktop done:* in-process crackles with two hands (GIL); a separate process (`--spike-process`) gives 0 underruns at 256 and 512. MacBook run pending (block size). Results in `docs/measurements.md` |
| B.1 | Engine, offline: mono voice (oscillators, semitone glide, gain ramps), delay, master; `scales.py`; WAV render; tests; `scipy` dependency + filter | Rendered WAVs sound clean (no clicks or aliasing at high notes) |
| B.2 | Engine in its own process, owned by the app (`AppContext.audio`, non-blocking start, `[audio]` config, `M` mutes all), underrun counter and latency in HUD; 10-min stress test with Geometry rendering. Removes the spike: `--audio-spike`, `--spike-gain`, `--spike-process`, `--switch-interval` and the mapping in `App._process`; its process code (spawn, shared array, stop with the parent) moves into the engine | 0 underruns in 10 min |
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
- **Stub engine:** `App` tests run with `[audio] enabled = false` (no child process, no audio
  device in CI); scene tests use a stub engine in `AppContext` that records the targets it gets.
  Only one test starts the real child process (start, set, stats, stop; and a device that can't
  open).
- **Scene lifecycle (stub):** `exit()` mid-note sends gain 0 with the release ramp; no camera
  frame for 200 ms → gain 0; `idle_reset()` restores the default scale; sideways movement changes
  only the timbre target.
- **Hand loss:** a `pinch_end` with `reason: lost` releases its note, so no stuck notes.
- **Fixture-driven:** replay `right_pinch_x5` in Piano mode → 5 note-on/off pairs.
