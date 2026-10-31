# Phase C — Physics Orchestra + Exhibit Hardening

**Outcome:**
- Geometry's shapes become musical voices: every collision is a note, so playing with physics
  composes music.
- The exhibit is hardened for a full event day, with an operations checklist.

Status: **direction spec.** Refine at the start of the phase.
Builds on: [Phase A](phaseA-magic-geometry.md) (physics, impacts), [Phase B](phaseB-airsynth.md) (audio engine, scales).

---

## 1. Orchestra scene (`scenes/orchestra.py`)

The Geometry scene plus a **SoundMapper** that turns the physics impact list (Phase A §3) into
note events for the audio engine (Phase B §2).

### 1.1 Mapping: data → sound

| Physics quantity | Sound parameter | Mapping |
|------------------|-----------------|---------|
| Shape type | Timbre | ● circle: soft sine/triangle pluck · ■ square: percussive (noise burst + low sine "thump") · ▲ triangle: FM bell · ▬ bar: marimba (sine + fast-decaying overtone) · ⬟ pentagon: filtered-saw pluck |
| Shape size | Pitch | Log-mapped over 2–3 octaves, bigger = lower, **quantized to the current scale** |
| Impact impulse | Velocity (loudness) | Log-mapped. Below `min_impulse`, ignored (resting contacts stay silent) |
| Horizontal position | Stereo pan | Linear across the screen |
| Shape–shape collision | Dyad | Both shapes sound. The second note optionally snaps to a chord tone of the first |

### 1.2 Keeping it musical, not noisy

- Per-body cooldown of 80 ms. Global limit of 16 note-ons per 100 ms. 16-voice pool with stealing.
- **Beat quantization** (toggle): delay each note-on to the next 16th note at 100 BPM (≤ 150 ms
  delay). Chaotic physics becomes rhythmic. Teaching point: rounding time values to a grid.
- *Optional:* a quiet pad drone on the scale root to glue things together.

### 1.3 New interaction: wind

An open hand moving faster than `wind_speed` applies a force to bodies within a radius along its
direction of motion. This comes on top of the Phase A paddle. A filtered-noise "whoosh" is mixed
in, with volume ∝ hand speed.

### 1.4 Custom gestures

Phase A effects work unchanged and now sound automatically: Star Rain becomes a cascade of notes,
and Big Bang a burst of chords.

### 1.5 Discover layer

On each audible impact, a fading label at the collision point, for example:
- `impact 3.2 → volume 0.8`
- `size 64 px → G3`

A small panel plots the mapping curves (impulse → volume, size → pitch) with the latest impacts
as dots. Teaching point: functions and transformations of data.

## 2. Final visitor flow

```
Attract ──(raise hand)──► Play (Orchestra) ──► 👁 See what the AI sees ──► 🧠 Train the AI ──► back to Play with your gestures
                              └──► 🎵 Music (AirSynth)
   ▲                                                                                      │
   └──────────────────────────────(no hands 20 s: reset)──────────────────────────────────┘
```

Decide in user testing whether **Play** opens Orchestra or silent Geometry. In a loud venue,
sound may not carry.

## 3. Hardening

- **Camera watchdog:** if the worker dies or delivers no frames for 2 s, restart it with backoff
  and show a discreet "reconnecting camera" badge.
- **Scene isolation:** an unhandled exception in a scene is logged, and the app returns to
  Attract instead of crashing.
- **Logging:** rotating log files in `logs/`, recording startup environment, measurements, errors,
  and counts per hour (sessions, trainings).
- **Resource stability:** no growth in bodies, particles, voices, or memory over time. Soak tests
  check this.
- **Startup self-test screen** (operator key `T`): camera FPS, inference ms, audio device and
  latency, a test tone, and model loaded.

## 4. Operations (`docs/OPERATIONS.md`, written in this phase)

- **Machine:** disable sleep and screen lock (macOS: `caffeinate -d` in the start script; Ubuntu:
  disable blanking); power connected; notifications off.
- **Camera:**
  - eye/chest level, visitor ~1–1.5 m away;
  - **no window or bright light behind the visitor**; soft light from the front (bring a ring light);
  - floor marker where visitors stand.
- **Display & audio:** external monitor/TV via `--display N --fullscreen`; external speakers set in
  `[audio] device`; volume check.
- **Venue config:** `config/venue-<name>.toml` with camera index, confidence thresholds,
  `min_hand_size`, background dimming, and audio device.
- **Start / stop:** one start command per OS (`scripts/start-exhibit.sh`). Quit by holding `Esc`.
- **Troubleshooting table:**
  - no camera → permissions/index;
  - jittery tracking → lighting;
  - bystanders tracked → raise `min_hand_size`;
  - crackling audio → blocksize;
  - and so on.
- **Recruitment materials:** a printed poster "How does this work?" with the 21-landmark diagram,
  a pipeline sketch (camera → neural network → 21 points → your code → shapes & sound), and a QR
  code to the public GitHub repo.

## 5. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| C.1 | SoundMapper: impacts → notes, timbres per shape, rate limiting | Dropping 20 shapes sounds pleasant, not like noise |
| C.2 | Beat quantization, wind, optional drone | – |
| C.3 | Data → sound discover layer | – |
| C.4 | Final flow and navigation; decide Play default | 5 new users complete the full flow unaided |
| C.5 | Hardening: watchdog, scene isolation, logging, self-test screen | Pulling the camera's USB and plugging it back recovers automatically |
| C.6 | Soak tests: 2 h replay loop (automated) + 2 h live, on **both** machines | No crash, 0 underruns, flat memory |
| C.7 | OPERATIONS.md, poster, dress rehearsal at a venue-like spot, tag `v1.0` | PLAN §3 Phase C exit criteria met |
