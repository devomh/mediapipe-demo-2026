# Phase A — Magic Geometry + Air Writing

**Outcome:** the booth exhibit. Visitors:
- create, throw, push, and grab glowing shapes that fall, collide, and **make sounds**;
- reveal what the AI sees;
- **write a character in the air and watch a neural network read it** ([Air Writing spec](phaseA-air-writing.md)).

It runs with staff at the table: a booth-minimum shell keeps every scene tidy between visitors.

Builds on: [Phase 0](phase0-foundation.md), [architecture](architecture.md).

> **Changed 2026-11-01 (PLAN D11, D14).** The event is a booth with people passing by, so demos
> must pay off within seconds. **Train-a-Gesture (§5)** and the **attract screen and dwell
> buttons (§6)** are parked, not deleted. Sound effects (§4) and the booth-minimum shell (§6) are
> new. Air Writing has its own spec. Milestones in §7 are renumbered from A.5.

---

## 1. Design principles

1. **One hand is enough.** Every core action works with a single hand. Two-hand actions are a bonus.
2. **Few, reliable gestures.** Pinch, open hand, and fist are the whole basic vocabulary. Each is
   robust thanks to Phase 0 hysteresis. (The brainstorm's thumb+middle pinch is dropped: at
   webcam distance it is too easily confused with thumb+index.)
3. **Instant feedback.** Every recognized gesture has a visible response within the same frame:
   a cursor change, a preview, or a particle burst.
4. **Teach without words where possible.** Hints appear only until the action has been performed once.

## 2. Interaction vocabulary (Magic Geometry scene)

| Gesture | Action | Feedback |
|---------|--------|----------|
| **Pinch & hold** (thumb + index) | Spawns a shape at the pinch point. While held, the shape **inflates** (min → max over ~1.5 s) and follows the pinch | Pulsing translucent preview, size ring |
| **Release pinch** | Shape becomes a physics body and **inherits release velocity** × `throw_gain`, so moving while releasing throws it | Small sparkle at release |
| **Other hand: hold up N fingers** | Chooses the shape type for the next or current pinch by **number of corners**: 1 ● circle · 2 ▬ bar · 3 ▲ triangle · 4 ■ square · 5 ⬟ pentagon | Shape palette at the screen edge highlights the choice |
| *(no other hand visible)* | Shape type cycles automatically on each spawn | Same palette shows "next" |
| **Two-hand pinch** (second hand pinches while the first holds a shape) | **Stretch:** the shape spans between the two pinch points. Size = distance, rotation = angle | Both pinch points connected by a glowing line |
| **Open hand** (not pinching, not fist) | The hand is a **kinematic paddle** (circle at palm centre, radius ∝ hand size) that pushes and bats shapes | Soft ring cursor |
| **Fist near a shape** | **Grab:** shape attaches to the hand via a pivot joint. Opening the hand releases or throws it | Shape outline brightens, grab ring |
| **Pinch on a shape** *(added in A.3)* | **Grab** by the pinch point; releasing the pinch drops or throws it. A pinch in empty space still creates a shape | Same as fist grab |
| **Both fists held 1.5 s** | **Clear:** every shape bursts into particles | Progress ring around both fists |
| **Custom gesture** (from Train scene) *(parked)* | Triggers its slot's effect (§5.6) | Slot icon flashes |

*Changed in A.3, from live testing and `recordings/grab_attempts`:*
- **Natural grabs look like pinches.** A visitor grabbing a shape curls the fingers with the
  thumb on the index, which is geometrically a pinch (MediaPipe labelled it `None`, not
  `Closed_Fist`). No hand-shape rule separates the two, because deliberate pinches are also made
  with curled fingers (`two_hand_stretch`). What is under the hand does: a pinch within
  `pinch_grab_reach` (0.06 window heights) of a shape grabs it. Claw grab measured 0.039;
  deliberate creation pinches started ≥ 0.35 from any shape. Stretch takes priority.
- **The paddle needs a deliberately open hand held for 0.4 s** (MediaPipe's debounced
  `Open_Palm`, ring fading in while it arms), and not within 0.4 s of ending a pinch. "Not
  pinching, not fist" made every half-open hand a paddle, including right after a pinch, where
  it batted away the new shape. Pinch, release, and grab stay instant: creating and throwing
  depend on timing, and the unwanted activations in the recording lasted > 1 s anyway.
- Fist reach is 0.09 window heights (a natural fist grab landed at 0.064).

## 3. Physics (`physics/world.py`)

- `pymunk.Space` in screen pixels, y down. **Gravity = `1.6 × window_height` px/s²**, so behavior
  doesn't depend on resolution.
- Static floor and left/right walls (segments). The ceiling is open, so thrown shapes come back down.
  *Added in A.1:* throws are capped at `max_speed` (4 window heights/s) and may rise at most
  `max_rise` (1 window height) above the release point. Uncapped, a hard upward flick flew ~7
  window heights up, stayed off screen for ~6 s, and with any sideways motion sailed over a wall.
- Fixed timestep 1/120 s using an accumulator, with at most 4 substeps per frame (no spiral of death).
- Materials: elasticity 0.6, friction 0.6. Density varies slightly by type, so bars feel heavier
  than circles. Sleeping is enabled.
- Shapes are `Circle`, regular `Poly` (n = 3, 4, 5), and bar (`Poly` box with a 4:1 aspect).
  Concave visuals such as stars use a convex hull for physics and a sprite for drawing.
- **Cap:** 40 dynamic bodies. Spawning the 41st fades out the oldest over 0.5 s, then removes it.
- **Hand bodies:** kinematic bodies whose velocity is set each frame from tracked motion
  (`(target − pos) / dt`), so collisions transfer momentum correctly. They are removed when the
  hand leaves or changes state.
  *Added in A.3:* `dt` is at least `hand_follow_s` (one camera frame), so motion stays smooth
  between 30 FPS camera updates at 60 FPS rendering. A grabbed piece is **thrown with the fastest
  velocity from the last 0.15 s**: `fist_end` arrives 70–130 ms after the fingers start to open
  (MediaPipe label lag plus debounce, measured on `fist_palm_cycle`), when a throwing hand is
  already slowing down.
- A collision handler records **impacts** (impulse magnitude, position, both shapes) into a
  per-step list. Phase A uses it for sparks; Phase C turns it into sound.

## 4. Visuals (`render/effects.py`, scene drawing)

- **Background:** camera at about 25 % brightness (config), or a dark gradient. `B` toggles.
- **Color:** a curated palette of 8 saturated hues that read well on dark backgrounds. Each spawn
  takes a random hue different from the previous one.
- **Glow:** pre-rendered radial-gradient sprites, cached per (hue, size bucket), drawn additively
  (`BLEND_ADD`) behind each shape. On top: a filled shape and a lighter outline.
- **Particles:** a pooled system with NumPy arrays for position, velocity, and life (vectorized
  update), drawn with `Surface.fblits` using small sprites. Cap 1,500. Emitted on impacts above a
  threshold, on clear, and on custom effects.
- **Cursors:** open-hand ring, pinch dot, fist grab ring, in the hand's track color.
- **Stretch goals (Level 3 from the brainstorm):** motion trails, shapes splitting on hard
  impacts, squares shattering into particles on the floor.

### 4.1 Sound effects (`audio/sfx.py`) — *added 2026-11-01, PLAN D13*

- `pygame.mixer` at 48 kHz stereo with a 512-sample buffer (~11 ms), 16 channels. Opened once by
  the app; scenes get an `Sfx` object through `AppContext`.
- **Sounds are synthesized with NumPy at startup** (fixed seed, < 100 ms total) and cached as
  `pygame.mixer.Sound`. No sample files.
- Volume and stereo pan are set per play with `Channel.set_volume(left, right)`; pan follows the
  event's x position.

| Event | Sound | Varies with |
|-------|-------|-------------|
| Spawn (pinch in empty space) | Soft "pop": short sine blip with a falling pitch | – |
| Inflate (while held) | Quiet rising tone that follows the size, on one reserved channel; stops on release | Size |
| Throw (release faster than `throw_min_speed`) | "Whoosh": band-passed noise burst | Release speed → volume |
| Impact | Short struck tone. **Timbre from the shape** (circle: sine; triangle: bright, with harmonics; square and pentagon: woody, noise attack; bar: low knock) | **Pitch from size** (bigger = lower, 5 size buckets); **volume from impulse** |
| Grab (fist or pinch) | Soft click | – |
| Clear (two fists) | Falling sweep plus a noise burst | – |

The bank is 5 shape types × 5 size buckets of impact sounds plus the five others: about 30
sounds.

**Impact limiter** (the 40-shape pile must not buzz):
- ignore impulses below `min_impulse`;
- at most `max_impacts_per_frame` (3) per frame, loudest first;
- a per-piece cooldown of `piece_cooldown_s` (0.08 s), so shapes resting on each other stay quiet.

**Robustness:** if the mixer cannot open (no audio device, CI), log one warning and run silently.
Sound never crashes the exhibit. Tests run with `SDL_AUDIODRIVER=dummy`.

**Config `[sound]`:** `enabled`, `volume`, `min_impulse`, `max_impacts_per_frame`,
`piece_cooldown_s`, `throw_min_speed`. Operator key **`M`** toggles mute; the state shows in the
keys overlay.

*Relation to Phase C:* Physics Orchestra later makes collisions musical through the Phase B synth
engine. These effects are a simpler precursor and may be replaced then.

## 5. Train-a-Gesture scene (`scenes/train.py`, `ml/`) — *parked (PLAN D11)*

> Parked on 2026-11-01: its ~60 s guided flow is too long for a booth. Air Writing is Phase A's
> ML mode instead. This section is kept as written so it can be picked up later.

### 5.1 Concept

Visitors at an exhibit can't type, so they **invent poses for three predefined effect slots**:

| Slot | Effect in Geometry |
|------|--------------------|
| ⭐ Star Rain | 12 small stars fall from the top over 1 s |
| 🔄 Gravity Flip | gravity reverses for 4 s |
| 💥 Big Bang | radial impulse from the hand + particle burst |

There is also a fourth class, **"Anything else"**. Teaching point: *a model also needs examples of
what is NOT the gesture.*

### 5.2 Flow (target under 60 s)

1. **Intro card:** "Teach the computer 3 new hand signs." Dwell button **Start**.
2. **For each slot:** icon and name → "Make your pose and hold it — move your hand a little" →
   3-2-1 countdown → **capture 40 frames** (~1.3 s) with a progress ring. Frames without a hand
   are skipped. The "move a little" instruction is deliberate: it produces varied examples,
   which is the data-augmentation teaching moment.
3. **"Anything else":** capture 60 frames while "Wave your hand around, make random shapes."
4. **Train:** instant. Show "Learning from 180 examples × 60 numbers…" as a ~1 s animation for
   drama, then the PCA map (§5.5).
5. **Test:** live prediction with confidence bars per class, and the live hand as a moving dot
   on the PCA map. A slot counts as **recognized** when its probability is ≥ 0.7 for 5 frames;
   its icon then lights up.
6. Dwell buttons **Try again** or **Use them in Magic Geometry**.

### 5.3 Features (`ml/features.py`, portable core)

- Use world landmarks. Subtract the wrist, divide by `palm_length`, mirror x for left hands
  (so a pose trained with one hand works with the other), and drop the all-zero wrist row
  → **60 numbers**.
- Invariant to position and distance from the camera. **Not** invariant to rotation, on purpose:
  "thumb up" and "thumb down" must differ.
- Tests: translation, scale, and mirror invariance on synthetic hands.

### 5.4 Model (`ml/classifier.py`)

- `KNeighborsClassifier(n_neighbors=5, weights="distance")` from scikit-learn, wrapped so the rest
  of the app sees `fit(X, y)` and `predict_proba(x) -> dict[class, p]`.
- **Unknown rejection:** if the mean distance to the k neighbors exceeds `reject_factor` × (the
  median nearest-neighbor distance within the training data), predict *unknown*. Default factor
  is 2.5, tuned with fixtures.
- **Overlap hint:** after training, measure for each pair of classes how many of one class's
  nearest neighbors belong to the other. Above 20 % → hint "⭐ and 🔄 look alike to the computer;
  try making them more different!" (Teaching point: separable classes.)

### 5.5 Visualizing the data (`ml/pca.py` portable, drawing in the scene)

- **PCA map:** 2-D projection computed with NumPy SVD on the training set. Points are colored per
  class, with the class icon at each centroid. The live hand is a bright dot with a short trail.
  Visitors literally see their hand moving between their own clusters.
- **Data panel:** a compact "spreadsheet" filling up row by row during capture (rows = examples,
  60 colored cells per row as a heatmap). It shows that *data is just numbers*.

### 5.6 Integration with Geometry

- `CustomGestureDetector` (in `gestures/`, portable) takes a `predict_proba` callable from
  `AppContext`. It emits `custom_start` with `data={"slot": "star_rain", "confidence": p}` using
  the same threshold and debounce as the test step, then `custom_end`. Each slot has a 2 s
  cooldown.
- Geometry maps slots to effects. Without a trained model, the detector is inactive.

### 5.7 Privacy

Training data lives **in memory only**. It is cleared when the attract screen returns (a new
visitor). The operator flag `--save-datasets DIR` writes the landmark datasets (never images)
for later classroom use. It is off by default.

## 6. Exhibit shell

### 6.0 Booth-minimum shell — *the part built now (PLAN D14)*

| Piece | Behavior |
|-------|----------|
| **Idle reset** | No hands for `idle_timeout` (20 s) → the current scene resets (Geometry: shapes fade out; Air Writing: ink and text line clear). The app stays in that scene; there is no attract screen |
| **Hint line** | One short line at the bottom of each scene, contextual as in "Onboarding hints" below: each hint disappears once its action has been done and comes back after an idle reset |
| **Operator keys** | `1` Vision · `2` Geometry · `3` Air Writing · `M` mute · `V` vision overlay · `H` HUD · `B` background · `C` clear · `F` fullscreen · `K` keys list |
| **Quit** | **Hold `Esc` 2 s** (today `Esc` quits at once), so a curious visitor can't close the app |
| **Fullscreen** | `--fullscreen` and `F` already exist; check both machines at the event resolution |

Below, the **attract scene and dwell buttons are parked**. The onboarding hints are used by the
hint line.

**Attract scene** (`scenes/attract.py`) *(parked)*
- Shapes fall in the background, with large text:
  *"Your hands are the controller. No keyboard. No mouse. Just artificial intelligence.
  Raise your hand to begin."*
- A hand tracked stably for 0.5 s → transition to Geometry with onboarding.
- No hands for `idle_timeout` (20 s) in any scene → fade back to Attract and reset state
  (shapes, trained model).

**Onboarding hints** are contextual. Each disappears once performed and is skipped if already done:
1. "Pinch your thumb and index finger to create a shape."
2. "Let go to drop it — move while letting go to throw!"
3. "Hold up fingers with your other hand to choose a shape: 1 ● 3 ▲ 4 ■ 5 ⬟"
4. "Make a fist on a shape to grab it."
5. "Push shapes with your open hand."

**Touchless navigation: dwell buttons** *(parked)*. A small strip of buttons along the **top edge**, which a
hand rarely reaches by accident. Hover the palm cursor for 1.2 s; a fill ring shows progress.
Buttons:
- **👁 See what the AI sees** (toggles the Vision overlay from Phase 0)
- **🧠 Train the AI**
- **✨ Play** (Geometry)

**Operator controls (keyboard)**
- `1` Attract · `2` Geometry · `3` Train · `V` vision · `H` HUD · `B` background · `C` clear ·
  `F` fullscreen.
- **Hold `Esc` 2 s** to quit, so a curious visitor tapping the keyboard can't close the app.

## 7. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| A.1 ✓ | Physics world; pinch → spawn, inflate, drop, throw; walls; body cap | Shapes fall, collide, and throw convincingly |
| A.2 ✓ | Shape choice by finger count; two-hand stretch | Choice is reliable for 1/3/4/5 fingers in the fixture test |
| A.3 ✓ | Hand paddle, fist grab, two-fist clear | Each action works for 2 new users without explanation |
| A.4 ✓ | Visual polish: palette, glow, particles, preview, cursors | 60 FPS with 40 bodies and 1,500 particles on both machines |
| A.5 | Sound effects (§4.1) | Every event in §4.1 sounds; a 40-shape pile stays quiet; 60 FPS kept; silent run without an audio device |
| A.6 | Booth-minimum shell (§6.0) | Idle reset, hint line, hold-Esc quit, and `1`/`2`/`3`/`M` work in both scenes |
| A.7 | Air Writing, milestones W.1–W.5 ([spec](phaseA-air-writing.md#10-milestones)) | Air Writing exit criteria in PLAN §3 met |
| A.8 | Test with ≥ 5 new users, 30-min soak test, Ubuntu laptop check, tag `v0.A` | PLAN §3 Phase A exit criteria met |

**Parked (PLAN D11, D14):** Train scene (capture flow, features, k-NN, rejection, test UI), PCA
map and data panel, custom gestures → Geometry effects, attract scene, dwell buttons. These were
the old A.5–A.7 and part of the old A.8.

## 8. Tests

- **Physics, headless:** spawn → step → body is below its start; release velocity mapping; body
  cap and fade-out; kinematic hand body transfers momentum; impact list is filled on collision.
- **Sound:** every synthesized sound is finite and within [−1, 1] with the expected length;
  the impact limiter keeps the loudest 3 per frame and honors the per-piece cooldown; the app
  runs when the mixer fails to open.
- *(parked)* **ML:** feature invariances; k-NN on synthetic poses (separable → high accuracy); rejection of a
  far-away pose; overlap hint triggers for near-identical classes; `CustomGestureDetector`
  threshold, debounce, and cooldown.
- **New fixtures:** `finger_count_1_to_5`, `grab_and_throw`, `two_hand_stretch`,
  `train_session_3_poses` (with Space marks at each pose change).
- **Scene smoke tests** under `SDL_VIDEODRIVER=dummy` with replayed fixtures: Geometry, Train, Attract.

## 9. Open questions (to settle during the phase)

- Should the camera image stay visible in Geometry (more "AI" feel) or be dimmed further (prettier)?
  Decide in user testing. It's a config value either way.
- Are the three effect slots the right ones? Swapping them is cheap. Choose what tests best with visitors.
