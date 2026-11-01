# Measurements

Phase 0 §13. Fill in each "pending" cell on the real machine with the real camera, using the HUD
(`H`) and the `right_static_open` / `fist_palm_cycle` / `right_pinch_x5` fixtures.

## MacBook Pro (Apple M5 Pro), macOS, Python 3.12.13, mediapipe 1.2.0

### Live camera (2026-10-30)

Built-in camera 0, 10.5 s run, one then two open hands, recording on.

| Metric | Value |
|--------|-------|
| Camera FPS negotiated / delivered | 640×480 @ 30 / 30.0 |
| Inference ms (mean / p95) | 6.6 / 7.6 |
| Render FPS | 59.5 |
| Pipeline latency ms (mean / p95) | 31.7 / 40.1 |
| CPU % | pending |
| Pinch delay after the raw threshold crossing | 1–2 frames, all 5 pinches (`right_pinch_x5`); Space marks too imprecise to measure it |
| False pinch starts / min | pending (`right_static_open`, `fist_palm_cycle`) |
| Open palm read as 5 fingers | 99 % (right), 100 % (left) of `Open_Palm` frames |

### Without a camera (2026-10-30)

Real `App`, real Cocoa window, `LiveSource` worker, and MediaPipe VIDEO mode at 640×480, fed by
a fake camera that serves `tests/fixtures/images/victory.jpg` (one hand) every 33 ms.

| Metric | Value | Note |
|--------|-------|------|
| Inference ms (mean / p95) | 10.2 / 14.1 | budget: worker thread, ~10–25 ms expected |
| Pipeline latency ms (mean / p95) | 36.3 / 44.5 | exit criterion < 100 ms |
| Render FPS (steady) | 58.8 | target 60 |
| Delivered frame rate | 21 | not representative: fake camera sleeps 33 ms *plus* inference |

Main-loop stage costs, two synthetic hands, mean (p95) in ms:

| Stage | 1280×720 | 1920×1080 | Budget (§10) |
|-------|----------|-----------|--------------|
| Tracking + filters + engine | 0.10 (0.16) | 0.13 (0.17) | ≤ 1 |
| Camera background blit (scaled) | 0.81 (0.90) | 1.69 (1.73) | ≤ 3 |
| Vision overlay + HUD | 0.52 (0.69) | 0.56 (0.61) | ≤ 6 |
| `display.flip()` (vsync) | 1.77 (6.76) | 5.83 (6.02) | |

Synthetic gesture checks (`tests/test_engine.py`): fast pinch fires 1 frame after the raw
crossing (criterion ≤ 2); a 30 s noisy pinch hold gives 1 start and no flicker.

### Phase A.4: 40 shapes + 1,500 particles (2026-10-31)

Exit criterion: 60 FPS with 40 bodies and 1,500 particles. Full Geometry frame: camera
background, glows, shapes, particles, hand cursors, palette, HUD, keys overlay.

| Measurement | 1280×720 | 1920×1080 |
|---|---|---|
| Frame work, mean / p95 (synthetic, real window) | 5.1 / 5.6 ms | 7.2 / 7.5 ms |
| 40 additive glows (all shapes at max stretch size) | 0.72 ms | 1.62 ms |
| 1,500 particles: update / draw | 0.06 / 0.64 ms | 0.07 / 0.64 ms |
| **Live camera, real app** | **60.0 FPS render**; camera 30.0 FPS; inference 8.8 / 10.0 ms; latency 33.7 / 35.2 ms | – |

- Render rate is set by frame pacing, not work: `pygame.time.Clock.tick(60)` gave 58.9–59.1 FPS
  with 5 ms of work per frame; the app now sleeps until ~1.5 ms before the deadline and then
  spins (`pacing.py`), giving 60.0 FPS without slowing the camera worker.
- Glow sprites are scaled from one base per colour: computing a large gradient directly took
  5–12 ms (a hitch while stretching a big shape at 1080p), scaling takes ≤ 0.64 ms.

## Ubuntu desktop: Dell XPS 8950 (Intel i7-12700, 20 threads), Ubuntu 26.04.1, Python 3.12.13, mediapipe 1.2.0

### Live camera, Geometry scene (2026-11-01)

USB "HD Web Camera" at index 0, `--scene geometry`, 1280×720 window under XWayland (GNOME
Wayland session). HUD values read from 3–4 window captures over about 10–15 s per run; CPU and
memory sampled with `pidstat` over the same period (20 s / 15 s).

| Metric | One hand | Two hands |
|--------|----------|-----------|
| Camera FPS delivered | 30.0 | 30.0 |
| Inference ms (mean; the HUD does not show p95) | 21 | 17–18 |
| Render FPS | 60.0 | 59.7–60.0 |
| Pipeline latency ms (mean / p95) | 46 / 49–60 | 40–41 / 51–54 |
| CPU % (of one core; 20 threads total) | 107 (102.5 user + 4.75 system) | 108 (103.8 user + 4.3 system) |
| Memory (RSS) | 345 MB | 404 MB |

- Inference takes about 2–2.4× as long as on the M5 Pro (17–21 vs 8.8 ms). It runs on the worker
  thread, so render stays at 60 FPS, and latency is about half of the 100 ms limit.
- The second hand added no measurable cost. The two-hand run was a little faster, which is
  within run-to-run variation (hand pose, CPU clock), not an effect of the second hand.
- RSS grew 345 → 404 MB between the runs, which were about 3 minutes apart. That is one sample,
  not a trend; the 30-minute soak test (A.9) is where to check for a leak.

## Ubuntu laptop (x86_64): pending
