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

## Ubuntu laptop (x86_64): pending
