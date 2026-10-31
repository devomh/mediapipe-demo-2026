import time

import pytest

from ai_playground.pacing import FramePacer


def test_paces_to_the_target_rate():
    p = FramePacer(100)
    p.tick()
    t0 = time.perf_counter()
    for _ in range(30):
        p.tick()
    assert (time.perf_counter() - t0) == pytest.approx(0.30, abs=0.03)
    assert p.fps == pytest.approx(100, rel=0.1)


def test_slow_frame_does_not_make_later_frames_rush():
    p = FramePacer(100)
    p.tick()
    time.sleep(0.05)  # a 50 ms hitch
    assert p.tick() >= 0.05
    dts = [p.tick() for _ in range(5)]
    assert min(dts) > 0.008  # back to ~10 ms frames, no burst of catch-up frames
