"""FramePacer logic on a fake clock: real-time tests are unreliable on shared CI runners
(30 frames at 100 FPS took 0.77 s instead of 0.30 s on a macOS runner)."""

import pytest

from ai_playground.pacing import FramePacer


class FakeTime:
    """A clock that advances when slept on, and by `spin_step` per read (a spinning loop)."""

    def __init__(self, oversleep: float = 0.0, spin_step: float = 1e-5):
        self.now, self.oversleep, self.spin_step = 0.0, oversleep, spin_step
        self.slept: list[float] = []

    def clock(self) -> float:
        self.now += self.spin_step
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s + self.oversleep


def pacer(t: FakeTime, fps: float = 100.0) -> FramePacer:
    return FramePacer(fps, spin_s=0.0015, clock=t.clock, sleep=t.sleep)


def test_frames_land_on_the_period():
    t = FakeTime()
    p = pacer(t)
    dts = [p.tick() for _ in range(30)]
    assert dts[1:] == pytest.approx([0.01] * 29, abs=1e-4)
    assert p.fps == pytest.approx(100, rel=0.01)


def test_sleeps_most_of_the_frame_and_spins_only_the_end():
    t = FakeTime()
    p = pacer(t)
    p.tick()
    t.now += 0.004  # 4 ms of frame work
    p.tick()
    assert t.slept[-1] == pytest.approx(0.01 - 0.004 - 0.0015, abs=1e-4)


def test_an_oversleeping_os_timer_does_not_slow_the_rate():
    """The reason for the pacer: OS sleep overshoots, the spin margin absorbs it."""
    t = FakeTime(oversleep=0.001)  # 1 ms overshoot, inside the 1.5 ms spin margin
    p = pacer(t)
    dts = [p.tick() for _ in range(30)]
    assert dts[1:] == pytest.approx([0.01] * 29, abs=1e-4)


def test_slow_frame_does_not_make_later_frames_rush():
    t = FakeTime()
    p = pacer(t)
    p.tick()
    t.now += 0.05  # a 50 ms hitch
    assert p.tick() == pytest.approx(0.05, abs=1e-3)
    dts = [p.tick() for _ in range(5)]
    assert dts == pytest.approx([0.01] * 5, abs=1e-4)  # no burst of catch-up frames
