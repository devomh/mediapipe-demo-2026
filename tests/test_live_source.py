"""LiveSource worker: keeps running through MediaPipe graph failures (no camera, no MediaPipe)."""

import time

import numpy as np
import pytest

from ai_playground.sources.live import LiveSource


class FakeCamera:
    def __init__(self):
        self.released = False

    def read(self):
        time.sleep(0.002)
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


class FakeRecognizer:
    """Fails on the calls listed in `fail_on` (global call numbers), like a broken graph."""

    calls = 0

    def __init__(self, fail_on=(), close_raises=True):
        self.fail_on, self.close_raises, self.closed = set(fail_on), close_raises, False

    def recognize(self, rgb, ts):
        FakeRecognizer.calls += 1
        if FakeRecognizer.calls in self.fail_on:
            raise RuntimeError("CalculatorGraph::Run() failed:\nPacket isn't the sole owner")
        return ()

    def close(self):
        self.closed = True
        if self.close_raises:
            raise RuntimeError("CalculatorGraph::Run() failed")  # what killed the thread live


def run_until(src, cond, timeout=2.0):
    src.start()
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.005)
    src.stop()


@pytest.fixture(autouse=True)
def reset_calls():
    FakeRecognizer.calls = 0


def test_recovers_from_a_graph_failure_and_keeps_publishing():
    made = []

    def make():
        made.append(FakeRecognizer(fail_on={5}))
        return made[-1]

    cam = FakeCamera()
    src = LiveSource(cam, make)
    run_until(src, lambda: src.slot.seq >= 20)
    assert src.error is None
    assert src.restarts == 1 and len(made) == 2
    assert made[0].closed  # the failed one was closed, and its raising close() was swallowed
    assert src.slot.seq >= 20  # kept publishing after the failure
    assert cam.released and made[1].closed


def test_gives_up_after_repeated_failures():
    src = LiveSource(FakeCamera(), lambda: FakeRecognizer(fail_on=set(range(1, 1000))))
    run_until(src, lambda: src.error is not None)
    assert src.error is not None and "worker crashed" in src.error
    assert src.restarts == LiveSource.MAX_RESTARTS
    assert "\n" not in src.error  # one line for the error screen


def test_a_raising_close_on_shutdown_does_not_escape(recwarn):
    cam = FakeCamera()
    src = LiveSource(cam, lambda: FakeRecognizer(close_raises=True))
    run_until(src, lambda: src.slot.seq >= 3)
    assert src.error is None and cam.released
