"""Live source: camera + recognizer in a worker thread, publishing HandFrames to a LatestSlot."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import cv2

from ai_playground.sources.camera import Camera
from ai_playground.sources.slot import LatestSlot
from ai_playground.tracking.recognizer import Recognizer
from ai_playground.types import HandFrame

log = logging.getLogger(__name__)


class LiveSource:
    """Owns an opened Camera and a Recognizer; releases both when stopped.

    MediaPipe's graph occasionally fails inside a frame (seen live with two hands: "Packet isn't
    the sole owner of the holder"), and a failed graph stays failed. For an unattended exhibit
    the worker then replaces the recognizer and carries on; only repeated failures stop it.
    """

    MAX_RESTARTS = 5  # within RESTART_WINDOW_S, before giving up with an error screen
    RESTART_WINDOW_S = 30.0

    def __init__(self, camera: Camera, make_recognizer: Callable[[], Recognizer]) -> None:
        self.camera = camera
        self.make_recognizer = make_recognizer
        self.recognizer = make_recognizer()
        self.slot: LatestSlot[HandFrame] = LatestSlot()
        self.error: str | None = None
        self.finished = False
        self.restarts = 0  # recognizer restarts since start, shown on the HUD
        self._recent: deque[float] = deque()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-source", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _restart_recognizer(self, exc: Exception) -> bool:
        """Replace a failed recognizer. False if it keeps failing and we should give up."""
        now = time.monotonic()
        self._recent.append(now)
        while self._recent and now - self._recent[0] > self.RESTART_WINDOW_S:
            self._recent.popleft()
        if len(self._recent) > self.MAX_RESTARTS:
            return False
        log.warning("recognizer failed (%s); restarting it", str(exc).splitlines()[0][:200])
        _close_quietly(self.recognizer)
        self.recognizer = self.make_recognizer()
        self.restarts += 1
        return True

    def _run(self) -> None:
        seq = 0
        failures = 0
        try:
            while not self._stop.is_set():
                ok, bgr = self.camera.read()
                t_capture = time.monotonic()
                if not ok or bgr is None:
                    failures += 1
                    if failures == 100:
                        log.warning("camera returned no frames 100 times in a row")
                    time.sleep(0.01)
                    continue
                failures = 0
                rgb = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
                try:
                    hands = self.recognizer.recognize(rgb, int(t_capture * 1000))
                except RuntimeError as e:  # MediaPipe graph failure; this frame is lost
                    if self._restart_recognizer(e):
                        continue
                    raise
                seq += 1
                h, w = rgb.shape[:2]
                self.slot.put(
                    HandFrame(
                        seq=seq,
                        t_capture=t_capture,
                        t_result=time.monotonic(),
                        size=(w, h),
                        hands=hands,
                        image=rgb,
                    )
                )
        except Exception as e:  # surface worker crashes on screen instead of dying silently
            log.exception("live source crashed")
            self.error = f"Camera/recognizer worker crashed: {str(e).splitlines()[0][:300]}"
        finally:
            try:
                self.camera.release()
            finally:
                _close_quietly(self.recognizer)


def _close_quietly(recognizer: Recognizer) -> None:
    """A failed MediaPipe graph re-raises its error on close; it must not kill the worker."""
    try:
        recognizer.close()
    except Exception as e:
        log.debug("recognizer close failed: %s", e)
