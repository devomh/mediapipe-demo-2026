"""Live source: camera + recognizer in a worker thread, publishing HandFrames to a LatestSlot."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from ai_playground.sources.camera import Camera
from ai_playground.sources.slot import LatestSlot
from ai_playground.tracking.recognizer import Recognizer
from ai_playground.types import HandFrame

log = logging.getLogger(__name__)


class LiveSource:
    """Owns an opened Camera and a Recognizer; releases both when stopped."""

    def __init__(self, camera: Camera, recognizer: Recognizer) -> None:
        self.camera = camera
        self.recognizer = recognizer
        self.slot: LatestSlot[HandFrame] = LatestSlot()
        self.error: str | None = None
        self.finished = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-source", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

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
                hands = self.recognizer.recognize(rgb, int(t_capture * 1000))
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
            self.error = f"Camera/recognizer worker crashed: {e}"
        finally:
            self.camera.release()
            self.recognizer.close()
