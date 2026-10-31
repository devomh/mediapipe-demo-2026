"""The ONLY module that imports mediapipe. Converts results into `Hand` objects (NumPy arrays)."""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ai_playground.config import RecognizerConfig
from ai_playground.types import Hand


def mediapipe_version() -> str:
    return mp.__version__


class Recognizer:
    """GestureRecognizer in VIDEO mode. Feed it mirrored RGB frames with increasing timestamps."""

    def __init__(self, cfg: RecognizerConfig, model_path: Path, video: bool = True) -> None:
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._r = vision.GestureRecognizer.create_from_options(opts)
        self._video = video
        self._last_ts = -1

    def recognize(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> tuple[Hand, ...]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            assert timestamp_ms is not None
            ts = max(int(timestamp_ms), self._last_ts + 1)  # must be strictly increasing
            self._last_ts = ts
            result = self._r.recognize_for_video(image, ts)
        else:
            result = self._r.recognize(image)
        return _to_hands(result)

    def close(self) -> None:
        self._r.close()


# On our mirrored frames MediaPipe labels each hand as the opposite of the visitor's own hand
# (verified on screen in milestone 0.2: right hand raised, mirror view, labelled "Left"), so we
# swap the label here and nothing downstream needs to know.
_VISITOR_HAND = {"Left": "Right", "Right": "Left"}


def _to_hands(result) -> tuple[Hand, ...]:
    hands = []
    for i, lms in enumerate(result.hand_landmarks):
        hd = result.handedness[i][0]
        g = result.gestures[i][0] if i < len(result.gestures) and result.gestures[i] else None
        hands.append(
            Hand(
                handedness=_VISITOR_HAND.get(hd.category_name, "Right"),
                handedness_score=float(hd.score),
                landmarks=np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32),
                world=np.array(
                    [(p.x, p.y, p.z) for p in result.hand_world_landmarks[i]], dtype=np.float32
                ),
                gesture=g.category_name if g and g.category_name else "None",
                gesture_score=float(g.score) if g else 0.0,
            )
        )
    return tuple(hands)
