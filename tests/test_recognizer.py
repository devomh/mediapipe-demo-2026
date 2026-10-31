"""Smoke: the MediaPipe wheel and model work on this platform (still images, no camera)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_playground.gestures import features as F
from ai_playground.tracking.recognizer import Recognizer

IMAGES = Path(__file__).parent / "fixtures" / "images"
# MediaPipe sample images (Apache-2.0). Not mirrored, so handedness is not asserted.
CASES = {"thumbs_up": "Thumb_Up", "victory": "Victory", "pointing_up": "Pointing_Up"}
# (thumb, index, middle, ring, pinky) on real MediaPipe landmarks; synthetic hands are too ideal
# to catch threshold problems on their own.
FINGERS = {
    "thumbs_up": (True, False, False, False, False),
    "thumbs_down": (True, False, False, False, False),
    "victory": (False, True, True, False, False),
    "pointing_up": (False, True, False, False, False),
}


def load_rgb(name):
    return cv2.cvtColor(cv2.imread(str(IMAGES / f"{name}.jpg")), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def recognizer(cfg):
    if not cfg.model_path().is_file():
        pytest.skip("model missing: run `uv run python scripts/fetch_models.py`")
    r = Recognizer(cfg.recognizer, cfg.model_path(), video=False)
    yield r
    r.close()


@pytest.mark.parametrize("name", CASES)
def test_recognizes_still_image(recognizer, name):
    hands = recognizer.recognize(load_rgb(name))
    assert len(hands) == 1
    h = hands[0]
    assert h.gesture == CASES[name]
    assert h.landmarks.shape == (21, 3) and h.landmarks.dtype == np.float32
    assert h.world.shape == (21, 3)
    assert h.landmarks[:, :2].min() >= 0.0 and h.landmarks[:, :2].max() <= 1.0
    assert 0.03 < np.linalg.norm(h.world[0] - h.world[9]) < 0.2  # palm length in metres


@pytest.mark.parametrize("mirrored", [False, True], ids=["raw", "mirrored"])
@pytest.mark.parametrize("name", FINGERS)
def test_finger_extended_on_real_landmarks(cfg, recognizer, name, mirrored):
    rgb = load_rgb(name)
    if mirrored:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    (h,) = recognizer.recognize(rgb)
    assert F.finger_extended(h.world, cfg.gestures.fingers) == FINGERS[name]


def test_video_mode_accepts_repeated_timestamps(cfg):
    if not cfg.model_path().is_file():
        pytest.skip("model missing")
    r = Recognizer(cfg.recognizer, cfg.model_path(), video=True)
    try:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            assert r.recognize(img, 1000) == ()  # forced strictly increasing internally
    finally:
        r.close()


def test_handedness_is_swapped_to_the_visitors_own_hand():
    """MediaPipe calls the visitor's right hand "Left" on mirrored frames (verified on screen)."""
    from types import SimpleNamespace as NS

    from ai_playground.tracking.recognizer import _to_hands

    pts = [NS(x=0.5, y=0.5, z=0.0)] * 21

    def result(label):
        return NS(
            hand_landmarks=[pts],
            hand_world_landmarks=[pts],
            handedness=[[NS(category_name=label, score=0.9)]],
            gestures=[[NS(category_name="Open_Palm", score=0.8)]],
        )

    assert _to_hands(result("Left"))[0].handedness == "Right"
    assert _to_hands(result("Right"))[0].handedness == "Left"
