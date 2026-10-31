import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Same rule as the app: pygame before cv2/mediapipe (see ai_playground/__main__.py).
import pygame  # noqa: E402, F401, I001

import pytest  # noqa: E402

from ai_playground import config  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> config.Config:
    return config.load()
