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


@pytest.fixture(autouse=True)
def _default_ui_scale():
    """The UI scale is module state (render/ui); start every test at the default window size."""
    from ai_playground.render import ui

    ui.set_factor(1.0)
    ui.set_window_height(ui.DESIGN_HEIGHT)
    yield
    ui.set_factor(1.0)
    ui.set_window_height(ui.DESIGN_HEIGHT)
