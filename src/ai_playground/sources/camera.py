"""OpenCV camera with per-platform backend and negotiated-format logging.

Open the camera on the main thread: on macOS, AVFoundation's permission prompt needs it.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import cv2

from ai_playground.config import CameraConfig

log = logging.getLogger(__name__)

PERMISSION_HINT = {
    "darwin": "System Settings > Privacy & Security > Camera: enable your terminal app, "
    "then restart it.",
    "linux": "Check that /dev/video* exists and that your user is in the 'video' group "
    "(sudo usermod -aG video $USER, then log out and in).",
}


def platform_hint() -> str:
    return PERMISSION_HINT.get(sys.platform, "Check that a camera is connected and not in use.")


def _backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


@dataclass(frozen=True, slots=True)
class Negotiated:
    width: int
    height: int
    fps: float
    fourcc: str


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, cfg: CameraConfig, index: int | None = None) -> None:
        self.cfg = cfg
        self.index = cfg.index if index is None else index
        self.cap: cv2.VideoCapture | None = None
        self.negotiated: Negotiated | None = None

    def open(self) -> Negotiated:
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if not cap.isOpened():
            raise CameraError(f"Camera {self.index} could not be opened. {platform_hint()}")
        if sys.platform.startswith("linux") and self.cfg.mjpg_on_linux:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored where unsupported
        self.cap = cap
        self.negotiated = _negotiated(cap)
        log.info(
            "camera %d: %dx%d @ %.1f fps, fourcc %s",
            self.index,
            self.negotiated.width,
            self.negotiated.height,
            self.negotiated.fps,
            self.negotiated.fourcc,
        )
        return self.negotiated

    def read(self):
        assert self.cap is not None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _negotiated(cap: cv2.VideoCapture) -> Negotiated:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    # AVFoundation reports -1 (no FourCC); only decode real codes.
    fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) if code > 0 else "?"
    return Negotiated(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        fourcc=fourcc.strip("\x00") or "?",
    )


def list_cameras(max_index: int = 5) -> list[tuple[int, Negotiated | None]]:
    """Probe indices 0..max_index. Returns (index, negotiated or None if it didn't open/read)."""
    out = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, _backend())
        if cap.isOpened() and cap.read()[0]:
            out.append((i, _negotiated(cap)))
        else:
            out.append((i, None))
        cap.release()
    return out
