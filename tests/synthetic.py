"""Build anatomically plausible hands programmatically, for tests without a camera.

World landmarks are in metres (x right, y down, z away from the camera, origin near the palm),
like MediaPipe's. Image landmarks are the same skeleton projected to normalized coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ai_playground.types import Hand, HandFrame

SIZE = (640, 480)
PALM_LENGTH = 0.09  # wrist -> middle MCP, metres

# MCP positions in the palm plane (x, y), right hand palm-on to the camera, fingers up (-y).
_WRIST = np.array([0.0, 0.045, 0.0])
_MCP = {
    1: np.array([-0.021, -0.040, 0.0]),  # index
    2: np.array([0.000, -0.045, 0.0]),  # middle  (|wrist - middle MCP| = PALM_LENGTH)
    3: np.array([0.019, -0.042, 0.0]),  # ring
    4: np.array([0.035, -0.034, 0.0]),  # pinky
}
_SEGMENTS = {
    1: (0.039, 0.022, 0.017),
    2: (0.044, 0.026, 0.018),
    3: (0.041, 0.025, 0.018),
    4: (0.032, 0.018, 0.016),
}
_SPREAD = {1: -0.10, 2: 0.0, 3: 0.10, 4: 0.20}  # radians, finger direction in the palm plane
_NORMAL = np.array([0.0, 0.0, 1.0])  # fingers curl away from the camera (toward the palm side)

CURL_EXTENDED = (0.0, 0.0, 0.0)  # MCP, PIP, DIP flexion in degrees
CURL_FIST = (85.0, 105.0, 65.0)
CURL_PINCH = (35.0, 45.0, 25.0)


def _finger(f: int, curl: Sequence[float]) -> list[np.ndarray]:
    """[MCP, PIP, DIP, TIP] for finger f (1..4)."""
    a = _SPREAD[f]
    up = np.array([math.sin(a), -math.cos(a), 0.0])
    pts, p, phi = [_MCP[f]], _MCP[f], 0.0
    for length, flex in zip(_SEGMENTS[f], curl, strict=True):
        phi += math.radians(flex)
        p = p + length * (math.cos(phi) * up + math.sin(phi) * _NORMAL)
        pts.append(p)
    return pts


def _thumb_extended() -> list[np.ndarray]:
    cmc = np.array([-0.022, 0.030, 0.0])
    d = np.array([-math.sin(0.75), -math.cos(0.75), 0.0])
    mcp = cmc + 0.035 * d
    ip = mcp + 0.032 * d
    tip = ip + 0.028 * d
    return [cmc, mcp, ip, tip]


def _thumb_to(target: np.ndarray, bend: float = 0.012) -> list[np.ndarray]:
    """Thumb chain from the CMC to `target`, bowed outward (-x) like a real thumb."""
    cmc = np.array([-0.022, 0.030, 0.0])
    pts = [cmc]
    for s in (0.38, 0.70, 1.0):
        p = cmc + s * (target - cmc)
        p = p + np.array([-bend * math.sin(math.pi * s), 0.0, 0.0])
        pts.append(p)
    return pts


def world_hand(
    extended: Sequence[bool] = (True, True, True, True, True),
    pinch: float | None = None,
    fist: bool = False,
) -> np.ndarray:
    """(21, 3) world landmarks of a right hand.

    `pinch` (palm lengths) places the thumb tip that far from a half-curled index tip.
    """
    lm = np.zeros((21, 3))
    lm[0] = _WRIST
    if fist:
        extended = (False, False, False, False, False)
    for f in range(1, 5):
        if pinch is not None and f == 1:
            curl = CURL_PINCH
        else:
            curl = CURL_EXTENDED if extended[f] else CURL_FIST
        lm[1 + 4 * f : 5 + 4 * f] = _finger(f, curl)
    if pinch is not None:
        target = lm[8] + np.array([-pinch * PALM_LENGTH, 0.0, 0.0])
        lm[1:5] = _thumb_to(target)
    elif extended[0]:
        lm[1:5] = _thumb_extended()
    else:
        lm[1:5] = _THUMB_FOLDED
    return lm


# Tucked across the palm, bent ~140 degrees at the IP joint, tip in front of the middle MCP.
_THUMB_FOLDED = np.array(
    [[-0.022, 0.030, 0.0], [-0.030, 0.005, 0.010], [-0.015, -0.010, 0.022], [0.005, -0.012, 0.025]]
)


def make_hand(
    extended: Sequence[bool] = (True, True, True, True, True),
    pinch: float | None = None,
    fist: bool = False,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.2,
    handedness: str = "Right",
    handedness_score: float = 0.97,
    gesture: str = "None",
    gesture_score: float = 0.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
    size: tuple[int, int] = SIZE,
) -> Hand:
    """A Hand with world landmarks and image landmarks projected at `center`.

    `scale` is the image-space palm length (wrist -> middle MCP) in image heights.
    A Left hand is the mirror image of a Right one. `noise` is the std dev of Gaussian jitter in
    normalized image units (world gets the equivalent in metres).
    """
    world = world_hand(extended, pinch, fist)
    if handedness == "Left":
        world[:, 0] *= -1
    w, h = size
    k = scale / PALM_LENGTH  # metres -> image heights
    img = np.empty_like(world)
    img[:, 0] = center[0] + world[:, 0] * k * h / w
    img[:, 1] = center[1] + world[:, 1] * k
    img[:, 2] = world[:, 2] * k
    if noise:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0.0, noise, img.shape)
        world = world + rng.normal(0.0, noise / k, world.shape)
    return Hand(
        handedness=handedness,  # type: ignore[arg-type]
        handedness_score=handedness_score,
        landmarks=img.astype(np.float32),
        world=world.astype(np.float32),
        gesture=gesture,
        gesture_score=gesture_score,
    )


def frame(seq: int, t: float, hands: Sequence[Hand], size: tuple[int, int] = SIZE) -> HandFrame:
    return HandFrame(
        seq=seq, t_capture=t, t_result=t + 0.015, size=size, hands=tuple(hands), image=None
    )


def frames(hand_seq: Sequence[Sequence[Hand]], fps: float = 30.0) -> list[HandFrame]:
    return [frame(i + 1, i / fps, hs) for i, hs in enumerate(hand_seq)]
