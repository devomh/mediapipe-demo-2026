"""Hand features: pure functions over landmark arrays (portable core: stdlib + NumPy only).

Gesture distances use world landmarks (metres) so they don't depend on distance to the camera.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ai_playground.config import FingersConfig

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
# Landmark chains per finger (thumb..pinky), used for drawing bones in per-finger colors.
FINGER_CHAINS = (
    (WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
PALM_EDGES = ((INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP))
_PIP_TIP = (
    (INDEX_PIP, INDEX_TIP),
    (MIDDLE_PIP, MIDDLE_TIP),
    (RING_PIP, RING_TIP),
    (PINKY_PIP, PINKY_TIP),
)
_PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def palm_length(world: np.ndarray) -> float:
    """Wrist -> middle MCP distance in metres."""
    return float(np.linalg.norm(world[WRIST] - world[MIDDLE_MCP]))


def pinch_ratio(world: np.ndarray, finger: int = 1) -> float:
    """Thumb tip -> fingertip distance in palm lengths. Pinched ~0.1-0.2, open ~0.8-1.2."""
    pl = palm_length(world)
    if pl <= 1e-9:
        return math.inf
    return float(np.linalg.norm(world[THUMB_TIP] - world[FINGERTIPS[finger]])) / pl


def pinch_point(landmarks: np.ndarray) -> tuple[float, float]:
    """Midpoint of thumb tip and index tip in normalized screen coordinates."""
    p = (landmarks[THUMB_TIP, :2] + landmarks[INDEX_TIP, :2]) * 0.5
    return float(p[0]), float(p[1])


def joint_angle_deg(points: np.ndarray, a: int, b: int, c: int) -> float:
    """Angle at b between b->a and b->c; 180 means a straight joint."""
    u, v = points[a] - points[b], points[c] - points[b]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu <= 1e-9 or nv <= 1e-9:
        return 180.0
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def finger_extended(world: np.ndarray, cfg: FingersConfig) -> tuple[bool, ...]:
    """(thumb, index, middle, ring, pinky) extended flags.

    Joint angles don't separate the states in MediaPipe world landmarks (curled fingers measure
    up to ~150 degrees at the PIP, extended ones as low as ~160; thumbs overlap the same way at
    the IP), so both rules use distances, checked on real hands:

    - fingers: tip -> wrist / PIP -> wrist. Extended: median 1.38 live, but down to ~1.1 while a
      hand enters the frame; curled: <= 1.17 (a fist seen edge-on). 1.2 favours curled-correct.
    - thumb: tip -> pinky MCP in palm lengths. Sticking out >= 0.88, folded over the palm <= 0.79.
    """
    pl = palm_length(world)
    thumb_reach = (
        float(np.linalg.norm(world[THUMB_TIP] - world[PINKY_MCP])) / pl if pl > 1e-9 else 0
    )
    flags = [thumb_reach > cfg.thumb_tip_pinky_dist]
    for finger in range(1, 5):
        flags.append(finger_extension(world, finger) > cfg.tip_pip_ratio)
    return tuple(flags)


def finger_extension(world: np.ndarray, finger: int) -> float:
    """Tip -> wrist / PIP -> wrist for a non-thumb finger (1..4). ~1.4 straight, < 1 curled."""
    pip, tip = _PIP_TIP[finger - 1]
    d_pip = float(np.linalg.norm(world[pip] - world[WRIST]))
    if d_pip <= 1e-9:
        return 0.0
    return float(np.linalg.norm(world[tip] - world[WRIST])) / d_pip


def palm_center(landmarks: np.ndarray) -> tuple[float, float]:
    p = landmarks[_PALM_POINTS, :2].mean(axis=0)
    return float(p[0]), float(p[1])


def palm_roll(landmarks: np.ndarray, size: tuple[int, int]) -> float:
    """Angle (radians) of the index-MCP -> pinky-MCP vector in pixel space."""
    d = (landmarks[PINKY_MCP, :2] - landmarks[INDEX_MCP, :2]) * np.asarray(size, dtype=np.float32)
    return math.atan2(float(d[1]), float(d[0]))


def hand_scale_image(landmarks: np.ndarray, size: tuple[int, int]) -> float:
    """Image-space wrist -> middle-MCP distance in units of image height."""
    w, h = size
    d = landmarks[MIDDLE_MCP, :2] - landmarks[WRIST, :2]
    return math.hypot(float(d[0]) * w / h, float(d[1]))


class VelocityTracker:
    """Finite-difference velocity of a 2D point over a short history window."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._hist: deque[tuple[float, float, float]] = deque()

    def update(self, t: float, p: tuple[float, float]) -> tuple[float, float]:
        self._hist.append((t, p[0], p[1]))
        while len(self._hist) > 2 and t - self._hist[1][0] >= self.window_s:
            self._hist.popleft()
        t0, x0, y0 = self._hist[0]
        dt = t - t0
        if dt <= 1e-6:
            return (0.0, 0.0)
        return ((p[0] - x0) / dt, (p[1] - y0) / dt)

    def reset(self) -> None:
        self._hist.clear()


@dataclass(frozen=True, slots=True)
class HandFeatures:
    palm_length: float
    pinch_ratio: float
    pinch_point: tuple[float, float]
    finger_extended: tuple[bool, ...]
    finger_count: int
    palm_center: tuple[float, float]
    palm_roll: float
    hand_scale_image: float
    index_extension: float  # finger_extension of the index; low while curling into a fist


def compute(
    landmarks: np.ndarray, world: np.ndarray, size: tuple[int, int], cfg: FingersConfig
) -> HandFeatures:
    ext = finger_extended(world, cfg)
    return HandFeatures(
        palm_length=palm_length(world),
        pinch_ratio=pinch_ratio(world),
        pinch_point=pinch_point(landmarks),
        finger_extended=ext,
        finger_count=sum(ext),
        palm_center=palm_center(landmarks),
        palm_roll=palm_roll(landmarks, size),
        hand_scale_image=hand_scale_image(landmarks, size),
        index_extension=finger_extension(world, 1),
    )
