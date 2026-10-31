"""Pooled particles: NumPy arrays, vectorized update, one batched additive blit per frame."""

from __future__ import annotations

import math

import numpy as np
import pygame

from ai_playground.render.effects import PALETTE, lighten

LEVELS = 8  # brightness steps; additive blending makes a darker sprite a fainter particle
SIZES = (3, 5, 8)  # sprite radii, px


def glow_sprite(color: tuple[int, int, int], radius: int, intensity: float = 1.0) -> pygame.Surface:
    """A soft radial dot for additive blending: bright centre, falling to black at the edge."""
    size = 2 * radius + 1
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) - radius
    falloff = np.clip(1.0 - np.sqrt(x * x + y * y) / radius, 0.0, 1.0) ** 2
    rgb = np.empty((size, size, 3), np.uint8)
    for i, c in enumerate(color):
        rgb[..., i] = np.clip(falloff * c * intensity, 0, 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return surf.convert() if pygame.display.get_surface() is not None else surf


class Particles:
    def __init__(self, cap: int = 1500, seed: int | None = None) -> None:
        self.cap = cap
        self.pos = np.zeros((cap, 2), np.float32)
        self.vel = np.zeros((cap, 2), np.float32)
        self.life = np.zeros(cap, np.float32)  # seconds left; <= 0 is a free slot
        self.max_life = np.ones(cap, np.float32)
        self.hue = np.zeros(cap, np.int16)
        self.size = np.zeros(cap, np.int16)  # index into SIZES
        self.rng = np.random.default_rng(seed)
        self._sprites: dict[tuple[int, int, int], pygame.Surface] = {}

    @property
    def alive(self) -> int:
        return int(np.count_nonzero(self.life > 0))

    def clear(self) -> None:
        self.life[:] = 0

    def emit(
        self,
        n: int,
        pos: tuple[float, float],
        hue: int,
        speed: tuple[float, float] = (100.0, 400.0),
        life: tuple[float, float] = (0.4, 0.9),
        base_vel: tuple[float, float] = (0.0, 0.0),
        spread: tuple[float, float] = (0.0, 2 * math.pi),  # angle range, radians
        size: int | None = None,
    ) -> None:
        """Emit `n` particles; when the pool is full, the ones closest to dying are reused."""
        n = min(n, self.cap)
        if n <= 0:
            return
        free = np.flatnonzero(self.life <= 0)
        if len(free) < n:
            others = np.argsort(self.life)[: n - len(free)]  # nearest the end of their life
            free = np.concatenate([free, others[np.isin(others, free, invert=True)]])[:n]
        idx = free[:n]
        k = len(idx)
        ang = self.rng.uniform(spread[0], spread[1], k)
        spd = self.rng.uniform(speed[0], speed[1], k)
        self.pos[idx] = pos
        self.vel[idx, 0] = base_vel[0] + np.cos(ang) * spd
        self.vel[idx, 1] = base_vel[1] + np.sin(ang) * spd
        lt = self.rng.uniform(life[0], life[1], k)
        self.life[idx] = lt
        self.max_life[idx] = lt
        self.hue[idx] = hue
        self.size[idx] = self.rng.integers(0, len(SIZES), k) if size is None else size

    def update(self, dt: float, gravity: float = 0.0, drag: float = 1.5) -> None:
        live = self.life > 0
        if not live.any():
            return
        v = self.vel[live]
        v *= max(0.0, 1.0 - drag * dt)
        v[:, 1] += gravity * dt
        self.vel[live] = v
        self.pos[live] += v * dt
        self.life[live] -= dt

    def _sprite(self, hue: int, size: int, level: int) -> pygame.Surface:
        key = (hue, size, level)
        s = self._sprites.get(key)
        if s is None:
            s = glow_sprite(lighten(PALETTE[hue], 0.35), SIZES[size], (level + 1) / LEVELS)
            self._sprites[key] = s
        return s

    def draw(self, surface: pygame.Surface) -> None:
        idx = np.flatnonzero(self.life > 0)
        if len(idx) == 0:
            return
        frac = np.clip(self.life[idx] / self.max_life[idx], 0.0, 1.0)
        levels = np.minimum((frac * LEVELS).astype(np.int32), LEVELS - 1)
        xy = self.pos[idx].astype(np.int32)
        hues, sizes = self.hue[idx], self.size[idx]
        seq = []
        for i in range(len(idx)):
            sp = self._sprite(int(hues[i]), int(sizes[i]), int(levels[i]))
            r = SIZES[sizes[i]]
            seq.append((sp, (int(xy[i, 0]) - r, int(xy[i, 1]) - r)))
        surface.fblits(seq, pygame.BLEND_ADD)


class GlowCache:
    """Soft halos behind shapes, drawn additively. Cached per (hue, size bucket, brightness).

    New sizes are scaled from one base sprite per (hue, brightness): computing a large gradient
    took 5-12 ms (a hitch while stretching a big shape at 1080p), scaling takes <= 0.5 ms, and
    a soft gradient looks the same either way.
    """

    BASE_RADIUS = 128

    def __init__(self, scale: float = 1.7, intensity: float = 0.45, bucket_px: int = 8) -> None:
        self.scale, self.intensity, self.bucket = scale, intensity, bucket_px
        self._cache: dict[tuple[int, int, int], pygame.Surface] = {}
        self._base: dict[tuple[int, int], pygame.Surface] = {}

    def sprite(self, hue: int, radius: float, alpha: float = 1.0) -> pygame.Surface:
        r = max(self.bucket, round(radius * self.scale / self.bucket) * self.bucket)
        level = max(0, min(LEVELS - 1, round(alpha * (LEVELS - 1))))
        key = (hue, r, level)
        s = self._cache.get(key)
        if s is None:
            base = self._base.get((hue, level))
            if base is None:
                k = self.intensity * (level + 1) / LEVELS
                base = self._base[(hue, level)] = glow_sprite(PALETTE[hue], self.BASE_RADIUS, k)
            s = self._cache[key] = pygame.transform.smoothscale(base, (2 * r + 1, 2 * r + 1))
        return s

    def draw(
        self,
        surface: pygame.Surface,
        pos: tuple[float, float],
        hue: int,
        radius: float,
        alpha: float = 1.0,
    ) -> None:
        if alpha <= 0:
            return
        s = self.sprite(hue, radius, alpha)
        w = s.get_width() // 2
        surface.blit(s, (round(pos[0]) - w, round(pos[1]) - w), special_flags=pygame.BLEND_ADD)

    def __len__(self) -> int:
        return len(self._cache)
