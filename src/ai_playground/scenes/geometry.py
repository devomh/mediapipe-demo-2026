"""Magic Geometry: pinch to create a shape, hold to inflate it, release to drop or throw it.

A.1 scope: spawn, inflate, follow, drop/throw, walls, body cap. Shape choice by finger count,
stretch, paddle, grab, clear gesture, and glow/particles come in later milestones.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pygame

from ai_playground.gestures.engine import HandState
from ai_playground.physics.world import KINDS, World
from ai_playground.render import effects
from ai_playground.scenes.base import AppContext
from ai_playground.types import GestureEvent


@dataclass(slots=True)
class Pending:
    """A shape being made: follows the pinch and inflates until released."""

    kind: str
    hue: int
    t0: float
    pos: tuple[float, float]  # px
    radius: float  # px


class GeometryScene:
    name = "geometry"
    show_vision = False  # the Vision overlay starts hidden here; V still toggles it

    def __init__(self, seed: int | None = None) -> None:
        self.ctx: AppContext | None = None
        self.world: World | None = None
        self.pending: dict[int, Pending] = {}
        self.sparkles = effects.Sparkles()
        self._rng = random.Random(seed)
        self._last_hue: int | None = None
        self._next_kind = 0
        self._t = 0.0

    # --- lifecycle ----------------------------------------------------------------------
    def enter(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.world = World(ctx.viewport.window_size, ctx.config.physics)
        ctx.background.mode = "dim"
        ctx.background.dim_brightness = ctx.config.geometry.background_brightness

    def exit(self) -> None:
        self.pending.clear()
        self.world = None
        self.ctx = None

    def clear(self) -> None:
        self.pending.clear()
        if self.world is not None:
            self.world.clear()

    # --- helpers ------------------------------------------------------------------------
    def _px(self, pos: tuple[float, float]) -> tuple[float, float]:
        assert self.ctx is not None
        return self.ctx.viewport.to_px(*pos)

    def _radius(self, held_s: float) -> float:
        assert self.ctx is not None
        g, h = self.ctx.config.geometry, self.ctx.viewport.window_size[1]
        k = min(1.0, max(0.0, held_s / g.inflate_s))
        return (g.min_size + (g.max_size - g.min_size) * k) * h

    def _take_kind(self) -> str:
        kind = KINDS[self._next_kind % len(KINDS)]
        self._next_kind += 1
        return kind

    # --- input --------------------------------------------------------------------------
    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None:
        assert self.ctx is not None and self.world is not None
        for e in events:
            if e.type == "pinch_start":
                hue = effects.next_hue(self._last_hue, self._rng)
                self._last_hue = hue
                self.pending[e.hand_id] = Pending(
                    self._take_kind(), hue, e.t, self._px(e.pos), self._radius(0.0)
                )
            elif e.type == "pinch_move" and e.hand_id in self.pending:
                p = self.pending[e.hand_id]
                p.pos, p.radius = self._px(e.pos), self._radius(e.t - p.t0)
            elif e.type == "pinch_end" and e.hand_id in self.pending:
                p = self.pending.pop(e.hand_id)
                vel = (0.0, 0.0)
                if e.data.get("reason") != "lost":  # a lost hand drops, it doesn't throw
                    vp, gain = self.ctx.viewport, self.ctx.config.geometry.throw_gain
                    vel = (e.vel[0] * vp.dest_size[0] * gain, e.vel[1] * vp.dest_size[1] * gain)
                self.world.spawn(p.kind, p.pos, p.radius, vel=vel, hue=p.hue)
                self.sparkles.add(p.pos, p.hue)

    # --- frame --------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        assert self.ctx is not None and self.world is not None
        self._t += dt
        if self.world.size != self.ctx.viewport.window_size:
            self.world.resize(self.ctx.viewport.window_size)
        self.world.step(dt)
        self.sparkles.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        assert self.ctx is not None and self.world is not None
        frame = self.ctx.frame
        self.ctx.background.draw(surface, self.ctx.viewport, frame.image if frame else None)
        for piece in self.world.pieces:
            effects.draw_piece(surface, piece)
        for p in self.pending.values():
            effects.draw_preview(surface, p.kind, p.pos, p.radius, p.hue, self._t)
        self.sparkles.draw(surface)
