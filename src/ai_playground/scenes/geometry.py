"""Magic Geometry: pinch to create a shape, hold to inflate it, release to drop or throw it.

- A.1: spawn, inflate, follow, drop/throw, walls, body cap.
- A.2: the other hand holds up 1-5 fingers to choose the shape (next or current pinch);
  a second pinch while one hand holds a shape stretches it between both pinch points.

- A.3: open palm = paddle, fist or pinch on a shape = grab and throw, both fists = clear
  (scenes/geometry_hands.py).

Glow and particles (A.4) come later.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import pygame

from ai_playground.gestures.engine import HandState
from ai_playground.physics.world import KINDS, World
from ai_playground.render import effects
from ai_playground.scenes.base import AppContext
from ai_playground.scenes.geometry_hands import HandPlay
from ai_playground.types import GestureEvent


@dataclass(slots=True)
class Pending:
    """A shape being made: follows the pinch and inflates (or stretches) until released."""

    kind: str
    hue: int
    t0: float
    pos: tuple[float, float]  # px
    radius: float  # px
    angle: float = 0.0  # radians
    partner: int | None = None  # second hand stretching this shape
    chosen: bool = False  # kind picked with fingers rather than by the automatic cycle


class FingerChooser:
    """Per-hand finger count that only counts once it has been held for `frames` frames.

    A hand closing in on a pinch is ignored: on its way it looks like 2, then 1 finger (index
    forward, the rest curled), which would otherwise turn the held shape into a bar, then a
    circle, just before a two-hand stretch. Deliberate counts keep thumb and index apart.
    """

    def __init__(self, frames: int, min_pinch_ratio: float = 0.0) -> None:
        self.frames = frames
        self.min_pinch_ratio = min_pinch_ratio
        self._runs: dict[int, tuple[int, int]] = {}  # hand id -> (count, consecutive frames)

    def update(self, hands: Sequence[HandState]) -> None:
        seen = set()
        for h in hands:
            if not h.visible:
                continue
            seen.add(h.id)
            count, n = self._runs.get(h.id, (-1, 0))
            if h.features.pinch_ratio < self.min_pinch_ratio:
                self._runs[h.id] = (-1, 0)  # about to pinch: not a choice
                continue
            c = h.features.finger_count
            self._runs[h.id] = (c, n + 1 if c == count else 1)
        for hid in [k for k in self._runs if k not in seen]:
            del self._runs[hid]

    def stable(self, hand_id: int) -> int | None:
        count, n = self._runs.get(hand_id, (-1, 0))
        return count if n >= self.frames else None


class GeometryScene:
    name = "geometry"
    show_vision = False  # the Vision overlay starts hidden here; V still toggles it

    def __init__(self, seed: int | None = None) -> None:
        self.ctx: AppContext | None = None
        self.world: World | None = None
        self.pending: dict[int, Pending] = {}  # by the hand that started it
        self.sparkles = effects.Sparkles()
        self.chooser = FingerChooser(5)
        self._rng = random.Random(seed)
        self._last_hue: int | None = None
        self._next_kind = 0
        self._t = 0.0
        self._hands: Sequence[HandState] = ()
        self._pinch: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}  # pos px, vel
        self._owner: dict[int, int] = {}  # stretching hand -> hand that owns the shape
        self.play: HandPlay | None = None

    # --- lifecycle ----------------------------------------------------------------------
    def enter(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.world = World(ctx.viewport.window_size, ctx.config.physics)
        g = ctx.config.geometry
        self.chooser = FingerChooser(g.choose_frames, g.choose_min_pinch_ratio)
        self.play = HandPlay(
            self.world,
            ctx.viewport,
            g,
            busy=lambda hid: hid in self.pending or hid in self._owner,
            on_clear=self._burst,
        )
        ctx.background.mode = "dim"
        ctx.background.dim_brightness = ctx.config.geometry.background_brightness

    def exit(self) -> None:
        self.clear()
        if self.play is not None:
            self.play.reset()
        self.world = None
        self.play = None
        self.ctx = None

    def clear(self) -> None:
        self.pending.clear()
        self._owner.clear()
        if self.play is not None:
            self.play.grabbed.clear()
        if self.world is not None:
            self.world.clear()

    def _burst(self) -> None:
        """Two-fist clear: every shape bursts (sparkles until A.4's particles)."""
        assert self.world is not None
        for piece in self.world.pieces:
            pos = piece.body.position
            self.sparkles.add((pos.x, pos.y), piece.hue)
        self.clear()

    # --- helpers ------------------------------------------------------------------------
    def _px(self, pos: tuple[float, float]) -> tuple[float, float]:
        assert self.ctx is not None
        return self.ctx.viewport.to_px(*pos)

    def _radius(self, held_s: float) -> float:
        assert self.ctx is not None
        g, h = self.ctx.config.geometry, self.ctx.viewport.window_size[1]
        k = min(1.0, max(0.0, held_s / g.inflate_s))
        return (g.min_size + (g.max_size - g.min_size) * k) * h

    def _vel_px(self, vel: tuple[float, float]) -> tuple[float, float]:
        assert self.ctx is not None
        vp, gain = self.ctx.viewport, self.ctx.config.geometry.throw_gain
        return (vel[0] * vp.dest_size[0] * gain, vel[1] * vp.dest_size[1] * gain)

    def chosen_kind(self, maker: int | None = None) -> str | None:
        """The kind another visible, non-pinching hand is choosing with its fingers, if any."""
        for h in self._hands:
            if h.id == maker or not h.visible or h.pinching or h.id in self._pinch:
                continue
            count = self.chooser.stable(h.id)
            if count is not None and 1 <= count <= len(KINDS):
                return KINDS[count - 1]
        return None

    def selection(self) -> tuple[str, bool]:
        """(kind, chosen) for the palette: the held shape, else the choice, else the next cycle."""
        if len(self.pending) == 1:  # one shape in hand, alone or stretched
            p = next(iter(self.pending.values()))
            return p.kind, p.chosen
        kind = self.chosen_kind()
        if kind is not None:
            return kind, True
        return KINDS[self._next_kind % len(KINDS)], False

    def _stretch(self, p: Pending, owner: int) -> None:
        assert self.ctx is not None and p.partner is not None
        (ax, ay), _ = self._pinch[owner]
        (bx, by), _ = self._pinch[p.partner]
        g, h = self.ctx.config.geometry, self.ctx.viewport.window_size[1]
        p.pos = ((ax + bx) / 2, (ay + by) / 2)
        # Scaled down: hands are naturally ~1 window height apart, so an exact span would sit at
        # the size cap and stop responding. The glowing line still joins both pinch points.
        r = g.stretch_gain * math.hypot(bx - ax, by - ay) / 2
        p.radius = min(max(r, g.min_size * h), g.stretch_max_size * h)
        p.angle = math.atan2(by - ay, bx - ax)

    def _finish(self, owner: int, e: GestureEvent) -> None:
        assert self.world is not None
        p = self.pending.pop(owner)
        vel = (0.0, 0.0)
        if e.data.get("reason") != "lost":  # a lost hand drops, it doesn't throw
            vels = [e.vel]
            other = p.partner if e.hand_id == owner else owner
            if p.partner is not None and other in self._pinch:
                vels.append(self._pinch[other][1])
            vel = self._vel_px(
                (sum(v[0] for v in vels) / len(vels), sum(v[1] for v in vels) / len(vels))
            )
        if p.partner is not None:
            self._owner.pop(p.partner, None)
        self.world.spawn(p.kind, p.pos, p.radius, vel=vel, angle=p.angle, hue=p.hue)
        self.sparkles.add(p.pos, p.hue)

    # --- input --------------------------------------------------------------------------
    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None:
        assert self.ctx is not None and self.world is not None
        self._hands = hands
        self.chooser.update(hands)
        for e in events:
            hid = e.hand_id
            if e.type == "pinch_start":
                self._pinch[hid] = (self._px(e.pos), e.vel)
                solo = [o for o, p in self.pending.items() if p.partner is None and o != hid]
                if solo:  # a second pinch while another hand holds a shape: stretch it
                    owner = solo[0]
                    self.pending[owner].partner = hid
                    self._owner[hid] = owner
                    self._stretch(self.pending[owner], owner)
                    continue
                # A pinch on an existing shape grabs it; in empty space it makes a new one.
                assert self.play is not None
                if self.play.try_grab(hid, self._pinch[hid][0], "pinch"):
                    continue
                kind = self.chosen_kind(maker=hid)
                chosen = kind is not None
                if kind is None:
                    kind = KINDS[self._next_kind % len(KINDS)]
                    self._next_kind += 1
                hue = effects.next_hue(self._last_hue, self._rng)
                self._last_hue = hue
                self.pending[hid] = Pending(
                    kind, hue, e.t, self._px(e.pos), self._radius(0.0), chosen=chosen
                )
            elif e.type == "pinch_move":
                self._pinch[hid] = (self._px(e.pos), e.vel)
                owner = self._owner.get(hid, hid)
                p = self.pending.get(owner)
                if p is None:
                    continue
                if p.partner is not None:
                    self._stretch(p, owner)
                else:
                    p.pos, p.radius = self._pinch[hid][0], self._radius(e.t - p.t0)
            elif e.type == "pinch_end":
                owner = self._owner.get(hid, hid)
                if owner in self.pending:
                    self._finish(owner, e)
                self._pinch.pop(hid, None)

        # The other hand can change the shape being held, too.
        for owner, p in self.pending.items():
            if p.partner is None:
                kind = self.chosen_kind(maker=owner)
                if kind is not None and kind != p.kind:
                    p.kind, p.chosen = kind, True

        assert self.play is not None
        self.play.handle(events, hands)

    # --- frame --------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        assert self.ctx is not None and self.world is not None
        self._t += dt
        if self.world.size != self.ctx.viewport.window_size:
            self.world.resize(self.ctx.viewport.window_size)
        assert self.play is not None
        self.play.update(dt)
        self.world.step(dt)
        self.sparkles.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        assert self.ctx is not None and self.world is not None
        frame = self.ctx.frame
        self.ctx.background.draw(surface, self.ctx.viewport, frame.image if frame else None)
        assert self.play is not None
        held = {id(p) for p in self.play.held.values()}
        for piece in self.world.pieces:
            effects.draw_piece(surface, piece, highlight=id(piece) in held)
        for hid, hb in self.world.hands.items():
            pos = (hb.body.position.x, hb.body.position.y)
            if hb.shape is not None:
                effects.draw_paddle(surface, pos, hb.shape.radius)
            elif hid in self.play.held:
                effects.draw_grab_ring(surface, pos, self._t)
        for pos, radius, frac in self.play.arming.values():
            effects.draw_paddle(surface, pos, radius, strength=frac)
        if self.play.clear_progress > 0:
            for pos in self.play.fist_pos.values():
                effects.draw_progress_ring(surface, pos, self.play.clear_progress)
        for owner, p in self.pending.items():
            if p.partner is not None and owner in self._pinch and p.partner in self._pinch:
                effects.draw_stretch_line(
                    surface, self._pinch[owner][0], self._pinch[p.partner][0], p.hue
                )
            effects.draw_preview(
                surface, p.kind, p.pos, p.radius, p.hue, self._t, p.angle, ring=p.partner is None
            )
        self.sparkles.draw(surface)
        effects.draw_palette(surface, *self.selection())
