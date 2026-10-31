"""2D physics world (pymunk) in screen pixels, y down. Fixed timestep, body cap, impact log."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymunk

from ai_playground.config import PhysicsConfig

# Shape kinds by number of corners, which is also how visitors choose them (finger count).
KINDS = ("circle", "bar", "triangle", "square", "pentagon")
_DENSITY = {"circle": 1.0, "bar": 1.4, "triangle": 1.0, "square": 1.1, "pentagon": 1.1}
_WALL = 1  # collision types
_PIECE = 2
_HAND = 3


def shape_vertices(kind: str, radius: float) -> list[tuple[float, float]]:
    """Local vertices for a polygon kind; `radius` is the circumradius (bar: half its length)."""
    if kind == "bar":
        w, h = radius, radius / 4  # 4:1 aspect
        return [(-w, -h), (w, -h), (w, h), (-w, h)]
    n = {"triangle": 3, "square": 4, "pentagon": 5}[kind]
    # Point up (-y) for odd n; flat bottom for squares.
    start = -math.pi / 2 if n % 2 else -math.pi / 4
    return [
        (
            radius * math.cos(start + 2 * math.pi * i / n),
            radius * math.sin(start + 2 * math.pi * i / n),
        )
        for i in range(n)
    ]


@dataclass(eq=False, slots=True)
class Piece:
    kind: str
    radius: float
    hue: int  # index into the palette
    body: pymunk.Body
    shape: pymunk.Shape
    fade: float | None = None  # seconds into fade-out, None while alive

    @property
    def alpha(self) -> float:
        return 1.0 if self.fade is None else max(0.0, 1.0 - self.fade / self._fade_s)

    _fade_s: float = 0.5


@dataclass(eq=False, slots=True)
class HandBody:
    """A hand in the physics world: a kinematic body steered toward the tracked palm.

    With a shape it is a paddle that pushes and bats pieces; without one it is an invisible
    anchor that a grabbed piece hangs from.
    """

    body: pymunk.Body
    shape: pymunk.Circle | None
    target: tuple[float, float]
    joint: pymunk.Constraint | None = None
    held: Piece | None = None


@dataclass(frozen=True, slots=True)
class Impact:
    impulse: float
    pos: tuple[float, float]
    pieces: tuple[Piece | None, Piece | None]  # None for walls


@dataclass
class World:
    size: tuple[int, int]
    cfg: PhysicsConfig
    pieces: list[Piece] = field(default_factory=list)
    impacts: list[Impact] = field(default_factory=list)  # filled by the last step()

    def __post_init__(self) -> None:
        self.space = pymunk.Space()
        self.space.sleep_time_threshold = self.cfg.sleep_time_s
        self.space.on_collision(post_solve=self._on_post_solve)
        self._walls: list[pymunk.Segment] = []
        self._acc = 0.0
        self._by_shape: dict[pymunk.Shape, Piece] = {}
        self.hands: dict[int, HandBody] = {}
        self.resize(self.size)

    @property
    def timestep(self) -> float:
        return 1.0 / self.cfg.steps_per_second

    # --- setup --------------------------------------------------------------------------
    def resize(self, size: tuple[int, int]) -> None:
        self.size = size
        w, h = size
        self.space.gravity = (0.0, self.cfg.gravity_factor * h)
        if self._walls:
            self.space.remove(*self._walls)
        sb = self.space.static_body
        r = 10.0
        top = -3.0 * h  # walls reach far above the screen; the ceiling stays open
        self._walls = [
            pymunk.Segment(sb, (-r, h + r), (w + r, h + r), r),  # floor, just below the screen
            pymunk.Segment(sb, (-r, top), (-r, h + r), r),
            pymunk.Segment(sb, (w + r, top), (w + r, h + r), r),
        ]
        for s in self._walls:
            s.elasticity, s.friction, s.collision_type = (
                self.cfg.elasticity,
                self.cfg.friction,
                _WALL,
            )
        self.space.add(*self._walls)

    # --- pieces -------------------------------------------------------------------------
    def spawn(
        self,
        kind: str,
        pos: tuple[float, float],
        radius: float,
        vel: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        hue: int = 0,
    ) -> Piece:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        alive = [p for p in self.pieces if p.fade is None]
        if len(alive) >= self.cfg.max_bodies:
            alive[0].fade = 0.0  # oldest first
        body = pymunk.Body()
        if kind == "circle":
            shape: pymunk.Shape = pymunk.Circle(body, radius)
        else:
            shape = pymunk.Poly(body, shape_vertices(kind, radius))
        shape.density = _DENSITY[kind] * 1e-3
        shape.elasticity, shape.friction = self.cfg.elasticity, self.cfg.friction
        shape.collision_type = _PIECE
        body.position = pos
        body.angle = angle
        body.velocity = self.limit_throw(vel)
        self.space.add(body, shape)
        piece = Piece(kind, radius, hue, body, shape, _fade_s=self.cfg.fade_s)
        self.pieces.append(piece)
        self._by_shape[shape] = piece
        return piece

    def limit_throw(self, vel: tuple[float, float]) -> tuple[float, float]:
        """Cap a throw so shapes stay in play: overall speed, and how high they can rise.

        With an open ceiling, an uncapped upward flick would leave the screen for seconds and
        could sail over a wall. Rising at most `max_rise` window heights brings it back quickly.
        """
        h = self.size[1]
        vx, vy = vel
        max_speed = self.cfg.max_speed * h
        speed = math.hypot(vx, vy)
        if speed > max_speed:
            vx, vy = vx * max_speed / speed, vy * max_speed / speed
        max_up = math.sqrt(2.0 * self.space.gravity.y * self.cfg.max_rise * h)
        return (vx, max(vy, -max_up))  # y is down, so rising is negative

    def remove(self, piece: Piece) -> None:
        for hb in self.hands.values():
            if hb.held is piece:
                self._unhook(hb)
        self.space.remove(piece.body, piece.shape)
        self._by_shape.pop(piece.shape, None)
        self.pieces.remove(piece)

    def clear(self) -> None:
        for p in list(self.pieces):
            self.remove(p)

    # --- hands --------------------------------------------------------------------------
    def set_hand(
        self, hand_id: int, pos: tuple[float, float], radius: float | None = None
    ) -> HandBody:
        """Steer a hand body toward `pos`; `radius` makes it a paddle, None an anchor."""
        hb = self.hands.get(hand_id)
        if hb is not None and (hb.shape is None) != (radius is None):
            self.remove_hand(hand_id)  # paddle <-> anchor: start over
            hb = None
        if hb is None:
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            body.position = pos
            shape = None
            if radius is not None:
                shape = pymunk.Circle(body, radius)
                shape.elasticity, shape.friction = self.cfg.elasticity, self.cfg.friction
                shape.collision_type = _HAND
                self.space.add(body, shape)
            else:
                self.space.add(body)
            hb = self.hands[hand_id] = HandBody(body, shape, pos)
        else:
            hb.target = pos
            if hb.shape is not None and abs(hb.shape.radius - radius) > 0.5:
                hb.shape.unsafe_set_radius(radius)
        return hb

    def remove_hand(self, hand_id: int) -> None:
        hb = self.hands.pop(hand_id, None)
        if hb is None:
            return
        self._unhook(hb)
        self.space.remove(*[x for x in (hb.body, hb.shape) if x is not None])

    def grab(self, hand_id: int, piece: Piece, point: tuple[float, float]) -> None:
        """Hang `piece` from the hand's anchor at world `point`; it can still swing."""
        hb = self.hands[hand_id]
        self._unhook(hb)
        joint = pymunk.PivotJoint(hb.body, piece.body, point)
        # Finite force: a piece pinned against a wall or the pile slips instead of exploding.
        joint.max_force = piece.body.mass * self.space.gravity.y * 60
        self.space.add(joint)
        hb.joint, hb.held = joint, piece
        piece.body.activate()

    def release(
        self, hand_id: int, throw: bool = True, velocity: tuple[float, float] | None = None
    ) -> Piece | None:
        """Let go of the held piece: thrown with `velocity` (default: its own), capped; or
        stopped if not thrown."""
        hb = self.hands.get(hand_id)
        if hb is None or hb.held is None:
            return None
        piece = hb.held
        self._unhook(hb)
        if piece in self.pieces:
            v = velocity if velocity is not None else tuple(piece.body.velocity)
            piece.body.velocity = self.limit_throw(v) if throw else (0.0, 0.0)
        return piece

    def _unhook(self, hb: HandBody) -> None:
        if hb.joint is not None:
            self.space.remove(hb.joint)
        hb.joint, hb.held = None, None

    def _steer_hands(self, dt: float) -> None:
        # Velocity, not position: collisions then transfer the hand's momentum. Reaching the
        # target over `hand_follow_s` (about one camera frame) keeps motion continuous between
        # 30 FPS camera updates at 60 FPS rendering.
        follow = max(dt, self.cfg.hand_follow_s)
        cap = self.cfg.max_speed * self.size[1]
        for hb in self.hands.values():
            x, y = hb.body.position
            vx, vy = (hb.target[0] - x) / follow, (hb.target[1] - y) / follow
            speed = math.hypot(vx, vy)
            if speed > cap:
                vx, vy = vx * cap / speed, vy * cap / speed
            hb.body.velocity = (vx, vy)

    # --- simulation ---------------------------------------------------------------------
    def step(self, dt: float) -> int:
        """Advance by `dt` in fixed steps; returns the number of substeps run."""
        self.impacts = []
        self._steer_hands(dt)
        ts, n = self.timestep, 0
        self._acc = min(self._acc + dt, self.cfg.max_substeps * ts)
        while self._acc >= ts - 1e-9 and n < self.cfg.max_substeps:
            self.space.step(ts)
            self._acc -= ts
            n += 1
        for p in [p for p in self.pieces if p.fade is not None]:
            p.fade += dt
            if p.fade >= self.cfg.fade_s:
                self.remove(p)
        # Anything that escaped (e.g. squeezed through a wall) is dropped.
        w, h = self.size
        for p in [
            p
            for p in self.pieces
            if not (-w < p.body.position.x < 2 * w and p.body.position.y < 3 * h)
        ]:
            self.remove(p)
        return n

    def _on_post_solve(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        if not arbiter.is_first_contact:
            return
        a, b = arbiter.shapes
        pts = arbiter.contact_point_set.points
        pos = tuple(pts[0].point_a) if pts else tuple(a.body.position)
        self.impacts.append(
            Impact(
                impulse=arbiter.total_impulse.length,
                pos=(float(pos[0]), float(pos[1])),
                pieces=(self._by_shape.get(a), self._by_shape.get(b)),
            )
        )
