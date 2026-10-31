"""Magic Geometry, A.3: hands as physical things.

- Open palm held for `paddle_hold_s`: a paddle that pushes and bats shapes; its ring fades in
  while it arms. It needs MediaPipe's debounced Open_Palm (a half-open hand right after a pinch
  counted 5 fingers and batted away the new shape) and waits `paddle_after_pinch_s` after a
  pinch. The hold keeps an open hand that is only repositioning from batting shapes.
- Grab, two ways: a fist on or near a shape, or a pinch on a shape. Natural grabs (fingers
  curled, thumb on the index) are geometrically pinches, so what is under the hand decides:
  a pinch on a shape grabs it, a pinch in empty space makes a new shape (the scene decides and
  calls `try_grab`). Releasing the fist or pinch drops the shape, or throws it.
- Both fists held for `clear_hold_s`: every shape bursts away.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ai_playground.config import GeometryConfig
from ai_playground.gestures.engine import HandState
from ai_playground.physics.world import Piece, World
from ai_playground.render.viewport import Viewport
from ai_playground.types import GestureEvent


@dataclass(slots=True)
class Grab:
    piece: Piece
    by: str  # "fist" or "pinch": which gesture's end releases it


class HandPlay:
    def __init__(
        self,
        world: World,
        viewport: Viewport,
        cfg: GeometryConfig,
        busy: Callable[[int], bool],
        on_clear: Callable[[], None],
    ) -> None:
        self.world, self.vp, self.cfg = world, viewport, cfg
        self.busy = busy  # hand is making a shape (pinch owner or stretch partner)
        self.on_clear = on_clear
        self.grabbed: dict[int, Grab] = {}  # hand id -> what it holds
        self.fist_pos: dict[int, tuple[float, float]] = {}  # px, hands currently in a fist
        # hand -> (palm px, paddle radius px, 0..1) while an open palm arms into a paddle
        self.arming: dict[int, tuple[tuple[float, float], float, float]] = {}
        self.clear_progress = 0.0  # 0..1 while both fists are held
        self._clear_latched = False  # one clear per double-fist; open a hand to re-arm
        self.grabs = 0  # total grabs, for tests and diagnostics
        self.throws: list[float] = []  # speed (px/s) of each release, for tests and diagnostics
        self._t = 0.0
        # Recent velocities of each held piece. fist_end arrives 70-130 ms after the fingers
        # start to open (MediaPipe label lag + debounce), when a throwing hand is already
        # slowing down, so a throw uses the fastest velocity from the last `throw_lookback_s`.
        self._vel: dict[int, deque[tuple[float, float, float]]] = {}
        self._pinch_end: dict[int, float] = {}  # hand id -> time of its last pinch_end
        self._palm_since: dict[int, float] = {}  # hand id -> when its open palm started arming

    @property
    def held(self) -> dict[int, Piece]:
        return {hid: g.piece for hid, g in self.grabbed.items()}

    def _px(self, pos: tuple[float, float]) -> tuple[float, float]:
        return self.vp.to_px(*pos)

    def _paddle_radius(self, s: HandState) -> float:
        return self.cfg.paddle_scale * s.features.hand_scale_image * self.vp.dest_size[1]

    # --- grabbing -----------------------------------------------------------------------
    def piece_at(self, pos: tuple[float, float], reach_wh: float) -> Piece | None:
        """The closest live, unheld piece whose surface is within `reach_wh` window heights."""
        reach = reach_wh * self.vp.window_size[1]
        taken = {id(g.piece) for g in self.grabbed.values()}
        best, best_d = None, reach
        for p in self.world.pieces:
            if p.fade is not None or id(p) in taken:
                continue
            d = p.shape.point_query(pos).distance  # negative inside the shape
            if d <= best_d:
                best, best_d = p, d
        return best

    def try_grab(self, hand_id: int, pos: tuple[float, float], by: str) -> bool:
        """Grab the piece at `pos` (px) if there is one in reach for this gesture."""
        if hand_id in self.grabbed:
            return False
        reach = self.cfg.grab_reach if by == "fist" else self.cfg.pinch_grab_reach
        piece = self.piece_at(pos, reach)
        if piece is None:
            return False
        self.world.set_hand(hand_id, pos)  # anchor at the grab point
        self.world.grab(hand_id, piece, pos)
        self.grabbed[hand_id] = Grab(piece, by)
        self._vel[hand_id] = deque()
        self.grabs += 1
        return True

    def _release(self, hand_id: int, lost: bool) -> None:
        self.grabbed.pop(hand_id, None)
        piece = self.world.release(hand_id, throw=not lost, velocity=self._peak_velocity(hand_id))
        if piece is not None:
            self.throws.append(math.hypot(*piece.body.velocity))

    def _drop_hand(self, hand_id: int) -> None:
        self.grabbed.pop(hand_id, None)
        self._palm_since.pop(hand_id, None)
        self.world.remove_hand(hand_id)

    # --- per frame ----------------------------------------------------------------------
    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None:
        w = self.world
        # 1. Releases first, before the state pass turns an opened hand into something else.
        for e in events:
            g = self.grabbed.get(e.hand_id)
            if g is not None and e.type == f"{g.by}_end":
                self._release(e.hand_id, lost=e.data.get("reason") == "lost")
            if e.type == "hand_exit":
                self._pinch_end.pop(e.hand_id, None)
                self._drop_hand(e.hand_id)
            elif e.type == "pinch_end":
                self._pinch_end[e.hand_id] = self._t

        # 2. Every visible hand: anchor (holding or fist), paddle (armed open palm), or nothing.
        present = set()
        self.fist_pos, self.arming = {}, {}
        for s in hands:
            present.add(s.id)
            if not s.visible:
                continue  # held through tracking grace: keep its body where it was
            palm = self._px(s.features.palm_center)
            g = self.grabbed.get(s.id)
            if s.fist:
                self.fist_pos[s.id] = palm
            if g is not None and g.by == "pinch":
                self._palm_since.pop(s.id, None)
                w.set_hand(s.id, self._px(s.features.pinch_point))
            elif s.fist:
                self._palm_since.pop(s.id, None)
                w.set_hand(s.id, palm)  # anchor, so a fist grab has something to hang from
            elif s.palm and not s.pinching and not self.busy(s.id) and self._settled(s.id):
                since = self._palm_since.setdefault(s.id, self._t)
                hold = self.cfg.paddle_hold_s
                frac = (self._t - since) / hold if hold > 0 else 1.0
                radius = self._paddle_radius(s)
                if frac >= 1.0:
                    w.set_hand(s.id, palm, radius=radius)
                else:
                    self.arming[s.id] = (palm, radius, frac)
                    w.remove_hand(s.id)
            else:
                self._drop_hand(s.id)
        for hid in [h for h in w.hands if h not in present]:
            self._drop_hand(hid)

        # 3. Fist grabs, now that the fist anchors exist.
        for e in events:
            if e.type == "fist_start" and e.hand_id in w.hands:
                self.try_grab(e.hand_id, self._px(e.pos), "fist")

    def _settled(self, hand_id: int) -> bool:
        """Long enough since this hand's last pinch for the new shape to fall clear."""
        t = self._pinch_end.get(hand_id)
        return t is None or self._t - t >= self.cfg.paddle_after_pinch_s

    def _peak_velocity(self, hand_id: int) -> tuple[float, float] | None:
        samples = self._vel.pop(hand_id, None)
        if not samples:
            return None
        _, vx, vy = max(samples, key=lambda s: math.hypot(s[1], s[2]))
        return (vx, vy)

    def update(self, dt: float) -> None:
        """Call before the physics step."""
        self._t += dt
        for hid, g in self.grabbed.items():
            v = g.piece.body.velocity
            samples = self._vel.setdefault(hid, deque())
            samples.append((self._t, v.x, v.y))
            while samples and self._t - samples[0][0] > self.cfg.throw_lookback_s:
                samples.popleft()
        if len(self.fist_pos) >= 2:
            if not self._clear_latched:
                self.clear_progress = min(1.0, self.clear_progress + dt / self.cfg.clear_hold_s)
                if self.clear_progress >= 1.0:
                    self._clear_latched = True
                    self.grabbed.clear()
                    self.on_clear()
        else:
            self.clear_progress = 0.0
            self._clear_latched = False

    def reset(self) -> None:
        for hid in list(self.world.hands):
            self.world.remove_hand(hid)
        self.grabbed.clear()
        self.fist_pos, self.arming = {}, {}
        self._palm_since.clear()
        self.clear_progress = 0.0
