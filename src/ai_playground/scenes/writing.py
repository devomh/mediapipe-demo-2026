"""Air Writing (Air Writing spec): pinch to write, open palm to erase, one fist to let the
neural network read it, two fists to start over.

- Any pinching hand draws its own stroke; releasing the pinch is pen-up, not "done", so
  multi-stroke characters work.
- The eraser arms like Geometry's paddle (hand_rules.PalmArming), so the half-open hand between
  the strokes of an "A" doesn't erase the first one.
- Letting go drags the pinch point as the fingers open, 2-4 frames before pinch_end fires: a
  pen-up drops the ink drawn after the pinch ratio rose above `release_ratio`. Curling into a
  fist opens the pinch too, so this also removes the hook of a fist made right after writing
  (measured on the air_write_digits fixture).
- A fist that closes while the pinch is still held ends the stroke and trims its last
  `tail_trim_s` as well: the fist label lags the curling fingers by 70-130 ms.
- One fist held `read_hold_s` reads the ink (W.4): the strokes are framed into the network's
  28x28 input (ml/strokes), CharNet reads it (ml/charnet), the ink shrinks into the 28x28 grid,
  and the panel shows the answer, the top three with their probabilities, and "0 or O?" for
  look-alike pairs. The answer goes on the text line at the top. Without the model file, a
  read shows the ink and its 28x28 image only.
- Collect mode (T, W.2): a panel asks for a character and a writing style; each read is saved
  with that label (scenes/collect.py). Backspace asks for the last one again.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pygame

from ai_playground.booth import Hints
from ai_playground.config import REPO_ROOT
from ai_playground.gestures.engine import HandState
from ai_playground.ml.charnet import CharNet, Guess
from ai_playground.ml.strokes import (
    Stroke,
    StrokeBuilder,
    bounds,
    char_size,
    cut_after,
    erase,
    strokes_to_input,
    trim_tail,
)
from ai_playground.render import effects, ink, reading
from ai_playground.scenes.base import AppContext
from ai_playground.scenes.collect import Collector, default_path
from ai_playground.scenes.hand_rules import HoldTrigger, PalmArming
from ai_playground.types import GestureEvent

log = logging.getLogger(__name__)

HINTS = {
    "write": "Pinch your fingers and write a number or a capital letter in the air",
    "read": "Make a fist to let the AI read it",
    "erase": "Hold your hand open to erase",
    "clear": "Make two fists and hold them to start over",
}
NO_INK = "Pinch to write first"
TOO_SMALL = "Write it bigger"
NUDGE_S = 2.5  # how long a nudge replaces the hint
INK_FADE_S = 0.5  # ink fades out after a read or a clear
CLEAR_HINT_AT = 3  # characters read before "two fists to start over" is suggested


@dataclass(slots=True, eq=False)  # compared by identity: strokes hold arrays
class Ink:
    stroke: Stroke
    color: tuple[int, int, int]


@dataclass(slots=True)
class Reading:
    """One read: what was written, what the network got, and what it answered."""

    ink: list[Ink]
    image: np.ndarray  # 28x28 network input
    guesses: list[Guess]  # top three
    pair: tuple[str, str] | None  # a look-alike pair to show instead of one answer
    t: float  # scene time of the read (animations)


@cache
def load_model(path: str) -> CharNet | None:
    """The network, loaded once per path. None (and a warning) if the file is missing."""
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    try:
        return CharNet(p)
    except (OSError, KeyError, ValueError) as e:
        log.warning("Air Writing model not loaded (%s): reads will show the ink only", e)
        return None


class WritingScene:
    name = "writing"
    show_vision = False  # V still toggles it

    def __init__(self) -> None:
        self.ctx: AppContext | None = None
        self.ink: list[Ink] = []  # finished strokes of the character being written
        self.active: dict[int, tuple[StrokeBuilder, tuple[int, int, int]]] = {}  # hand -> pen
        self.fading: list[Ink] = []
        self._fade = 0.0  # seconds into fading out `fading`
        self.reads: list[list[Stroke]] = []  # characters read since the last start over
        self.last_input: np.ndarray | None = None  # 28x28 network input of the last read
        self.result: Reading | None = None  # the last read, shown until the next one
        self.text = ""  # the text line: the characters read since the last start over
        self.net: CharNet | None = None
        self.collector: Collector | None = None
        self.hints = Hints(HINTS)
        self.effects_on = True
        self._glow = ink.InkGlow()
        self._t = 0.0  # scene clock (sum of dts)
        self._hands: Sequence[HandState] = ()
        self._firm: dict[int, float] = {}  # pen -> capture t of its last firmly pinched frame
        self._nudge: tuple[str, float] | None = None  # (text, until scene time)
        self.erasers: dict[int, tuple[tuple[float, float], float, float]] = {}  # pos, r, 0..1
        self.fists: dict[int, tuple[float, float]] = {}  # hand -> palm px
        self._arming = PalmArming(0.4, 0.4)
        self.read_hold = HoldTrigger(0.3)
        self.clear_hold = HoldTrigger(1.5)

    # --- lifecycle ----------------------------------------------------------------------
    def enter(self, ctx: AppContext) -> None:
        self.ctx = ctx
        g, w = ctx.config.geometry, ctx.config.writing
        self._arming = PalmArming(g.paddle_hold_s, g.paddle_after_pinch_s)
        self.read_hold = HoldTrigger(w.read_hold_s)
        self.clear_hold = HoldTrigger(g.clear_hold_s)
        self.effects_on = ctx.config.effects.enabled
        self.net = load_model(w.model)
        ctx.background.empty = "gradient"
        ctx.background.mode = "dim"
        ctx.background.dim_brightness = w.background_brightness

    def exit(self) -> None:
        self.ctx = None

    def clear(self) -> None:
        """Ink, pens, the result and the text line (C key, two fists, idle)."""
        self.fading = self.ink + self.fading
        self._fade = 0.0
        self.ink, self.active, self.reads = [], {}, []
        self.last_input = None
        self.result, self.text = None, ""
        self._firm.clear()
        self._nudge = None

    # --- collect mode -------------------------------------------------------------------
    def toggle_collect(self) -> None:
        assert self.ctx is not None
        if self.collector is not None:
            c = self.collector
            log.info("collect mode off: %d samples in %s", len(c.samples), c.path)
            self.collector = None
            return
        self.collector = Collector(default_path(), self.ctx.config.writing.writer)
        log.info("collect mode on: saving to %s", self.collector.path)

    def redo_last(self) -> None:
        if self.collector is not None and self.collector.undo() is not None:
            self._nudge_with("Write that one again")

    def idle_reset(self) -> None:
        self.clear()
        self._arming.reset()
        self.read_hold.reset()
        self.clear_hold.reset()
        self.hints.reset()

    # --- helpers ------------------------------------------------------------------------
    def _h(self) -> float:
        assert self.ctx is not None
        return self.ctx.viewport.window_size[1]

    def _px(self, pos: tuple[float, float]) -> tuple[float, float]:
        assert self.ctx is not None
        return self.ctx.viewport.to_px(*pos)

    def strokes(self) -> list[Stroke]:
        return [i.stroke for i in self.ink]

    def _nudge_with(self, msg: str) -> None:
        self._nudge = (msg, self._t + NUDGE_S)

    # --- pens ---------------------------------------------------------------------------
    def _pen_down(self, hid: int, pos: tuple[float, float], t: float) -> None:
        assert self.ctx is not None
        b = StrokeBuilder(self.ctx.config.writing.resample_step * self._h())
        b.add(self._px(pos), t)
        self.active[hid] = (b, effects.hand_color(hid))
        self._firm[hid] = t

    def _pen_up(self, hid: int, t: float) -> Ink | None:
        assert self.ctx is not None
        b, color = self.active.pop(hid)
        stroke = b.finish()
        firm = self._firm.pop(hid, None)
        if stroke is not None and firm is not None:
            # Drop the release drag: the ink drawn while the fingers were already opening.
            limit = stroke.times[-1] - self.ctx.config.writing.release_trim_max_s
            stroke = cut_after(stroke, max(firm, limit))
        if stroke is None:
            return None
        piece = Ink(stroke, color)
        self.ink.append(piece)
        self.hints.mark("write")
        return piece

    def _on_fist(self, hid: int, t: float) -> None:
        """The hand closed while still pinching: end its stroke, without the curl's hook."""
        assert self.ctx is not None
        if hid not in self.active:
            return  # the pen-up already dropped the release drag
        piece = self._pen_up(hid, t)
        if piece is None:
            return
        trimmed = trim_tail(piece.stroke, self.ctx.config.writing.tail_trim_s)
        if trimmed is None:
            self.ink.remove(piece)
        else:
            piece.stroke = trimmed

    # --- input --------------------------------------------------------------------------
    def handle(self, events: Sequence[GestureEvent], hands: Sequence[HandState]) -> None:
        assert self.ctx is not None
        self._hands = hands
        ratios = {s.id: s.features.pinch_ratio for s in hands}
        release = self.ctx.config.writing.release_ratio
        for e in events:
            hid = e.hand_id
            if e.type == "pinch_start":
                self._pen_down(hid, e.pos, e.t)
            elif e.type == "pinch_move":
                if hid in self.active:
                    self.active[hid][0].add(self._px(e.pos), e.t)
                    if ratios.get(hid, 0.0) < release:
                        self._firm[hid] = e.t
            elif e.type == "pinch_end":
                if hid in self.active:
                    self._pen_up(hid, e.t)
                self._arming.pinch_ended(hid, self._t)
            elif e.type == "fist_start":
                self._on_fist(hid, e.t)
            elif e.type == "hand_exit":
                if hid in self.active:
                    self._pen_up(hid, e.t)
                self._arming.forget(hid)

        # Erasers and fists, from each visible hand's state.
        g, w = self.ctx.config.geometry, self.ctx.config.writing
        self.erasers, self.fists = {}, {}
        for s in hands:
            if not s.visible:
                continue
            palm = self._px(s.features.palm_center)
            if s.fist:
                self.fists[s.id] = palm
            if s.palm and not s.pinching and s.id not in self.active:
                frac = self._arming.update(s.id, self._t)
                if frac is not None:
                    dest_h = self.ctx.viewport.dest_size[1]
                    radius = w.eraser_scale * g.paddle_scale * s.features.hand_scale_image * dest_h
                    self.erasers[s.id] = (palm, radius, min(1.0, frac))
            else:
                self._arming.stop(s.id)
        for pos, radius, frac in self.erasers.values():
            if frac >= 1.0:
                self._erase(pos, radius)

    def _erase(self, pos: tuple[float, float], radius: float) -> None:
        out, changed = [], False
        for piece in self.ink:
            parts = erase([piece.stroke], pos, radius)
            if len(parts) == 1 and parts[0] is piece.stroke:
                out.append(piece)  # untouched
            else:
                changed = True
                out += [Ink(part, piece.color) for part in parts]
        if changed:
            self.ink = out
            self.hints.mark("erase")

    # --- frame --------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        self._t += dt
        if self.fading:
            self._fade += dt
            if self._fade >= INK_FADE_S:
                self.fading, self._fade = [], 0.0
        if self._nudge is not None and self._t >= self._nudge[1]:
            self._nudge = None
        if self.clear_hold.update(dt, len(self.fists) >= 2):
            self.hints.mark("clear")
            self.clear()
        if self.read_hold.update(dt, len(self.fists) == 1):
            self.read()

    def read(self) -> None:
        """Read the ink. W.1 has no model: the character is stored and the ink fades away."""
        assert self.ctx is not None
        strokes = self.strokes()
        if not strokes:
            self._nudge_with(NO_INK)
            return
        if char_size(strokes) < self.ctx.config.writing.min_char_size * self._h():
            self._nudge_with(TOO_SMALL)
            return
        self.reads.append(strokes)
        self.hints.mark("read")
        w, net = self.ctx.config.writing, self.net
        meta: dict = {}
        if net is not None:  # frame exactly as the network was trained
            x = strokes_to_input(strokes, net.raster_width, net.ink_threshold, net.fill)
            guesses = net.top(x, 3)
            pair = reading.lookalike(guesses, w.confusable_pairs, w.confusable_p)
            self.text = (self.text + guesses[0].label)[-max(1, w.text_max) :]
            self.result = Reading(self.ink, x, guesses, pair, self._t)
            meta = {"read": guesses[0].label, "p": round(guesses[0].p, 3)}
            log.info(
                "read %d: %s", len(self.reads), " ".join(f"{g.label} {g.p:.2f}" for g in guesses)
            )
        else:
            x = strokes_to_input(strokes, w.raster_width, w.ink_threshold, w.fill)
            self.fading = self.ink + self.fading  # no model: the ink just fades
            self._fade = 0.0
            log.info("read %d: %d strokes (no model)", len(self.reads), len(strokes))
        self.last_input = x
        saved = None
        if self.collector is not None and not self.collector.done:
            saved = self.collector.add(strokes, self._h(), meta)
        self.ink = []
        self._nudge = None
        if saved is not None:
            self._nudge_with(f"Saved {saved.label}")

    def hint(self) -> str | None:
        if self._nudge is not None:
            return self._nudge[0]
        applicable = ["write"]
        if self.ink or self.active:
            applicable += ["read", "erase"]
        if len(self.reads) >= CLEAR_HINT_AT:
            applicable.append("clear")
        return self.hints.first(applicable)

    def area(self) -> pygame.Rect:
        assert self.ctx is not None
        w, h = self.ctx.viewport.window_size
        side = round(self.ctx.config.writing.area_size * h)
        return pygame.Rect((w - side) // 2, (h - side) // 2 - round(0.03 * h), side, side)

    def draw(self, surface: pygame.Surface) -> None:
        assert self.ctx is not None
        frame = self.ctx.frame
        self.ctx.background.draw(surface, self.ctx.viewport, frame.image if frame else None)
        ink.draw_area(surface, self.area())
        width = self.ctx.config.writing.ink_width * self._h()
        glow = self._glow if self.effects_on else None
        fade = max(0.0, 1.0 - self._fade / INK_FADE_S)
        for piece in self.fading:
            ink.draw_ink(surface, piece.stroke.points, piece.color, width, fade, glow)
        for piece in self.ink:
            ink.draw_ink(surface, piece.stroke.points, piece.color, width, 1.0, glow)
        for b, color in self.active.values():
            ink.draw_ink(surface, b.points, color, width, 1.0, glow)
        if self.result is not None:
            self._draw_reading(surface, self.result, width, glow)
        elif self.reads:  # no model
            ink.draw_thumbnail(surface, self.reads[-1], "You wrote", image=self.last_input)
        newest = self._t - self.result.t if self.result is not None else 1.0
        reading.draw_text_line(surface, self.text, newest)
        if self.collector is not None:
            ink.draw_collect_prompt(surface, self.collector.target, self.collector.progress())
        self._draw_cursors(surface)

    def _draw_reading(self, surface: pygame.Surface, r: Reading, width: float, glow) -> None:
        """The panel and, while the read is young, the ink flying into the 28x28 grid."""
        age = self._t - r.t
        lay = reading.draw_result(surface, r.guesses, r.pair, r.image, age)
        if age >= reading.SHRINK_S or not r.ink:
            return
        b = bounds([i.stroke for i in r.ink])
        assert b is not None and self.net is not None
        k = age / reading.SHRINK_S
        for piece in r.ink:
            pts, scale = reading.shrink(piece.stroke.points, b, lay.grid, self.net.fill, k)
            ink.draw_ink(surface, pts, piece.color, max(1.0, width * scale), 1.0 - 0.5 * k, glow)

    def _draw_cursors(self, surface: pygame.Surface) -> None:
        assert self.ctx is not None
        vp = self.ctx.viewport
        for s in self._hands:
            if not s.visible:
                continue
            color = effects.hand_color(s.id)
            if s.id in self.erasers:
                pos, radius, frac = self.erasers[s.id]
                effects.draw_paddle(surface, pos, radius, strength=frac, color=color)
            elif s.pinching or s.id in self.active:
                effects.draw_pinch_dot(surface, vp.to_px(*s.features.pinch_point), color)
            elif s.id not in self.fists:
                effects.draw_hand_marker(surface, vp.to_px(*s.features.palm_center), color)
        if len(self.fists) == 1:
            pos = next(iter(self.fists.values()))
            effects.draw_progress_ring(surface, pos, self.read_hold.progress)
        elif len(self.fists) >= 2:
            for pos in self.fists.values():
                effects.draw_progress_ring(surface, pos, self.clear_hold.progress)
