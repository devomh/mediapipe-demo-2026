"""Main loop, scene manager, hotkeys. Runs on the main thread (macOS requires UI there)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pygame

from ai_playground.config import Config
from ai_playground.gestures.engine import GestureEngine
from ai_playground.render import text
from ai_playground.render.background import Background
from ai_playground.render.hud import Hud, Stats
from ai_playground.render.keys import Hotkey, KeysOverlay
from ai_playground.render.overlay import VisionOverlay
from ai_playground.render.viewport import Viewport
from ai_playground.scenes.base import AppContext, Scene
from ai_playground.scenes.geometry import GeometryScene
from ai_playground.scenes.vision import VisionScene
from ai_playground.sources.recorder import Recorder, default_path
from ai_playground.sources.slot import LatestSlot
from ai_playground.tracking.identity import HandTracker
from ai_playground.types import HandFrame

log = logging.getLogger(__name__)

SCENES: dict[str, Callable[[], Scene]] = {"vision": VisionScene, "geometry": GeometryScene}
# Operator scene keys. `1` is Vision until the Attract scene exists (A.8).
SCENE_KEYS = {pygame.K_1: "vision", pygame.K_2: "geometry"}


class Source(Protocol):
    slot: LatestSlot[HandFrame]
    error: str | None
    finished: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...


@dataclass
class RunOptions:
    scene: str = "vision"
    source_label: str = "LIVE"
    record_path: Path | None = None
    record_note: str = ""
    recording_meta: dict | None = None
    max_frames: int | None = None  # stop after this many rendered frames (smoke tests)
    no_frames_hint: str = ""  # shown if a live source sends nothing within the timeout


class App:
    def __init__(self, cfg: Config, source: Source, opts: RunOptions) -> None:
        self.cfg, self.source, self.opts = cfg, source, opts
        text.reset()
        d = cfg.display
        self.fullscreen = d.fullscreen
        self.surface = self._set_mode()
        pygame.display.set_caption("AI Playground")
        self.viewport = Viewport(self.surface.get_size(), (cfg.camera.width, cfg.camera.height))
        self.background = Background()
        self.ctx = AppContext(
            config=cfg,
            viewport=self.viewport,
            background=self.background,
            switch_scene=self.switch_scene,
        )
        self.tracker = HandTracker(cfg.tracking)
        self.engine = GestureEngine(cfg.gestures)
        self.overlay = VisionOverlay(cfg.gestures.pinch)
        self.stats = Stats(cfg.hud.latency_window_s)
        self.hud = Hud(self.stats, cfg.hud.visible)
        self.keys_overlay = KeysOverlay(cfg.hud.keys_visible)
        self.hotkeys = self._hotkeys()
        self.clock = pygame.time.Clock()
        self.recorder: Recorder | None = None
        self._rec_started = 0.0
        self.scene: Scene | None = None
        self.switch_scene(opts.scene)
        self.running = True
        self.frames_rendered = 0
        self.hands_tracked = 0

    # --- window -------------------------------------------------------------------------
    def _set_mode(self) -> pygame.Surface:
        d = self.cfg.display
        if self.fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=d.display_index)
        return pygame.display.set_mode(
            (d.width, d.height), pygame.RESIZABLE, display=d.display_index
        )

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.surface = self._set_mode()
        self.viewport.update(window_size=self.surface.get_size())

    # --- scenes -------------------------------------------------------------------------
    def switch_scene(self, name: str) -> None:
        if name not in SCENES:
            raise ValueError(f"unknown scene {name!r}; available: {', '.join(SCENES)}")
        if self.scene is not None:
            self.scene.exit()
        self.scene = SCENES[name]()
        self.scene.enter(self.ctx)
        self.overlay.visible = getattr(self.scene, "show_vision", True)

    # --- recording ----------------------------------------------------------------------
    def start_recording(self, path: Path | None = None) -> None:
        path = path or default_path()
        self.recorder = Recorder(path, meta=self.opts.recording_meta, note=self.opts.record_note)
        self._rec_started = time.monotonic()
        log.info("recording to %s", path)

    def stop_recording(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            log.info("recorded %d frames to %s", self.recorder.frames, self.recorder.path)
            self.recorder = None

    # --- input --------------------------------------------------------------------------
    def _hotkeys(self) -> list[Hotkey]:
        """Every operator key, in display order. The keys overlay is generated from this list."""

        def onoff(flag: bool) -> str:
            return "on" if flag else "off"

        def toggle(obj, attr: str):
            return lambda _key: setattr(obj, attr, not getattr(obj, attr))

        def recording(_key: int) -> None:
            if self.recorder is None:
                self.start_recording()
            else:
                self.stop_recording()

        scene_keys = tuple(SCENE_KEYS)
        scene_label = " / ".join(pygame.key.name(k).upper() for k in scene_keys)
        scene_desc = "Scene: " + " / ".join(n.capitalize() for n in SCENE_KEYS.values())
        return [
            Hotkey((pygame.K_k,), "K", "Keys (this list)", toggle(self.keys_overlay, "visible")),
            Hotkey(
                scene_keys,
                scene_label,
                scene_desc,
                lambda key: self.switch_scene(SCENE_KEYS[key]),
                state=lambda: self.scene.name if self.scene else "-",
            ),
            Hotkey(
                (pygame.K_v,),
                "V",
                "Vision overlay",
                toggle(self.overlay, "visible"),
                state=lambda: onoff(self.overlay.visible),
            ),
            Hotkey(
                (pygame.K_l,),
                "L",
                "Landmark numbers",
                toggle(self.overlay, "show_indices"),
                state=lambda: onoff(self.overlay.show_indices),
                available=lambda: self.overlay.visible,
            ),
            Hotkey(
                (pygame.K_h,),
                "H",
                "HUD",
                toggle(self.hud, "visible"),
                state=lambda: onoff(self.hud.visible),
            ),
            Hotkey(
                (pygame.K_b,),
                "B",
                "Background",
                lambda _k: self.background.cycle(),
                state=lambda: self.background.mode,
            ),
            Hotkey(
                (pygame.K_c,),
                "C",
                "Clear shapes",
                lambda _k: self.scene.clear(),
                available=lambda: hasattr(self.scene, "clear"),
            ),
            Hotkey(
                (pygame.K_r,),
                "R",
                "Record landmarks",
                recording,
                state=lambda: onoff(self.recorder is not None),
            ),
            Hotkey(
                (pygame.K_SPACE,),
                "Space",
                "Mark (while recording)",
                lambda _k: self.recorder.mark(),
                available=lambda: self.recorder is not None,
            ),
            Hotkey(
                (pygame.K_f,),
                "F",
                "Fullscreen",
                lambda _k: self.toggle_fullscreen(),
                state=lambda: onoff(self.fullscreen),
            ),
            Hotkey((pygame.K_ESCAPE,), "Esc", "Quit", lambda _k: setattr(self, "running", False)),
        ]

    def _on_key(self, key: int) -> None:
        for hk in self.hotkeys:
            if key in hk.keys and hk.is_available():
                hk.action(key)
                return

    # --- main loop ----------------------------------------------------------------------
    def run(self) -> int:
        if self.opts.record_path is not None:
            self.start_recording(self.opts.record_path)
        self.source.start()
        started = time.monotonic()
        seen_seq = 0
        try:
            while self.running:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.running = False
                    elif ev.type == pygame.KEYDOWN:
                        self._on_key(ev.key)
                    elif ev.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                        self.surface = pygame.display.get_surface()
                        self.viewport.update(window_size=self.surface.get_size())

                if self.source.error:
                    return self.error_screen("Something went wrong", self.source.error)
                if (
                    self.opts.no_frames_hint
                    and self.source.slot.seq == 0
                    and time.monotonic() - started > self.cfg.camera.open_timeout_s
                ):
                    return self.error_screen(
                        "The camera sends no pictures", self.opts.no_frames_hint
                    )

                seen_seq, frame = self.source.slot.get_newer(seen_seq)
                new_frame = frame is not None
                if frame is not None:
                    self._process(frame)

                dt = self.clock.tick(self.cfg.display.target_fps) / 1000.0
                assert self.scene is not None
                self.scene.update(dt)
                self.overlay.update(dt)
                self.scene.draw(self.surface)
                self.overlay.draw(self.surface, self.viewport)
                status = "replay finished" if self.source.finished else ""
                if getattr(self.source, "restarts", 0):
                    status = f"recognizer restarts {self.source.restarts}"
                rec_s = None if self.recorder is None else time.monotonic() - self._rec_started
                self.hud.draw(
                    self.surface, self.opts.source_label, self.hands_tracked, rec_s, status
                )
                self.keys_overlay.draw(self.surface, self.hotkeys)
                pygame.display.flip()

                self.stats.render_fps = self.clock.get_fps()
                if new_frame:
                    now = time.monotonic()
                    self.stats.latency.add(now, now - frame.t_capture)
                self.frames_rendered += 1
                if (
                    self.opts.max_frames is not None
                    and self.frames_rendered >= self.opts.max_frames
                ):
                    break
        finally:
            self.source.stop()
            self.stop_recording()
        return 0

    def _process(self, frame: HandFrame) -> None:
        self.ctx.frame = frame
        self.viewport.update(image_size=frame.size)
        self.engine.size = frame.size
        self.stats.camera.add(frame.t_capture, 0.0)
        self.stats.inference.add(frame.t_capture, frame.t_result - frame.t_capture)
        if self.recorder is not None:
            self.recorder.write(frame)
        tracked = self.tracker.update(frame)
        self.hands_tracked = sum(1 for th in tracked if th.missed == 0)
        events, states = self.engine.update(tracked, frame.t_capture)
        if events and log.isEnabledFor(logging.DEBUG):
            for e in events:
                if not e.type.endswith("_move"):
                    log.debug("event %s", e.to_json())
        assert self.scene is not None
        self.scene.handle(events, states)
        self.overlay.handle(events, states)

    def error_screen(self, title: str, message: str) -> int:
        return show_error(self.surface, self.clock, title, message)


def show_error(surface: pygame.Surface, clock: pygame.time.Clock, title: str, message: str) -> int:
    """Full-screen error with a fix hint. Waits for Esc or window close; returns exit code 1."""
    log.error("%s: %s", title, message)
    text.reset()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return 1
        sw, sh = surface.get_size()
        surface.fill((40, 12, 16))
        text.draw(
            surface,
            title,
            (sw // 2, sh // 3),
            (255, 255, 255),
            40,
            mono=False,
            bold=True,
            anchor="midbottom",
        )
        width = min(900, sw - 80)
        for i, line in enumerate(text.wrap(message, 22, width)):
            text.draw(
                surface,
                line,
                (sw // 2, sh // 3 + 30 + 32 * i),
                (240, 210, 210),
                22,
                mono=False,
                anchor="midtop",
            )
        text.draw(
            surface,
            "Press Esc to quit",
            (sw // 2, sh - 40),
            (180, 150, 150),
            16,
            mono=False,
            anchor="midbottom",
        )
        pygame.display.flip()
        clock.tick(20)
