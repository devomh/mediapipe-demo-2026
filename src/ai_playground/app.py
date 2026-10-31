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
from ai_playground.render.overlay import VisionOverlay
from ai_playground.render.viewport import Viewport
from ai_playground.scenes.base import AppContext, Scene
from ai_playground.scenes.vision import VisionScene
from ai_playground.sources.recorder import Recorder, default_path
from ai_playground.sources.slot import LatestSlot
from ai_playground.tracking.identity import HandTracker
from ai_playground.types import HandFrame

log = logging.getLogger(__name__)

SCENES: dict[str, Callable[[], Scene]] = {"vision": VisionScene}


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
    def _on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_v:
            self.overlay.visible = not self.overlay.visible
        elif key == pygame.K_h:
            self.hud.visible = not self.hud.visible
        elif key == pygame.K_b:
            self.background.cycle()
        elif key == pygame.K_l:
            self.overlay.show_indices = not self.overlay.show_indices
        elif key == pygame.K_r:
            if self.recorder is None:
                self.start_recording()
            else:
                self.stop_recording()
        elif key == pygame.K_SPACE and self.recorder is not None:
            self.recorder.mark()
        elif key == pygame.K_f:
            self.toggle_fullscreen()

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
                rec_s = None if self.recorder is None else time.monotonic() - self._rec_started
                self.hud.draw(
                    self.surface, self.opts.source_label, self.hands_tracked, rec_s, status
                )
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
