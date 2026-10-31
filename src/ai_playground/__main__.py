"""CLI entry: `uv run ai-playground [--replay f] [--record f] [--scene s] ...`."""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# pygame MUST be imported before cv2/mediapipe. Both wheels bundle SDL2 on macOS (OpenCV via
# FFmpeg's libavdevice); whichever loads first registers its Objective-C classes, and pygame's
# SDL must get its own.
import pygame  # noqa: I001

import argparse
import hashlib
import logging
import sys
from importlib import metadata
from pathlib import Path

from ai_playground import config as config_mod
from ai_playground.config import Config, ConfigError

log = logging.getLogger("ai_playground")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ai-playground", description="Your hands are the controller.")
    p.add_argument("--scene", default="vision", help="scene to start in (default: vision)")
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="TOML overrides merged over config/default.toml (repeatable)",
    )
    p.add_argument("--camera", type=int, help="camera index (overrides [camera].index)")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices 0-5 and exit")
    p.add_argument(
        "--record",
        type=Path,
        metavar="PATH",
        help="record landmarks from startup (.jsonl or .jsonl.gz)",
    )
    p.add_argument("--note", default="", help="note stored in the recording header")
    p.add_argument("--replay", type=Path, metavar="PATH", help="replay a recording, no camera")
    p.add_argument("--loop", action="store_true", help="loop the replay")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--display", type=int, help="display index (overrides [display].display_index)")
    p.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)  # smoke tests
    p.add_argument("--debug", action="store_true", help="debug logging, including events")
    return p.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict:
    o: dict = {}
    if args.camera is not None:
        o.setdefault("camera", {})["index"] = args.camera
    if args.fullscreen:
        o.setdefault("display", {})["fullscreen"] = True
    if args.display is not None:
        o.setdefault("display", {})["display_index"] = args.display
    return o


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def startup_checks(cfg: Config, live: bool) -> list[str]:
    """Problems that stop the app before the window opens, each with a one-line fix."""
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is running, but the "
            "project needs 3.12. Start it with `uv run ai-playground`."
        )
    if live and not cfg.model_path().is_file():
        problems.append(
            f"Model file missing: {cfg.model_path()}. Fix: `uv run python scripts/fetch_models.py`."
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_cameras:
        from ai_playground.sources.camera import list_cameras

        for i, neg in list_cameras():
            print(
                f"camera {i}: " + (f"{neg.width}x{neg.height} @ {neg.fps:.0f} fps" if neg else "-")
            )
        return 0

    try:
        cfg = config_mod.load(*args.config, overrides=_overrides(args))
    except (ConfigError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    live = args.replay is None
    problems = startup_checks(cfg, live)
    if problems:
        for msg in problems:
            print(f"error: {msg}", file=sys.stderr)
        return 1

    from ai_playground.app import SCENES, App, RunOptions, show_error

    if args.scene not in SCENES:
        print(
            f"error: unknown scene {args.scene!r}; available: {', '.join(SCENES)}", file=sys.stderr
        )
        return 2

    meta = {"app": _version("ai-playground"), "mediapipe": _version("mediapipe")}
    if live:
        from ai_playground.sources.camera import Camera, CameraError, platform_hint
        from ai_playground.sources.live import LiveSource
        from ai_playground.tracking.recognizer import Recognizer

        meta["model_sha256"] = _sha256(cfg.model_path())
        camera = Camera(cfg.camera)
        try:
            camera.open()  # main thread, before the window: macOS permission prompt needs it
            camera_error = None
        except CameraError as e:
            camera_error = str(e)
        if camera_error is None:
            source = LiveSource(camera, lambda: Recognizer(cfg.recognizer, cfg.model_path()))
        label = f"LIVE cam {camera.index}"
        hint = (
            f"Camera {camera.index} opened but sent no frames within "
            f"{cfg.camera.open_timeout_s:.0f} s. {platform_hint()} "
            "Try another index with --camera N (see --list-cameras)."
        )
    else:
        from ai_playground.sources.replay import ReplaySource

        camera_error = None
        try:
            source = ReplaySource(args.replay, loop=args.loop, speed=args.speed)
        except (OSError, ValueError) as e:
            print(f"error: cannot replay {args.replay}: {e}", file=sys.stderr)
            return 1
        meta["model_sha256"] = source.clip.header.get("model_sha256", "")
        label = f"REPLAY {args.replay.name}"
        hint = ""

    pygame.init()
    try:
        if camera_error is not None:
            d = cfg.display
            surface = pygame.display.set_mode((d.width, d.height), display=d.display_index)
            return show_error(surface, pygame.time.Clock(), "Camera not available", camera_error)
        opts = RunOptions(
            scene=args.scene,
            source_label=label,
            record_path=args.record,
            record_note=args.note,
            recording_meta=meta,
            max_frames=args.max_frames,
            no_frames_hint=hint,
        )
        return App(cfg, source, opts).run()
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
