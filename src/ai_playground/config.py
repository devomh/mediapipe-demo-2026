"""TOML -> frozen dataclasses. `--config` files deep-merge over `config/default.toml`."""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.toml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int
    mjpg_on_linux: bool
    open_timeout_s: float


@dataclass(frozen=True, slots=True)
class RecognizerConfig:
    model_path: str
    num_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True, slots=True)
class OneEuroConfig:
    min_cutoff: float
    beta: float
    d_cutoff: float


@dataclass(frozen=True, slots=True)
class OneEuroPair:
    image: OneEuroConfig
    world: OneEuroConfig


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    min_hand_size: float
    match_gate: float
    grace_frames: int
    handedness_window: int
    one_euro: OneEuroPair


@dataclass(frozen=True, slots=True)
class PinchConfig:
    enter: float
    exit: float
    min_frames: int
    block_on_fist: bool
    min_index_extension: float


@dataclass(frozen=True, slots=True)
class BuiltinConfig:
    min_score: float
    start_frames: int
    end_frames: int


@dataclass(frozen=True, slots=True)
class FingersConfig:
    tip_pip_ratio: float
    thumb_tip_pinky_dist: float


@dataclass(frozen=True, slots=True)
class VelocityConfig:
    window_s: float


@dataclass(frozen=True, slots=True)
class GesturesConfig:
    pinch: PinchConfig
    builtin: BuiltinConfig
    fingers: FingersConfig
    velocity: VelocityConfig


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    width: int
    height: int
    fullscreen: bool
    display_index: int
    target_fps: int
    ui_scale: float  # text, panels and cursors: this x (window height / 720)


@dataclass(frozen=True, slots=True)
class HudConfig:
    visible: bool
    keys_visible: bool
    latency_window_s: float


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    gravity_factor: float
    steps_per_second: int
    max_substeps: int
    elasticity: float
    friction: float
    max_bodies: int
    fade_s: float
    sleep_time_s: float
    max_speed: float
    max_rise: float
    hand_follow_s: float


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    min_size: float
    max_size: float
    inflate_s: float
    throw_gain: float
    background_brightness: float
    choose_frames: int
    choose_min_pinch_ratio: float
    stretch_max_size: float
    stretch_gain: float
    paddle_scale: float
    paddle_after_pinch_s: float
    grab_reach: float
    pinch_grab_reach: float
    paddle_hold_s: float
    clear_hold_s: float
    throw_lookback_s: float


@dataclass(frozen=True, slots=True)
class EffectsConfig:
    enabled: bool
    particle_cap: int
    glow_scale: float
    glow_intensity: float
    impact_min_dv: float
    impact_max_particles: int
    burst_particles: int
    release_particles: int


@dataclass(frozen=True, slots=True)
class SoundConfig:
    enabled: bool
    volume: float
    impact_min_dv: float
    impact_full_dv: float
    max_impacts_per_frame: int
    piece_cooldown_s: float
    throw_min_speed: float


@dataclass(frozen=True, slots=True)
class WritingConfig:
    ink_width: float
    resample_step: float
    min_char_size: float
    read_hold_s: float
    tail_trim_s: float
    release_ratio: float
    release_trim_max_s: float
    eraser_scale: float
    area_size: float
    background_brightness: float
    raster_width: float
    ink_threshold: float
    fill: float
    writer: str


@dataclass(frozen=True, slots=True)
class BoothConfig:
    idle_timeout_s: float
    hints: bool
    quit_hold_s: float


@dataclass(frozen=True, slots=True)
class Config:
    camera: CameraConfig
    recognizer: RecognizerConfig
    tracking: TrackingConfig
    gestures: GesturesConfig
    display: DisplayConfig
    hud: HudConfig
    physics: PhysicsConfig
    geometry: GeometryConfig
    effects: EffectsConfig
    sound: SoundConfig
    writing: WritingConfig
    booth: BoothConfig

    def model_path(self) -> Path:
        p = Path(self.recognizer.model_path)
        return p if p.is_absolute() else REPO_ROOT / p


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{path}] must be a table")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown key(s) in [{path}]: {', '.join(sorted(unknown))}")
    missing = names - set(data)
    if missing:
        raise ConfigError(f"missing key(s) in [{path}]: {', '.join(sorted(missing))}")
    kwargs = {}
    for name in names:
        hint, value, sub = hints[name], data[name], f"{path}.{name}" if path else name
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, sub)
        else:
            kwargs[name] = _coerce(hint, value, sub)
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    if isinstance(hint, types.UnionType):
        raise ConfigError(f"{path}: unsupported type")
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is bool and not isinstance(value, bool):
        raise ConfigError(f"{path}: expected true/false, got {value!r}")
    if hint is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConfigError(f"{path}: expected an integer, got {value!r}")
    if not isinstance(value, hint):
        raise ConfigError(f"{path}: expected {hint.__name__}, got {value!r}")
    return value


def from_dict(data: dict[str, Any]) -> Config:
    return _build(Config, data, "")


def load(*override_paths: Path | str, overrides: dict[str, Any] | None = None) -> Config:
    """Load the defaults, then merge each override file and finally the `overrides` dict."""
    with DEFAULT_CONFIG.open("rb") as f:
        data = tomllib.load(f)
    for p in override_paths:
        with Path(p).open("rb") as f:
            data = deep_merge(data, tomllib.load(f))
    if overrides:
        data = deep_merge(data, overrides)
    return from_dict(data)
