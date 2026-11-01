"""Sound effects for Geometry (Phase A spec §4.1): synthesized with NumPy, played by pygame.mixer.

- `synthesize_bank()` builds every sound as mono float32 in [-1, 1]. Pure NumPy, no files, fixed
  seed, so it is testable without an audio device.
- `ImpactLimiter` picks which collisions get a sound, so a 40-shape pile doesn't buzz.
- `Sfx` owns the mixer. If the mixer can't open (no device, CI), it logs once and every call is
  a no-op: sound never stops the exhibit.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pygame

log = logging.getLogger(__name__)

MIXER_RATE = 48_000
MIXER_BUFFER = 512  # samples, ~11 ms at 48 kHz
NUM_CHANNELS = 16
INFLATE_CHANNEL = 0  # reserved, so impacts never steal it

SIZE_BUCKETS = 5
# Impact pitch per size bucket, smallest shape first: C major pentatonic, C6 down to C5.
BUCKET_NOTES = (84, 81, 79, 76, 72)
# Timbre per shape kind: (octave shift in semitones, partials as (ratio, amp, decay_s), noise attack
# (duration_s, amp, lowpass_hz) or None).
TIMBRES: dict[str, tuple[int, tuple[tuple[float, float, float], ...], tuple | None]] = {
    "circle": (0, ((1.0, 1.0, 0.45), (2.76, 0.3, 0.12), (5.4, 0.1, 0.05)), None),  # glassy
    "triangle": (0, ((1.0, 1.0, 0.3), (2.0, 0.5, 0.2), (3.0, 0.35, 0.12), (4.2, 0.2, 0.06)), None),
    "square": (-12, ((1.0, 1.0, 0.09), (1.58, 0.5, 0.06), (3.9, 0.25, 0.03)), (0.006, 0.5, 4000)),
    "pentagon": (-12, ((1.0, 1.0, 0.18), (3.93, 0.35, 0.05), (9.2, 0.1, 0.02)), (0.004, 0.4, 6000)),
    "bar": (-24, ((1.0, 1.0, 0.22), (2.3, 0.4, 0.08)), (0.015, 0.6, 800)),
}
# Peak level of each sound before the master volume, so the mix is balanced by design.
LEVELS = {"impact": 0.9, "spawn": 0.55, "grab": 0.45, "throw": 0.6, "clear": 0.8, "inflate": 0.3}


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12)


def impact_name(kind: str, bucket: int) -> str:
    return f"impact.{kind}.{bucket}"


def size_bucket(radius_wh: float, smallest: float, largest: float) -> int:
    """0 (smallest) .. SIZE_BUCKETS-1 (largest), log-spaced: size steps sound like pitch steps."""
    if radius_wh <= smallest:
        return 0
    if radius_wh >= largest:
        return SIZE_BUCKETS - 1
    k = math.log(radius_wh / smallest) / math.log(largest / smallest)
    return min(SIZE_BUCKETS - 1, int(k * SIZE_BUCKETS))


# --- synthesis (pure NumPy) ---------------------------------------------------------------------
def _time(rate: int, dur: float) -> np.ndarray:
    return np.arange(int(rate * dur), dtype=np.float64) / rate


def _lowpass(x: np.ndarray, rate: int, cutoff: float) -> np.ndarray:
    """Zero-phase spectral low-pass with a smooth roll-off (no SciPy needed)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    return np.fft.irfft(spec / (1.0 + (f / cutoff) ** 4), len(x))


def _bandpass(x: np.ndarray, rate: int, center: float, octaves: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.maximum(np.fft.rfftfreq(len(x), 1.0 / rate), 1.0)
    gain = np.exp(-0.5 * (np.log2(f / center) / octaves) ** 2)
    return np.fft.irfft(spec * gain, len(x))


def _finish(x: np.ndarray, rate: int, level: float) -> np.ndarray:
    """Click-free edges (1 ms in, 5 ms out), peak-normalized to `level`."""
    x = np.asarray(x, dtype=np.float64).copy()
    n_in, n_out = min(len(x), int(0.001 * rate)), min(len(x), int(0.005 * rate))
    if n_in:
        x[:n_in] *= np.linspace(0.0, 1.0, n_in)
    if n_out:
        x[-n_out:] *= np.linspace(1.0, 0.0, n_out)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0:
        x *= level / peak
    return x.astype(np.float32)


def _sweep(rate: int, t: np.ndarray, f0: float, f1: float, dur: float) -> np.ndarray:
    """Phase of an exponential sweep f0 -> f1 over `dur`, holding f1 afterwards."""
    k = np.clip(t / dur, 0.0, 1.0)
    freq = f0 * (f1 / f0) ** k
    return 2 * np.pi * np.cumsum(freq) / rate


def _impact(rate: int, kind: str, bucket: int, rng: np.random.Generator) -> np.ndarray:
    shift, partials, noise = TIMBRES[kind]
    f = midi_hz(BUCKET_NOTES[bucket] + shift)
    dur = min(0.8, 5 * max(d for _, _, d in partials))
    t = _time(rate, dur)
    x = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d) for r, a, d in partials)
    if noise is not None:
        n_dur, n_amp, cutoff = noise
        burst = _lowpass(rng.standard_normal(len(t)), rate, cutoff)
        burst /= np.max(np.abs(burst)) or 1.0
        x = x + n_amp * burst * np.exp(-t / n_dur)
    return _finish(x, rate, LEVELS["impact"])


def synthesize_bank(rate: int, inflate_s: float, seed: int = 7) -> dict[str, np.ndarray]:
    """Every Geometry sound, mono float32 in [-1, 1] at `rate` Hz."""
    rng = np.random.default_rng(seed)
    bank: dict[str, np.ndarray] = {}

    for kind in TIMBRES:
        for b in range(SIZE_BUCKETS):
            bank[impact_name(kind, b)] = _impact(rate, kind, b, rng)

    # Spawn: a soft pop, sine falling 900 -> 350 Hz.
    t = _time(rate, 0.12)
    bank["spawn"] = _finish(
        np.sin(_sweep(rate, t, 900, 350, 0.06)) * np.exp(-t / 0.04), rate, LEVELS["spawn"]
    )

    # Grab: a short, high click.
    t = _time(rate, 0.03)
    click = np.sin(2 * np.pi * 2000 * t) * np.exp(-t / 0.006)
    click += 0.5 * rng.standard_normal(len(t)) * np.exp(-t / 0.002)
    bank["grab"] = _finish(click, rate, LEVELS["grab"])

    # Throw: band-passed noise whose band rises, with a quick swell and a longer tail.
    t = _time(rate, 0.4)
    noise = rng.standard_normal(len(t))
    low, high = _bandpass(noise, rate, 600, 0.8), _bandpass(noise, rate, 1800, 0.8)
    mix = np.clip(t / 0.25, 0.0, 1.0)
    swell = np.minimum(t / 0.08, 1.0) * np.exp(-np.maximum(t - 0.08, 0.0) / 0.1)
    bank["throw"] = _finish(((1 - mix) * low + mix * high) * swell, rate, LEVELS["throw"])

    # Clear: a falling sweep with a noise burst underneath.
    t = _time(rate, 0.9)
    phase = _sweep(rate, t, 1200, 120, 0.6)
    tone = (np.sin(phase) + 0.3 * np.sin(2 * phase)) * np.exp(-t / 0.35)
    rumble = _lowpass(rng.standard_normal(len(t)), rate, 1500) * np.exp(-t / 0.2)
    rumble /= np.max(np.abs(rumble)) or 1.0
    bank["clear"] = _finish(tone + 0.6 * rumble, rate, LEVELS["clear"])

    # Inflate: a quiet tone rising with the shape's growth (linear over inflate_s), then fading.
    dur = max(inflate_s, 0.2)
    t = _time(rate, dur + 0.15)
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 6 * t)
    freq = 220.0 * 3.0 ** np.clip(t / dur, 0.0, 1.0) * vibrato  # 220 -> 660 Hz
    phase = 2 * np.pi * np.cumsum(freq) / rate
    amp = np.minimum(t / 0.1, 1.0) * np.clip((dur + 0.15 - t) / 0.15, 0.0, 1.0)
    bank["inflate"] = _finish(
        (np.sin(phase) + 0.2 * np.sin(2 * phase)) * amp, rate, LEVELS["inflate"]
    )
    return bank


# --- choosing which impacts sound ---------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Hit:
    key: int  # identifies the sounding piece, for the cooldown
    volume: float  # 0..1
    x: float  # 0..1 across the window, for panning
    name: str  # sound to play


class ImpactLimiter:
    """At most `max_per_frame` impact sounds per frame, loudest first; each piece then stays
    silent for `cooldown_s`, so shapes resting on each other don't buzz."""

    def __init__(self, max_per_frame: int, cooldown_s: float) -> None:
        self.max_per_frame = max_per_frame
        self.cooldown_s = cooldown_s
        self._last: dict[int, float] = {}

    def select(self, hits: Sequence[Hit], t: float) -> list[Hit]:
        loudest: dict[int, Hit] = {}
        for h in hits:
            if h.key not in loudest or h.volume > loudest[h.key].volume:
                loudest[h.key] = h
        chosen: list[Hit] = []
        for h in sorted(loudest.values(), key=lambda h: -h.volume):
            if len(chosen) >= self.max_per_frame:
                break
            if t - self._last.get(h.key, -math.inf) < self.cooldown_s:
                continue
            self._last[h.key] = t
            chosen.append(h)
        if len(self._last) > 256:  # forget pieces that have been quiet for a while
            self._last = {k: v for k, v in self._last.items() if t - v < self.cooldown_s}
        return chosen

    def reset(self) -> None:
        self._last.clear()


# --- playback ----------------------------------------------------------------------------------
class Sfx:
    """Plays the bank through pygame.mixer. Constructed directly it is silent (no device needed);
    `Sfx.open()` opens the mixer."""

    def __init__(self, volume: float = 1.0) -> None:
        self.available = False
        self.muted = False
        self.volume = volume
        self._sounds: dict[str, pygame.mixer.Sound] = {}

    @classmethod
    def open(cls, enabled: bool, volume: float, inflate_s: float) -> Sfx:
        sfx = cls(volume)
        if not enabled:
            return sfx
        try:
            # pygame.init() already opened the mixer with its defaults; reopen with ours.
            pygame.mixer.quit()
            pygame.mixer.init(
                frequency=MIXER_RATE, size=-16, channels=2, buffer=MIXER_BUFFER, allowedchanges=0
            )
            rate, fmt, channels = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(NUM_CHANNELS)
            pygame.mixer.set_reserved(1)
            sfx._sounds = {
                name: pygame.sndarray.make_sound(_to_mixer(x, fmt, channels))
                for name, x in synthesize_bank(rate, inflate_s).items()
            }
        except (pygame.error, ValueError) as e:
            log.warning("sound effects off: %s", e)
            return sfx
        sfx.available = True
        return sfx

    def play(self, name: str, volume: float = 1.0, x: float = 0.5) -> None:
        """Play `name` at `volume` (0..1), panned by `x` (0 = left edge, 1 = right edge)."""
        if not self.available or self.muted:
            return
        snd = self._sounds.get(name)
        if snd is None:
            return
        ch = pygame.mixer.find_channel(True)  # True: steal the oldest if all are busy
        if ch is None:
            return
        left, right = _pan(self.volume * max(0.0, min(1.0, volume)), x)
        ch.set_volume(left, right)
        ch.play(snd)
        ch.set_volume(left, right)  # in case play() reset it

    def start_inflate(self, x: float = 0.5) -> None:
        if not self.available or self.muted:
            return
        ch = pygame.mixer.Channel(INFLATE_CHANNEL)
        left, right = _pan(self.volume, x)
        ch.set_volume(left, right)
        ch.play(self._sounds["inflate"])
        ch.set_volume(left, right)

    def stop_inflate(self) -> None:
        if self.available:
            pygame.mixer.Channel(INFLATE_CHANNEL).fadeout(80)

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        if self.muted and self.available:
            pygame.mixer.stop()

    def close(self) -> None:
        if self.available and pygame.mixer.get_init():
            pygame.mixer.stop()


def _pan(volume: float, x: float) -> tuple[float, float]:
    """Constant-power pan, narrowed so nothing plays from one speaker only."""
    p = 0.5 + 0.6 * (max(0.0, min(1.0, x)) - 0.5)
    return volume * math.cos(p * math.pi / 2), volume * math.sin(p * math.pi / 2)


def _to_mixer(x: np.ndarray, fmt: int, channels: int) -> np.ndarray:
    """Mono float32 -> the mixer's sample format, one column per channel."""
    if fmt == -16:
        samples = np.round(np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    elif fmt == 32:
        samples = x.astype(np.float32)
    else:
        raise ValueError(f"unsupported mixer format {fmt}")
    return np.ascontiguousarray(np.repeat(samples[:, None], channels, axis=1))
