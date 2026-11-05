"""B.0 spike (Phase B spec §8): does a Python audio callback keep up under the real app's load?

Throwaway. A `sounddevice` stream plays one sine whose pitch follows the pitch hand, in any scene
(`--audio-spike`). The callback records its own duration, the gap since the previous callback and
PortAudio's underflow flags; the HUD shows them and the app logs a summary every minute. The real
engine (B.1/B.2) replaces this module.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

RATE = 48_000
HISTORY = 4096  # callbacks kept for the stats (~44 s at 512 samples)

# A minor pentatonic over two octaves from A3, like the Theremin's default scale.
_PENTA = (0, 3, 5, 7, 10)
NOTES = tuple(57 + 12 * o + d for o in range(2) for d in _PENTA) + (81,)
BAND = (0.15, 0.85)  # active band of the screen height (spec §4)


def pitch_for_height(y: float) -> float:
    """Normalized screen y (0 = top) -> MIDI note, hard-snapped to NOTES (higher hand, higher)."""
    lo, hi = BAND
    u = min(1.0, max(0.0, (hi - y) / (hi - lo)))
    return float(NOTES[min(len(NOTES) - 1, int(u * len(NOTES)))])


def targets(hands: Sequence[tuple[float, float, float]]) -> tuple[float | None, float]:
    """(palm x, index-tip y, palm y) per hand -> (MIDI note or None, volume 0..1).

    The rightmost hand on screen is the pitch hand; another hand's palm height sets the volume.
    """
    if not hands:
        return None, 0.0
    order = sorted(hands, key=lambda h: h[0])
    pitch_hand = order[-1]
    volume = 0.7
    if len(order) > 1:
        lo, hi = BAND
        volume = 0.1 + 0.9 * min(1.0, max(0.0, (hi - order[0][2]) / (hi - lo)))
    return pitch_for_height(pitch_hand[1]), volume


@dataclass(frozen=True, slots=True)
class SpikeStats:
    callbacks: int
    underflows: int  # PortAudio output_underflow flags
    other_flags: int  # any other status flag (priming, overflow)
    cb_mean_ms: float  # callback duration over the history
    cb_max_ms: float
    gap_max_ms: float  # largest time between two callbacks over the history
    block_ms: float
    out_latency_ms: float  # PortAudio's reported output latency


class SpikeSynth:
    def __init__(self, blocksize: int = 512, gain: float = 0.3, glide_s: float = 0.06) -> None:
        self.blocksize = blocksize
        self.gain = gain  # master level; 0 still synthesizes, silently (measurements)
        self.glide_s = glide_s
        self.fade_s = 0.3
        # Targets, written by the main thread (plain floats: atomic under the GIL).
        self.target_midi = 69.0
        self.target_gain = 0.0
        # Audio-thread state.
        self._midi = 69.0
        self._gain = 0.0
        self._phase = 0.0
        self._last_t: float | None = None
        self.callbacks = 0
        self.underflows = 0
        self.other_flags = 0
        self._dur = np.zeros(HISTORY)
        self._gap = np.zeros(HISTORY)
        self._stream = None
        self.error: str | None = None

    # --- main thread --------------------------------------------------------------------
    def open(self, device: int | str | None = None) -> bool:
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=RATE,
                channels=2,
                dtype="float32",
                blocksize=self.blocksize,
                latency="low",
                device=device,
                callback=self.callback,
            )
            self._stream.start()
        except Exception as e:  # no device, PortAudio missing: the app goes on silently
            self.error = str(e)
            self._stream = None
            log.warning("audio spike: stream did not open: %s", e)
            return False
        log.info(
            "audio spike: block %d (%.1f ms), output latency %.1f ms, switch interval %.1f ms",
            self.blocksize,
            1000 * self.blocksize / RATE,
            1000 * self._stream.latency,
            1000 * _switch_interval(),
        )
        return True

    def set(self, midi: float | None, volume: float) -> None:
        if midi is None:
            self.target_gain = 0.0
        else:
            self.target_midi = midi
            self.target_gain = volume

    def stats(self) -> SpikeStats:
        n = min(self.callbacks, HISTORY)
        dur, gap = self._dur[:n], self._gap[:n]
        lat = self._stream.latency if self._stream is not None else float("nan")
        return SpikeStats(
            callbacks=self.callbacks,
            underflows=self.underflows,
            other_flags=self.other_flags,
            cb_mean_ms=1000 * float(dur.mean()) if n else float("nan"),
            cb_max_ms=1000 * float(dur.max()) if n else float("nan"),
            gap_max_ms=1000 * float(gap.max()) if n > 1 else float("nan"),
            block_ms=1000 * self.blocksize / RATE,
            out_latency_ms=1000 * lat,
        )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # --- audio thread -------------------------------------------------------------------
    def callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        t0 = time.perf_counter()
        i = self.callbacks % HISTORY
        if self._last_t is not None:
            self._gap[i] = t0 - self._last_t
        self._last_t = t0
        if status:
            if status.output_underflow:
                self.underflows += 1
            else:
                self.other_flags += 1

        target_gain = self.target_gain
        target_midi = self.target_midi
        if self._gain <= 0.0 and target_gain <= 0.0:
            outdata.fill(0.0)
            self._midi = target_midi  # silent: jump, so re-entry doesn't glide (spec §4)
        else:
            if self._gain <= 0.0:
                self._midi = target_midi
            # One-pole glide in semitones, evaluated per sample in closed form.
            a = math.exp(-1.0 / (self.glide_s * RATE))
            decay = a ** np.arange(1, frames + 1)
            midi = target_midi + (self._midi - target_midi) * decay
            self._midi = float(midi[-1])
            inc = (2.0 * math.pi / RATE) * 440.0 * np.exp2((midi - 69.0) / 12.0)
            phase = self._phase + np.cumsum(inc)
            self._phase = float(phase[-1] % (2.0 * math.pi))
            # Gain: linear ramp at most 1/fade_s per second, so there are no clicks.
            step = frames / (self.fade_s * RATE)
            g1 = self._gain + max(-step, min(step, target_gain - self._gain))
            gain = np.linspace(self._gain, g1, frames, endpoint=False)
            self._gain = g1 if g1 > 1e-6 else 0.0
            outdata[:] = (np.sin(phase) * (gain * self.gain))[:, None]

        self._dur[i] = time.perf_counter() - t0
        self.callbacks += 1


def _switch_interval() -> float:
    import sys

    return sys.getswitchinterval()


def hud_line(s: SpikeStats) -> str:
    return (
        f"audio blk {s.block_ms:4.1f} ms  underruns {s.underflows}  "
        f"cb {s.cb_mean_ms:.2f}/{s.cb_max_ms:.2f} ms  gap max {s.gap_max_ms:4.1f} ms"
    )


def log_line(s: SpikeStats, elapsed_s: float) -> str:
    return (
        f"audio spike {elapsed_s:5.0f} s: callbacks {s.callbacks}, underruns {s.underflows}, "
        f"other flags {s.other_flags}, callback mean {s.cb_mean_ms:.3f} ms max "
        f"{s.cb_max_ms:.3f} ms, gap max {s.gap_max_ms:.1f} ms (block {s.block_ms:.1f} ms), "
        f"output latency {s.out_latency_ms:.1f} ms"
    )


# --- separate process (spec §2 fallback) ----------------------------------------------------------
# Shared float slots. The parent writes the targets and the run flag; the child writes the rest.
_MIDI, _GAIN, _RUN, _STATE = 0, 1, 2, 3  # _STATE: 0 starting, 1 running, -1 failed
_STATS = 4  # SpikeStats fields, in order, from here
_SLOTS = _STATS + len(SpikeStats.__slots__)


class ProcessSpike:
    """Same interface as SpikeSynth, but the stream runs in a child process with its own GIL, so
    the main thread's Python work can't delay the callback. Targets and stats go through a shared
    float array (no locks: one writer per slot)."""

    START_TIMEOUT_S = 10.0

    def __init__(self, blocksize: int = 512, gain: float = 0.3) -> None:
        import multiprocessing as mp

        self.blocksize, self.gain = blocksize, gain
        self._ctx = mp.get_context("spawn")  # never fork a process running MediaPipe threads
        self._shared = self._ctx.RawArray("d", _SLOTS)
        self._proc = None
        self.error: str | None = None

    def open(self, device: int | str | None = None) -> bool:
        self._shared[_RUN] = 1.0
        self._proc = self._ctx.Process(
            target=_child_main,
            args=(self._shared, self.blocksize, self.gain, device, _switch_interval()),
            name="audio-spike",
            daemon=True,
        )
        self._proc.start()
        deadline = time.monotonic() + self.START_TIMEOUT_S
        while self._shared[_STATE] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._shared[_STATE] != 1:
            self.error = "audio process did not start (see its log)"
            log.warning("audio spike: %s", self.error)
            self.close()
            return False
        log.info("audio spike: running in process %d", self._proc.pid)
        return True

    def set(self, midi: float | None, volume: float) -> None:
        if midi is None:
            self._shared[_GAIN] = 0.0
        else:
            self._shared[_MIDI] = midi
            self._shared[_GAIN] = volume

    def stats(self) -> SpikeStats:
        v = self._shared[_STATS:_SLOTS]
        return SpikeStats(int(v[0]), int(v[1]), int(v[2]), *v[3:])

    def close(self) -> None:
        if self._proc is not None:
            self._shared[_RUN] = 0.0
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None


class _SharedTargetSynth(SpikeSynth):
    def __init__(self, shared, *args) -> None:
        super().__init__(*args)
        self._shared = shared

    def callback(self, outdata, frames, time_info, status) -> None:
        self.target_midi = self._shared[_MIDI]
        self.target_gain = self._shared[_GAIN]
        super().callback(outdata, frames, time_info, status)


def _child_main(shared, blocksize: int, gain: float, device, switch_interval: float) -> None:
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.setswitchinterval(switch_interval)
    parent = os.getppid()
    synth = _SharedTargetSynth(shared, blocksize, gain)
    if not synth.open(device):
        shared[_STATE] = -1.0
        return
    shared[_STATE] = 1.0
    try:
        while shared[_RUN] == 1.0 and os.getppid() == parent:  # parent gone: stop
            s = synth.stats()
            shared[_STATS:_SLOTS] = [float(getattr(s, f)) for f in SpikeStats.__slots__]
            time.sleep(0.05)
    finally:
        synth.close()
