"""B.0 audio spike (Phase B spec §8): the callback, run offline without an audio device."""

import math

import numpy as np
import sounddevice as sd

from ai_playground.audio import spike
from ai_playground.audio.spike import RATE, SpikeSynth, pitch_for_height, targets


def run(synth: SpikeSynth, blocks: int, frames: int = 512, status=None) -> np.ndarray:
    out = np.zeros((blocks * frames, 2), dtype=np.float32)
    for b in range(blocks):
        synth.callback(
            out[b * frames : (b + 1) * frames], frames, None, status or sd.CallbackFlags()
        )
    return out[:, 0]


def peak_hz(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / RATE)[np.argmax(spec)])


def test_silent_until_a_hand_appears():
    s = SpikeSynth()
    assert not run(s, 4).any()


def test_plays_the_target_pitch_without_clicks():
    s = SpikeSynth(gain=0.5)
    s.set(69.0, 1.0)
    x = run(s, 100)
    assert abs(peak_hz(x[-RATE // 4 :]) - 440.0) < 5
    assert np.abs(x).max() <= 0.5 + 1e-6
    # A 440 Hz sine at 0.5 changes by at most 2*pi*440/48000*0.5 per sample; a click is far more.
    assert np.abs(np.diff(x)).max() < 2 * math.pi * 440 / RATE * 0.5 * 1.01


def test_release_fades_out_and_reentry_jumps_instead_of_gliding():
    s = SpikeSynth()
    s.set(57.0, 1.0)
    run(s, 60)
    s.set(None, 0.0)
    tail = run(s, 40)  # 300 ms fade = 28 blocks of 512
    assert not tail[-512:].any()
    s.set(81.0, 1.0)  # two octaves up
    run(s, 2)
    assert s._midi > 80.9  # jumped while silent; a glide would still be far below


def test_counts_underflows_and_keeps_stats():
    s = SpikeSynth()
    run(s, 3, status=sd.CallbackFlags())
    flags = sd.CallbackFlags()
    flags.output_underflow = True
    run(s, 2, status=flags)
    st = s.stats()
    assert (st.callbacks, st.underflows) == (5, 2)
    assert st.cb_max_ms >= st.cb_mean_ms >= 0
    assert "underruns 2" in spike.hud_line(st)


def test_mapping_rightmost_hand_is_pitch_other_palm_is_volume():
    assert targets([]) == (None, 0.0)
    assert pitch_for_height(0.85) == spike.NOTES[0]  # bottom of the band
    assert pitch_for_height(0.0) == spike.NOTES[-1]  # above the band clamps
    midi, vol = targets([(0.8, 0.2, 0.3)])
    assert vol == 0.7 and midi > spike.NOTES[5]
    midi, vol = targets([(0.8, 0.84, 0.5), (0.2, 0.5, 0.15)])  # left palm at the band top
    assert midi == spike.NOTES[0] and vol == 1.0


def test_process_spike_reports_a_device_that_cannot_open():
    s = spike.ProcessSpike()
    assert not s.open(device=10_000)  # no such device: the child fails, the app goes on
    assert s.error and s._proc is None
