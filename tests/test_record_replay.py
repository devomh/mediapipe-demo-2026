import gzip
import json
import time

import numpy as np
import pytest
from synthetic import frame, make_hand

from ai_playground.sources import replay
from ai_playground.sources.recorder import Recorder
from ai_playground.sources.slot import LatestSlot


def write_clip(path, n=30, fps=30.0, t0=1000.0, marks=()):
    rec = Recorder(path, meta={"app": "test"}, note="synthetic")
    for i in range(n):
        hands = [make_hand(pinch=0.1, center=(0.3 + 0.01 * i, 0.5))] if i % 10 else []
        rec.write(frame(i + 1, t0 + i / fps, hands))
    for t in marks:
        rec.mark("pinch", t=t0 + t)
    rec.close()


@pytest.mark.parametrize("name", ["clip.jsonl", "clip.jsonl.gz"])
def test_round_trip(tmp_path, name):
    p = tmp_path / name
    write_clip(p, marks=[0.5])
    clip = replay.load_clip(p)
    assert clip.header["note"] == "synthetic"
    assert clip.header["size"] == [640, 480] and clip.header["mirrored"] is True
    assert len(clip.frames) == 30
    assert clip.frames[0].t_capture == 0.0
    assert clip.frames[1].t_capture == pytest.approx(1 / 30, abs=1e-4)
    assert clip.frames[1].t_result - clip.frames[1].t_capture == pytest.approx(0.015, abs=1e-4)
    assert clip.marks == [replay.Mark(t=0.5, label="pinch")]
    original = make_hand(pinch=0.1, center=(0.31, 0.5))
    got = clip.frames[1].hands[0]
    assert np.allclose(got.landmarks, original.landmarks, atol=1e-4)
    assert np.allclose(got.world, original.world, atol=1e-4)
    assert got.handedness == "Right"
    assert clip.frames[0].hands == ()


def test_gzip_is_actually_gzip_and_never_has_images(tmp_path):
    p = tmp_path / "c.jsonl.gz"
    write_clip(p, n=3)
    with gzip.open(p, "rt") as f:
        lines = [json.loads(line) for line in f]
    assert lines[0]["type"] == "header"
    assert all("image" not in line for line in lines)


def test_mark_before_first_frame_is_ignored(tmp_path):
    p = tmp_path / "c.jsonl"
    rec = Recorder(p)
    rec.mark()
    rec.write(frame(1, 5.0, []))
    rec.close()
    assert replay.load_clip(p).marks == []


def test_replay_source_publishes_on_recorded_timing(tmp_path):
    p = tmp_path / "c.jsonl"
    write_clip(p, n=30)
    src = replay.ReplaySource(p, speed=10.0)
    start = time.monotonic()
    src.start()
    while not src.finished and time.monotonic() - start < 2.0:
        time.sleep(0.005)
    src.stop()
    elapsed = time.monotonic() - start
    seq, last = src.slot.get()
    assert src.finished and seq == 30
    assert 29 / 30 / 10 * 0.8 < elapsed < 1.0
    assert last.image is None


def test_replay_loop_keeps_timestamps_increasing(tmp_path):
    p = tmp_path / "c.jsonl"
    write_clip(p, n=5)
    src = replay.ReplaySource(p, loop=True, speed=20.0)
    src.start()
    seen, stamps, start = 0, [], time.monotonic()
    while len(stamps) < 12 and time.monotonic() - start < 2.0:
        seen, fr = src.slot.get_newer(seen)
        if fr is not None:
            stamps.append(fr.t_capture)
        time.sleep(0.001)
    src.stop()
    assert not src.finished
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_latest_slot_newest_wins():
    s = LatestSlot()
    assert s.get_newer(0) == (0, None)
    s.put("a")
    s.put("b")
    assert s.get_newer(0) == (2, "b")
    assert s.get_newer(2) == (2, None)


def test_crop_recording_keeps_and_retimes_a_window(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "crop_recording", Path(__file__).parents[1] / "scripts" / "crop_recording.py"
    )
    crop_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crop_mod)

    src, dst = tmp_path / "full.jsonl.gz", tmp_path / "crop.jsonl.gz"
    write_clip(src, n=30, marks=[0.2, 0.5, 0.9])  # frames at 0, 1/30, ... 29/30 s
    frames, marks = crop_mod.crop(src, dst, start=0.3, end=0.7)
    clip = replay.load_clip(dst)
    assert frames == len(clip.frames) == 12  # 0.3 <= t < 0.7 at 30 FPS
    assert clip.frames[0].t_capture == 0.0
    assert clip.frames[0].seq == 10  # frame 9 (t = 0.3 s); original sequence numbers are kept
    assert [m.t for m in clip.marks] == [pytest.approx(0.5 - 0.3, abs=1e-3)]
    assert "cropped 0.3-0.7 s of full.jsonl.gz" in clip.header["note"]
