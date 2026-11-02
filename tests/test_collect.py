"""Collect mode (Air Writing spec §8, W.2): prompts, saving, redo; the merge/grid script."""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ai_playground.ml.strokes import CLASSES, read_samples, resample
from ai_playground.scenes.collect import STYLES, Collector

ROOT = Path(__file__).resolve().parents[1]


def strokes():
    return [resample([(100, 100), (100, 300)], [0.0, 0.5], 3.0)]


def test_plan_asks_every_character_once_per_style():
    c = Collector(Path("unused.jsonl"), "w1", seed=1)
    assert len(c.plan) == 36 * len(STYLES)
    for i in range(len(STYLES)):
        rnd = c.plan[36 * i : 36 * (i + 1)]
        assert Counter(lbl for lbl, _ in rnd) == Counter(CLASSES)
        assert {si for _, si in rnd} == {i}
    assert c.plan[:36] != [(ch, 0) for ch in CLASSES]  # shuffled
    assert c.target[1] == "natural" and c.progress() == "round 1 of 3 · 1 of 36"


def test_add_saves_with_the_prompted_label_and_undo_asks_again(tmp_path):
    path = tmp_path / "s.jsonl"
    c = Collector(path, "w1", seed=2)
    first = c.target
    c.add(strokes(), 720)
    second = c.target
    c.add(strokes(), 720)
    header, saved = read_samples(path)
    assert [s.label for s in saved] == [first[0], second[0]]
    assert header["writer"] == "w1" and saved[0].style == "natural"
    assert c.undo().label == second[0] and c.target == second
    assert len(read_samples(path)[1]) == 1


def test_collection_finishes(tmp_path):
    c = Collector(tmp_path / "s.jsonl", "w1", styles=STYLES[:1], classes="AB", seed=0)
    c.add(strokes(), 720)
    c.add(strokes(), 720)
    assert c.done and c.target is None and c.progress() == "all 2 written"


def test_merge_and_grid_script(tmp_path):
    a = Collector(tmp_path / "a.jsonl", "w1", styles=STYLES[:1], seed=0)
    for _ in range(36):
        a.add(strokes(), 720)
    out, png = tmp_path / "eval.jsonl.gz", tmp_path / "grid.png"
    script = str(ROOT / "scripts" / "air_samples.py")
    run = [sys.executable, script]
    r = subprocess.run([*run, "merge", str(out), str(tmp_path / "a.jsonl")], capture_output=True)
    assert r.returncode == 0, r.stderr
    header, merged = read_samples(out)
    assert [s.label for s in merged] == list(CLASSES) and header["writers"] == ["w1"]
    r = subprocess.run([*run, "grid", str(out), "-o", str(png)], capture_output=True)
    assert r.returncode == 0, r.stderr
    assert png.stat().st_size > 1000
    assert np.isfinite(merged[0].strokes[0]).all()


def test_the_committed_evaluation_set_is_complete():
    """eval_v1 (spec §8): one writer, every class once per style, all framable."""
    from ai_playground import config
    from ai_playground.ml.strokes import strokes_to_input

    header, samples = read_samples(ROOT / "tests" / "fixtures" / "strokes" / "eval_v1.jsonl.gz")
    assert header["writers"] == ["w1"] and len(samples) == 108
    assert Counter((s.label, s.style) for s in samples) == Counter(
        (c, style) for c in CLASSES for style, _ in STYLES
    )
    w = config.load().writing
    for s in samples:
        x = strokes_to_input(s.points, w.raster_width, w.ink_threshold, w.fill)
        assert x.max() == 1.0 and 0.05 < x.mean() < 0.4, s.label  # some ink, not a blob
