"""Collect mode for Air Writing (spec §8): prompt each character, save what was written.

One writer imitates several (W.2 decision, spec §8): every character is written once per style
round, and each round shuffles the 36 characters. Samples are saved after every read (the whole
file is rewritten, it is small), so quitting midway keeps what was written.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Sequence
from pathlib import Path

from ai_playground.ml.strokes import CLASSES, Sample, Stroke, sample_from_strokes, write_samples

# (style id, what the writer is asked to do)
STYLES: tuple[tuple[str, str], ...] = (
    ("natural", "your normal writing"),
    ("fast", "fast and big, without lifting your fingers between strokes"),
    ("other", "with your other hand, or slanted"),
)


def default_path(directory: Path | str = "recordings/strokes") -> Path:
    return Path(directory) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl")


class Collector:
    def __init__(
        self,
        path: Path,
        writer: str,
        styles: Sequence[tuple[str, str]] = STYLES,
        classes: str = CLASSES,
        seed: int | None = None,
    ) -> None:
        self.path, self.writer, self.styles = path, writer, tuple(styles)
        rng = random.Random(seed)
        self.plan: list[tuple[str, int]] = []  # (label, style index)
        for i in range(len(self.styles)):
            labels = list(classes)
            rng.shuffle(labels)
            self.plan += [(c, i) for c in labels]
        self.samples: list[Sample] = []
        self._per_round = len(classes)

    @property
    def done(self) -> bool:
        return len(self.samples) >= len(self.plan)

    @property
    def target(self) -> tuple[str, str, str] | None:
        """(label, style id, instruction) to write next, or None when finished."""
        if self.done:
            return None
        label, si = self.plan[len(self.samples)]
        return label, *self.styles[si]

    def progress(self) -> str:
        n = len(self.samples)
        if self.done:
            return f"all {n} written"
        rnd, i, per = n // self._per_round, n % self._per_round, self._per_round
        return f"round {rnd + 1} of {len(self.styles)} · {i + 1} of {per}"

    def add(self, strokes: Sequence[Stroke], height_px: float) -> Sample:
        target = self.target
        if target is None:
            raise RuntimeError("collection is finished")
        label, style, _ = target
        smp = sample_from_strokes(label, strokes, height_px, style=style, writer=self.writer)
        self.samples.append(smp)
        self._save()
        return smp

    def undo(self) -> Sample | None:
        """Drop the last sample: its character is asked for again."""
        if not self.samples:
            return None
        smp = self.samples.pop()
        self._save()
        return smp

    def _save(self) -> None:
        header = {
            "kind": "air_writing_samples",
            "writer": self.writer,
            "styles": dict(self.styles),
            "classes": CLASSES,
        }
        write_samples(self.path, self.samples, header)
