"""Replay recorded `.jsonl[.gz]` clips, either directly (tests) or on recorded timing (app)."""

from __future__ import annotations

import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai_playground.sources.slot import LatestSlot
from ai_playground.types import Hand, HandFrame


@dataclass(frozen=True, slots=True)
class Mark:
    t: float
    label: str


@dataclass(slots=True)
class Clip:
    header: dict[str, Any]
    frames: list[HandFrame] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)


def _hand(d: dict[str, Any]) -> Hand:
    return Hand(
        handedness=d["h"],
        handedness_score=float(d["hs"]),
        landmarks=np.asarray(d["lm"], dtype=np.float32).reshape(21, 3),
        world=np.asarray(d["w"], dtype=np.float32).reshape(21, 3),
        gesture=d["g"],
        gesture_score=float(d["gs"]),
    )


def load_clip(path: Path | str) -> Clip:
    """Times in the returned frames are relative to the first frame (t_capture = t)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    clip: Clip | None = None
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "header":
                clip = Clip(header=obj)
            elif clip is None:
                raise ValueError(f"{path}: first line must be a header")
            elif kind == "frame":
                size = tuple(clip.header["size"])
                clip.frames.append(
                    HandFrame(
                        seq=int(obj["seq"]),
                        t_capture=float(obj["t"]),
                        t_result=float(obj["t"]) + float(obj["lat"]),
                        size=size,
                        hands=tuple(_hand(h) for h in obj["hands"]),
                        image=None,
                    )
                )
            elif kind == "mark":
                clip.marks.append(Mark(t=float(obj["t"]), label=obj["label"]))
    if clip is None:
        raise ValueError(f"{path}: empty recording")
    return clip


def load(path: Path | str) -> list[HandFrame]:
    return load_clip(path).frames


class ReplaySource:
    """Publishes a clip's frames on their recorded timing, rebased to time.monotonic()."""

    def __init__(self, path: Path | str, loop: bool = False, speed: float = 1.0) -> None:
        self.clip = load_clip(path)
        if not self.clip.frames:
            raise ValueError(f"{path}: recording has no frames")
        self.loop, self.speed = loop, speed
        self.slot: LatestSlot[HandFrame] = LatestSlot()
        self.error: str | None = None
        self.finished = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="replay-source", daemon=True)

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self.clip.header["size"])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        frames = self.clip.frames
        duration = frames[-1].t_capture + 1.0 / 30.0
        seq = 0
        base = time.monotonic()
        while not self._stop.is_set():
            for fr in frames:
                t_capture = base + fr.t_capture / self.speed
                delay = t_capture - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    return
                seq += 1
                self.slot.put(
                    HandFrame(
                        seq=seq,
                        t_capture=t_capture,
                        t_result=t_capture + (fr.t_result - fr.t_capture),
                        size=fr.size,
                        hands=fr.hands,
                        image=None,
                    )
                )
            if not self.loop:
                self.finished = True
                return
            base += duration / self.speed
