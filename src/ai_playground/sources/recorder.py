"""Record HandFrames to JSON Lines (optionally gzip). Landmarks only, never images."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import time
from pathlib import Path
from typing import IO, Any

from ai_playground.types import Hand, HandFrame

SCHEMA = 1


def _floats(arr) -> list[float]:
    return [round(v, 4) for v in arr.ravel().tolist()]


def hand_to_json(h: Hand) -> dict[str, Any]:
    return {
        "h": h.handedness,
        "hs": round(h.handedness_score, 4),
        "g": h.gesture,
        "gs": round(h.gesture_score, 4),
        "lm": _floats(h.landmarks),
        "w": _floats(h.world),
    }


def _open(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


class Recorder:
    """Writes the header lazily on the first frame, since it needs the image size."""

    def __init__(
        self, path: Path | str, meta: dict[str, Any] | None = None, note: str = ""
    ) -> None:
        self.path = Path(path)
        self.meta = meta or {}
        self.note = note
        self._f = _open(self.path)
        self._t0: float | None = None
        self.frames = 0

    def _write(self, obj: dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def write(self, frame: HandFrame) -> None:
        if self._t0 is None:
            self._t0 = frame.t_capture
            self._write(
                {
                    "type": "header",
                    "schema": SCHEMA,
                    **self.meta,
                    "size": list(frame.size),
                    "mirrored": True,
                    "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "note": self.note,
                }
            )
        self._write(
            {
                "type": "frame",
                "seq": frame.seq,
                "t": round(frame.t_capture - self._t0, 4),
                "lat": round(frame.t_result - frame.t_capture, 4),
                "hands": [hand_to_json(h) for h in frame.hands],
            }
        )
        self.frames += 1

    def mark(self, label: str = "mark", t: float | None = None) -> None:
        """Ground-truth mark (operator presses Space). Ignored before the first frame."""
        if self._t0 is None:
            return
        t = time.monotonic() if t is None else t
        self._write({"type": "mark", "t": round(t - self._t0, 4), "label": label})

    def close(self) -> None:
        self._f.close()


def default_path(directory: Path | str = "recordings") -> Path:
    return Path(directory) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl.gz")
