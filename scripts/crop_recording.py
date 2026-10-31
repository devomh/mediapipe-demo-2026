"""Crop a landmark recording to a time window, e.g. to trim a fixture clip.

    uv run python scripts/crop_recording.py IN OUT --start 12.75 --end 20.9

Times are seconds from the first frame, as `--replay` and the fixture analysis report them.
Frames and Space marks in [start, end) are kept and re-timed to start at 0. The header is kept,
with a note of the crop added.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def crop(src: Path, dst: Path, start: float, end: float) -> tuple[int, int]:
    """Returns (frames kept, marks kept)."""
    with _open(src, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    header = lines[0]
    if header.get("type") != "header":
        raise ValueError(f"{src}: first line must be a header")
    frames = [o for o in lines if o["type"] == "frame" and start <= o["t"] < end]
    if not frames:
        raise ValueError(f"no frames between {start} and {end} s")
    t0 = frames[0]["t"]
    marks = [o for o in lines if o["type"] == "mark" and start <= o["t"] < end]
    note = header.get("note", "")
    header = {**header, "note": f"{note} [cropped {start:g}-{end:g} s of {src.name}]".strip()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _open(dst, "w") as f:
        for obj in [header, *frames, *marks]:
            if obj["type"] != "header":
                obj = {**obj, "t": round(obj["t"] - t0, 4)}
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")
    return len(frames), len(marks)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--start", type=float, default=0.0, help="seconds from the first frame")
    p.add_argument("--end", type=float, default=float("inf"), help="seconds from the first frame")
    args = p.parse_args()
    if args.src.resolve() == args.dst.resolve():
        print("error: write to a different file (keep the original)", file=sys.stderr)
        return 2
    n, m = crop(args.src, args.dst, args.start, args.end)
    print(f"kept {n} frames and {m} marks -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
