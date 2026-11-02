"""EMNIST ByClass -> digits + uppercase, 36 classes (Air Writing spec §5.2).

    python prepare_data.py [--out /content/data] [--zip path/to/gzip.zip]

Downloads NIST's EMNIST archive (pinned URL; MD5 as published by torchvision, SHA-256 printed
for the model card), or uses one downloaded by hand with `--zip` (NIST's server may refuse
scripted downloads; get `gzip.zip` from the EMNIST page in a browser). Keeps the ByClass
split, drops the 26 lowercase classes, fixes the transposed orientation, and writes
`emnist36.npz` (uint8 images, int64 labels, 0-9 then A-Z).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
MD5 = "58c8d27c78d21e728a6bc7b3cc06412e"  # torchvision.datasets.EMNIST
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _idx(raw: bytes) -> np.ndarray:
    """IDX format: magic (type, ndim), big-endian dims, then the data."""
    data = gzip.decompress(raw)
    ndim = data[3]
    dims = np.frombuffer(data, ">i4", count=ndim, offset=4)
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/content/data"))
    ap.add_argument("--zip", type=Path, help="EMNIST gzip.zip downloaded by hand")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    zpath = a.zip or a.out / "emnist_gzip.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    md5 = _digest(zpath, "md5")
    if md5 != MD5:
        raise SystemExit(f"MD5 mismatch: {md5} (expected {MD5}); delete {zpath} and retry")
    sha = _digest(zpath, "sha256")
    print(f"archive ok: {zpath.stat().st_size / 1e6:.0f} MB, sha256 {sha}")

    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zpath) as z:
        names = {Path(n).name: n for n in z.namelist()}
        mapping = z.read(names["emnist-byclass-mapping.txt"]).decode().split("\n")
        ascii_of = {int(k): chr(int(v)) for k, v in (ln.split() for ln in mapping if ln.strip())}
        assert "".join(ascii_of[i] for i in range(36)) == CLASSES, "unexpected ByClass mapping"
        for split in ("train", "test"):
            x = _idx(z.read(names[f"emnist-byclass-{split}-images-idx3-ubyte.gz"]))
            y = _idx(z.read(names[f"emnist-byclass-{split}-labels-idx1-ubyte.gz"])).astype(np.int64)
            keep = y < 36  # 0-9, A-Z; 36-61 are lowercase
            # EMNIST stores each image transposed.
            out[f"x_{split}"] = np.ascontiguousarray(x[keep].transpose(0, 2, 1))
            out[f"y_{split}"] = y[keep]
            counts = np.bincount(y[keep], minlength=36)
            print(
                f"{split}: {keep.sum()} of {len(y)} images; per class min {counts.min()} "
                f"({CLASSES[counts.argmin()]}), max {counts.max()} ({CLASSES[counts.argmax()]})"
            )
    np.savez_compressed(a.out / "emnist36.npz", **out)
    meta = {
        "url": URL,
        "md5": md5,
        "sha256": sha,
        "split": "byclass",
        "classes": CLASSES,
        "n_train": int(len(out["y_train"])),
        "n_test": int(len(out["y_test"])),
    }
    (a.out / "emnist36.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
