"""Download MediaPipe models into assets/models/ from pinned, versioned URLs and verify SHA-256.

uv run python scripts/fetch_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "assets" / "models"

MODELS = {
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task",
        "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, url: str, expected: str) -> bool:
    dest = MODELS_DIR / name
    if dest.is_file() and sha256(dest) == expected:
        print(f"ok       {name} (already present)")
        return True
    print(f"fetching {name} from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        print(f"error    {name}: SHA-256 mismatch\n  expected {expected}\n  got      {got}")
        return False
    tmp.replace(dest)
    print(f"ok       {name}")
    return True


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = [fetch(name, url, sha) for name, (url, sha) in MODELS.items()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
