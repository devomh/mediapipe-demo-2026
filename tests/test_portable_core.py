"""Architecture §4: the portable core imports only the standard library and NumPy.

It may also import other portable modules and the plain-data modules `types` and `config`.
"""

import ast
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "ai_playground"
PORTABLE = [
    *sorted((PKG / "gestures").glob("*.py")),
    PKG / "ml" / "strokes.py",
    PKG / "ml" / "charnet.py",
    PKG / "ml" / "features.py",
    PKG / "ml" / "pca.py",
    PKG / "audio" / "scales.py",
]
ALLOWED_INTERNAL = {"ai_playground.types", "ai_playground.config", "ai_playground.gestures"}
ALLOWED_THIRD_PARTY = {"numpy"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _ok(module: str) -> bool:
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY:
        return True
    return any(module == m or module.startswith(m + ".") for m in ALLOWED_INTERNAL)


@pytest.mark.parametrize("path", [p for p in PORTABLE if p.exists()], ids=lambda p: p.name)
def test_portable_imports(path):
    bad = [m for m in _imports(path) if not _ok(m)]
    assert not bad, f"{path.relative_to(PKG)} imports non-portable modules: {bad}"


def test_internal_imports_are_themselves_portable():
    for name in ("types.py", "config.py"):
        bad = [m for m in _imports(PKG / name) if not _ok(m)]
        assert not bad, f"{name} imports {bad}"


def test_only_recognizer_imports_mediapipe():
    users = [
        p.relative_to(PKG).as_posix()
        for p in PKG.rglob("*.py")
        if any(m.split(".")[0] == "mediapipe" for m in _imports(p))
    ]
    assert users == ["tracking/recognizer.py"]
