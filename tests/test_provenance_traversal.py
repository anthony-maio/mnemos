"""Provenance must not traverse excluded environments or lose eligible files."""

import hashlib
import glob
import os
from pathlib import Path

import pytest

from benchmarks.local_memory.dev_smoke import _code_provenance


def test_excluded_directories_are_not_scanned(monkeypatch, tmp_path):
    root = tmp_path / ".venv" / "snapshot"
    root.mkdir(parents=True)
    (root / "main.py").write_bytes(b"main = 1\n")
    for name in (".venv", ".git", "__pycache__"):
        (root / "nested" / name).mkdir(parents=True)
        (root / "nested" / name / "irrelevant.py").write_bytes(b"irrelevant = 1\n")
    original = os.scandir

    def guarded(path):
        relative = Path(path).relative_to(root)
        if any(part in {".venv", ".git", "__pycache__"} for part in relative.parts):
            pytest.fail("provenance descended into an excluded directory")
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)
    # Python 3.13 pathlib caches scandir on its globber at import time.
    if hasattr(glob, "_StringGlobber"):
        monkeypatch.setattr(glob._StringGlobber, "scandir", staticmethod(guarded))
    assert _code_provenance(root)[1] == ["main.py"]


def test_provenance_preserves_eligible_paths_and_exact_hash(tmp_path):
    expected = {
        "main.py": b"main = 1\n",
        "pyproject.toml": b"[project]\n",
        "docs/validation/runner.py": b"runner = 2\n",
        "nested/config.TOML": b"value = 3\n",
    }
    for name, value in {
        **expected,
        "nested/readme.md": b"not code",
        ".git/hidden.py": b"ignored",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    paths = sorted(tmp_path / name for name in expected)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(tmp_path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(expected[relative])
        digest.update(b"\0")
    assert _code_provenance(tmp_path) == (
        digest.hexdigest(),
        [p.relative_to(tmp_path).as_posix() for p in paths],
    )


def test_unreadable_eligible_tree_cannot_produce_partial_provenance(monkeypatch, tmp_path):
    (tmp_path / "main.py").write_bytes(b"main=1")
    (tmp_path / "protected").mkdir()
    original = os.scandir

    def guarded(path):
        if Path(path).name == "protected":
            raise PermissionError("unreadable source directory")
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)
    with pytest.raises(PermissionError, match="unreadable source"):
        _code_provenance(tmp_path)
