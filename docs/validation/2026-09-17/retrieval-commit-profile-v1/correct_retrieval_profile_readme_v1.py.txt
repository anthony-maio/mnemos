"""Correct one archive statement and refresh its byte/hash index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "docs" / "validation" / "2026-09-17" / "retrieval-commit-profile-v1"
README = ARCHIVE / "README.md"
OLD = "The executed profiler helper and both JSON\nreports are copied byte-for-byte."
NEW = (
    "The executed profiler helper and original profiler report are copied byte-for-byte; "
    "the stage summary and performance reference are derived records."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not ARCHIVE.is_dir():
        raise FileNotFoundError(ARCHIVE)
    text = README.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError("expected exactly one old README phrase")
    README.write_text(text.replace(OLD, NEW), encoding="utf-8", newline="\n")
    archive_copy = ARCHIVE / "correct_retrieval_profile_readme_v1.py.txt"
    shutil.copyfile(Path(__file__), archive_copy)
    files: dict[str, dict[str, object]] = {}
    for path in sorted(ARCHIVE.iterdir()):
        if path.name == "evidence-index.json":
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (ARCHIVE / "evidence-index.json").write_text(
        json.dumps({"files": files}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    index = json.loads((ARCHIVE / "evidence-index.json").read_text(encoding="utf-8"))
    for name, metadata in index["files"].items():
        path = ARCHIVE / name
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"index mismatch: {name}")
    print(json.dumps({"archive": str(ARCHIVE), "file_count": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
