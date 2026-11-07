"""Apply and roll back the private host patch plan on protected disposable copies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def digest(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


def read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def apply_additions(original: bytes | None, patch: str, expected_sha: str) -> bytes:
    """Reconstruct zero-context additions while retaining the original line endings."""
    before = original or b""
    text = before.decode("utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("Unsupported original without a final newline")
    newline = "\r\n" if "\r\n" in text else "\n"
    source = text.splitlines(keepends=True)
    lines = patch.splitlines()
    if len(lines) < 3 or not lines[0].startswith("--- ") or not lines[1].startswith("+++ "):
        raise ValueError("Missing patch headers")
    output: list[str] = []
    cursor = 0
    added = 0
    index = 2
    while index < len(lines):
        match = re.fullmatch(r"@@ -(\d+),0 \+(\d+),(\d+) @@", lines[index])
        if match is None:
            raise ValueError("Only explicit zero-context addition hunks are supported")
        offset, destination, count = map(int, match.groups())
        if not cursor <= offset <= len(source) or destination != offset + added + 1 or count <= 0:
            raise ValueError("Patch offsets or counts differ")
        additions = lines[index + 1 : index + 1 + count]
        if len(additions) != count or any(not line.startswith("+") for line in additions):
            raise ValueError("Hunk is not additions only")
        output.extend(source[cursor:offset])
        output.extend(line[1:] + newline for line in additions)
        cursor = offset
        added += count
        index += count + 1
    output.extend(source[cursor:])
    result = "".join(output).encode("utf-8")
    if digest(result) != expected_sha:
        raise ValueError("Reconstructed bytes differ from the reviewed proposal")
    return result


def guarded_replace(
    path: Path, *, expected: bytes | None, replacement: bytes | None, root: Path
) -> None:
    """Refuse drift before modifying a named file inside the disposable root."""
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Rehearsal target escaped its disposable root")
    if read(path) != expected or (expected is None and path.exists()):
        raise ValueError("Configuration drift; retain the current file for review")
    if replacement is None:
        path.unlink(missing_ok=True)
        return
    handle, temporary = tempfile.mkstemp(prefix=".pilot-rehearsal-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(plan_path: Path, output: Path) -> dict[str, Any]:
    from mnemos.service import _protect_private_path

    plan = json.loads(plan_path.read_bytes())
    if plan["status"] != "UNAPPLIED_DRAFT" or plan["activation_authorized"] is not False:
        raise ValueError("Expected the unapplied private draft")
    watched: dict[Path, bytes | None] = {}
    proposals = []
    for change in plan["changes"]:
        target = Path(change["target"])
        original = read(target)
        if digest(original) != change["prior_sha256"] or target.exists() != change["prior_exists"]:
            raise ValueError("Live configuration drifted since the reviewed draft")
        patch = plan_path.parent / change["patch"]
        if digest(patch.read_bytes()) != change["patch_sha256"]:
            raise ValueError("Patch bytes changed")
        proposed = apply_additions(
            original, patch.read_text(encoding="utf-8"), change["proposed_sha256"]
        )
        watched[target] = original
        proposals.append((change, original, proposed))
    for reference in plan["unchanged_reference_files"]:
        target = Path(reference["path"])
        original = read(target)
        if digest(original) != reference["sha256"]:
            raise ValueError("Live reference file drifted since the reviewed draft")
        watched[target] = original
    root = output.resolve()
    if root.exists() or any(
        root == path.resolve() or root.is_relative_to(path.resolve().parent) for path in watched
    ):
        raise ValueError(
            "Use a fresh disposable directory outside the live configuration directories"
        )
    root.mkdir(parents=True)
    _protect_private_path(root)
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "plan_sha256": digest(plan_path.read_bytes()),
        "changes": [],
        "live_configuration_changed": False,
    }
    try:
        for index, (change, original, proposed) in enumerate(proposals, 1):
            directory = root / f"change-{index}"
            directory.mkdir()
            clone = directory / "config"
            backup = directory / "before.bin"
            if original is not None:
                clone.write_bytes(original)
                backup.write_bytes(original)
            guarded_replace(clone, expected=original, replacement=proposed, root=root)
            assert digest(read(clone)) == change["proposed_sha256"]
            restored = backup.read_bytes() if original is not None else None
            guarded_replace(clone, expected=proposed, replacement=restored, root=root)
            assert read(clone) == original and clone.exists() == (original is not None)
            # Deliberate drift belongs only to this disposable copy.
            before_drift = (original or b"") + b"\n# rehearsal-before-drift\n"
            clone.write_bytes(before_drift)
            try:
                guarded_replace(clone, expected=original, replacement=proposed, root=root)
            except ValueError as error:
                assert "Configuration drift" in str(error) and clone.read_bytes() == before_drift
            else:
                raise AssertionError("Apply overwrote drift")
            guarded_replace(clone, expected=before_drift, replacement=proposed, root=root)
            after_drift = proposed + b"\n# rehearsal-after-drift\n"
            clone.write_bytes(after_drift)
            try:
                guarded_replace(clone, expected=proposed, replacement=restored, root=root)
            except ValueError as error:
                assert "Configuration drift" in str(error) and clone.read_bytes() == after_drift
            else:
                raise AssertionError("Rollback overwrote drift")
            assert original is None or backup.read_bytes() == original
            guarded_replace(clone, expected=after_drift, replacement=restored, root=root)
            report["changes"].append(
                {
                    "index": index,
                    "original_existed": original is not None,
                    "original_sha256": digest(original),
                    "applied_sha256": digest(proposed),
                    "restored_sha256": digest(read(clone)),
                    "restored_absence": not clone.exists(),
                    "before_apply_drift_preserved": True,
                    "before_rollback_drift_preserved": True,
                    "backup_unchanged": True,
                }
            )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        unchanged = all(
            read(path) == original and path.exists() == (original is not None)
            for path, original in watched.items()
        )
        report.update(
            live_paths_rechecked=len(watched),
            live_paths_unchanged=unchanged,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if not unchanged:
            report["status"] = "failed"
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "live_paths_rechecked", "live_paths_unchanged")}
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
