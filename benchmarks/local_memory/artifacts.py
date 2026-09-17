"""Truthful, content-addressed experiment artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import tempfile
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = "local-memory-artifact-v1"


def file_digest(path: Path) -> dict[str, Any]:
    """Return byte count and SHA-256 for an existing artifact input."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def environment_metadata() -> dict[str, str]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sqlite": sqlite3.sqlite_version,
    }
    for package in ("numpy", "pydantic", "httpx", "sqlite-vec"):
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            metadata[package] = "not_installed"
    return metadata


def build_not_run_artifact(
    *,
    dataset_path: Path,
    dataset_digest: str,
    profile: str,
    provider: str,
    repetitions: int,
    config: dict[str, Any],
    reason: str = "No experiment adapter was supplied; real-model evaluation has not run.",
) -> dict[str, Any]:
    """Build an explicit pending artifact without inventing measurements."""
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "not_run",
        "gates": {"real_model": "NOT RUN", "human_acceptance": "NOT RUN"},
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_digest,
            "bytes": dataset_path.stat().st_size,
            "source_label": "synthetic_seed",
            "advertised_as_real": False,
        },
        "profile": profile,
        "provider": provider,
        "repetitions_requested": repetitions,
        "config": config,
        "config_sha256": json_digest(config),
        "environment": environment_metadata(),
        "measurements": {},
        "raw_outputs": [],
    }


class ArtifactRecorder:
    """Write a complete run report and its raw-output files atomically."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def write(self, artifact: dict[str, Any]) -> Path:
        payload = dict(artifact)
        payload.setdefault("schema_version", ARTIFACT_SCHEMA_VERSION)
        payload.setdefault("written_at", datetime.now(timezone.utc).isoformat())
        _atomic_write_json(self.output, payload)
        return self.output

    def write_jsonl(self, relative_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Write raw observations and return the measured file metadata."""
        target = self.output.parent / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n" for row in rows
        )
        _atomic_write_text(target, content)
        return file_digest(target)

    def write_run(self, artifact: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
        """Write raw observations first, then attach their measured file digest."""
        # Profile reports commonly share one output directory. Keep each raw
        # stream beside its report so a later profile cannot overwrite an
        # earlier digest.
        raw_metadata = self.write_jsonl(f"{self.output.stem}.raw-outputs.jsonl", rows)
        payload = dict(artifact)
        payload["raw_outputs_file"] = raw_metadata
        return self.write(payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
