"""Crash-safe checkpoints for long local-model benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = "local-memory-checkpoint-v1"


@dataclass
class Checkpoint:
    run_id: str
    dataset_sha256: str
    config_sha256: str
    profile: str
    repetitions: int
    next_session_index: int = 0
    completed_observation_ids: list[str] = field(default_factory=list)
    completed_observations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    model_sha256: str = ""
    embedding_sha256: str = ""
    options_sha256: str = ""
    code_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Persist a checkpoint with replace semantics so interruption is recoverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.as_dict()
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint(
    path: Path,
    *,
    dataset_sha256: str,
    config_sha256: str,
    profile: str,
    repetitions: int,
    model_sha256: str,
    embedding_sha256: str,
    options_sha256: str,
    code_sha256: str,
    run_id: str | None = None,
) -> Checkpoint:
    """Load only a checkpoint matching the exact experiment identity."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported or malformed checkpoint schema")
    expected = {
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "profile": profile,
        "repetitions": repetitions,
        "model_sha256": model_sha256,
        "embedding_sha256": embedding_sha256,
        "options_sha256": options_sha256,
        "code_sha256": code_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} does not match this experiment")
    next_index = payload.get("next_session_index")
    if not isinstance(next_index, int) or next_index < 0:
        raise ValueError("checkpoint next_session_index must be a non-negative integer")
    completed = payload.get("completed_observation_ids", [])
    if not isinstance(completed, list) or any(not isinstance(item, str) for item in completed):
        raise ValueError("checkpoint completed_observation_ids must be a string list")
    observations = payload.get("completed_observations", [])
    if not isinstance(observations, list) or any(
        not isinstance(item, dict) for item in observations
    ):
        raise ValueError("checkpoint completed_observations must be an object list")
    if len(observations) != len(completed):
        raise ValueError("checkpoint observation data must match completed observation IDs")
    stored_run_id = payload.get("run_id")
    if not isinstance(stored_run_id, str) or not stored_run_id:
        raise ValueError("checkpoint run_id is required")
    if run_id is not None and stored_run_id != run_id:
        raise ValueError("checkpoint run_id does not match this experiment")
    return Checkpoint(
        run_id=stored_run_id,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        profile=profile,
        repetitions=repetitions,
        next_session_index=next_index,
        completed_observation_ids=list(completed),
        completed_observations=list(observations),
        status=str(payload.get("status", "running")),
        model_sha256=model_sha256,
        embedding_sha256=embedding_sha256,
        options_sha256=options_sha256,
        code_sha256=code_sha256,
    )
