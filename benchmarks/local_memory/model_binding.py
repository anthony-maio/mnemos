"""Explicit local model selection and native tokenizer identity for comparisons."""

from dataclasses import dataclass
import re
from typing import Any

import httpx


@dataclass(frozen=True)
class StudyModel:
    name: str
    digest: str
    blob_sha256: str


def select_model(name: object, digest: object, blob: object) -> StudyModel | None:
    if name is None and digest is None and blob is None:
        return None
    if (
        not isinstance(name, str)
        or not name
        or name.strip() != name
        or any(c.isspace() for c in name)
    ):
        raise ValueError("Explicit study model requires a nonempty name and both digests")
    if any(
        not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in (digest, blob)
    ):
        raise ValueError("Explicit study model requires two lowercase SHA-256 digests")
    assert isinstance(digest, str) and isinstance(blob, str)
    return StudyModel(name, digest, blob)


def validate_runtime_binding(model: StudyModel, running: Any, props: Any) -> dict[str, Any]:
    """Require one loaded model and its own native tokenizer with the frozen context."""
    if not isinstance(running, dict) or not isinstance(props, dict):
        raise ValueError("Invalid study runtime identity")
    models = running.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError("Study requires exactly one loaded model")
    loaded = models[0]
    if (
        loaded.get("name") != model.name
        or loaded.get("digest") != model.digest
        or loaded.get("context_length") != 8192
    ):
        raise ValueError("Loaded study model or context differs from binding")
    native_path = props.get("model_path")
    if (
        not isinstance(native_path, str)
        or re.split(r"[/\\]", native_path)[-1] != "sha256-" + model.blob_sha256
    ):
        raise ValueError("Native tokenizer model differs from binding")
    generation = props.get("default_generation_settings")
    if (
        not isinstance(generation, dict)
        or generation.get("n_ctx") != 8192
        or props.get("total_slots") != 1
    ):
        raise ValueError("Native study context or slots differ from binding")
    return {
        "loaded_model": loaded,
        "native_model_path": native_path,
        "native_context_length": 8192,
        "native_slots": 1,
    }


async def verify_runtime_binding(
    client: httpx.AsyncClient, endpoint: str, tokenizer_url: str, model: StudyModel
) -> dict[str, Any]:
    running = await client.get(endpoint + "/api/ps")
    running.raise_for_status()
    props = await client.get(tokenizer_url + "/props")
    props.raise_for_status()
    return validate_runtime_binding(model, running.json(), props.json())
