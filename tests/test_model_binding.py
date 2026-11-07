"""A comparison must never count with another model's tokenizer."""

import copy
import json

import pytest

from benchmarks.local_memory.model_binding import select_model, validate_runtime_binding
from benchmarks.local_memory.extraction_evaluation_run import RecordedProvider
from tests.test_ollama_structured_reasoning import install_client


@pytest.mark.parametrize(
    "args",
    [
        (None, "a" * 64, None),
        ("qwen", None, "b" * 64),
        ("qwen", "a" * 64, None),
        ("qwen", "A" * 64, "b" * 64),
        (" qwen", "a" * 64, "b" * 64),
        ("qwen", "a" * 63, "b" * 64),
    ],
)
def test_incomplete_or_invalid_model_binding_is_rejected(args):
    with pytest.raises(ValueError):
        select_model(*args)


def identities():
    model = select_model("qwen", "a" * 64, "b" * 64)
    running = {"models": [{"name": "qwen", "digest": "a" * 64, "context_length": 8192}]}
    props = {
        "model_path": "C:\\models\\sha256-" + "b" * 64,
        "default_generation_settings": {"n_ctx": 8192},
        "total_slots": 1,
    }
    return model, running, props


def test_matching_runtime_and_omitted_selection():
    assert select_model(None, None, None) is None
    model, running, props = identities()
    assert validate_runtime_binding(model, running, props)["native_slots"] == 1


@pytest.mark.parametrize(
    "field,value", [("name", "other"), ("digest", "c" * 64), ("context_length", 4096)]
)
def test_loaded_model_mismatch_is_rejected(field, value):
    model, running, props = identities()
    running["models"][0][field] = value
    with pytest.raises(ValueError, match="Loaded study"):
        validate_runtime_binding(model, running, props)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_path", "C:/models/sha256-" + "c" * 64),
        ("total_slots", 2),
        ("default_generation_settings", {"n_ctx": 4096}),
    ],
)
def test_native_tokenizer_mismatch_is_rejected(field, value):
    model, running, props = identities()
    props[field] = value
    with pytest.raises(ValueError, match="Native"):
        validate_runtime_binding(model, running, props)


def test_multiple_loaded_models_are_rejected():
    model, running, props = identities()
    running["models"].append(copy.deepcopy(running["models"][0]))
    with pytest.raises(ValueError, match="exactly one"):
        validate_runtime_binding(model, running, props)


@pytest.mark.asyncio
async def test_recorded_provider_binds_model_and_actual_sampling(tmp_path, monkeypatch):
    requests = install_client(monkeypatch, {"tokens": [1], "response": "ok"})
    provider = RecordedProvider(
        "http://local",
        "http://counter",
        tmp_path,
        model="bound",
        sampling_options={"presence_penalty": 0},
    )
    assert await provider.predict("text") == "ok"
    assert requests[1][1]["model"] == "bound"
    assert requests[1][1]["options"]["presence_penalty"] == 0
    record = json.loads((tmp_path / "provider-requests.jsonl").read_text())
    assert record["provider_settings"]["model"] == "bound"
    assert record["provider_settings"]["sampling_options"] == {"presence_penalty": 0}
    requests.clear()
    with pytest.raises(ValueError, match="bound study tokenizer"):
        await provider.predict("text", model="other")
    assert not requests
