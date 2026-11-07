"""Reasoning studies reserve context and record the actual requested settings."""

import json

import pytest

from benchmarks.local_memory.extraction_evaluation_run import RecordedProvider
from tests.test_ollama_structured_reasoning import install_client


def test_constructor_rejects_overlapping_input_and_output_allowances(tmp_path) -> None:
    with pytest.raises(ValueError, match="exceed the study context"):
        RecordedProvider("http://local", "http://counter", tmp_path, max_tokens=4096)


@pytest.mark.asyncio
async def test_reasoning_route_and_budget_are_recorded_with_final_text(
    tmp_path, monkeypatch
) -> None:
    raw = {"tokens": [1, 2, 3], "message": {"content": "{}", "thinking": "separate trace"}}
    requests = install_client(monkeypatch, raw)
    provider = RecordedProvider(
        "http://local",
        "http://counter",
        tmp_path,
        timeout=90,
        max_tokens=4096,
        max_input_tokens=3584,
    )
    provider.think = True
    provider.force_chat = True
    assert await provider.predict("same prompt", format={"type": "object"}) == "{}"
    record = json.loads((tmp_path / "provider-requests.jsonl").read_text())
    assert record["provider_settings"] == {
        "think": True,
        "force_chat": True,
        "max_tokens": 4096,
        "timeout": 90,
        "max_input_tokens": 3584,
    }
    assert [url for url, _ in requests] == ["http://counter/tokenize", "http://local/api/chat"]
    assert requests[1][1]["options"]["num_predict"] == 4096
    assert json.loads((tmp_path / "provider-responses.jsonl").read_text())["response"] == raw


@pytest.mark.asyncio
async def test_native_count_above_reserved_allowance_stops_before_inference(
    tmp_path, monkeypatch
) -> None:
    requests = install_client(monkeypatch, {"tokens": [1] * 3585})
    provider = RecordedProvider(
        "http://local", "http://counter", tmp_path, max_tokens=4096, max_input_tokens=3584
    )
    with pytest.raises(ValueError, match="prospective input allowance"):
        await provider.predict("prompt", format={"type": "object"})
    assert len(requests) == 1 and requests[0][0].endswith("/tokenize")
    assert not (tmp_path / "provider-responses.jsonl").exists()


@pytest.mark.asyncio
async def test_per_call_output_override_cannot_overrun_reserved_context(
    tmp_path, monkeypatch
) -> None:
    requests = install_client(monkeypatch, {"tokens": [1]})
    provider = RecordedProvider(
        "http://local", "http://counter", tmp_path, max_tokens=4096, max_input_tokens=3584
    )
    with pytest.raises(ValueError, match="reserved study context"):
        await provider.predict("prompt", max_tokens=8192)
    assert requests == []
