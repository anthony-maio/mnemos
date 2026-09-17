"""Provider diagnostics retain failure categories without exposing request data."""

import json

import pytest

from mnemos.utils import embeddings, llm
from mnemos.utils.reliability import MnemosConfigurationError

SECRET = "private-prompt-canary Authorization=fixture-only-token"


def _assert_safe_failure(caplog, provider, operation, error):
    assert SECRET not in caplog.text
    events = [json.loads(record.getMessage()) for record in caplog.records]
    failures = [event for event in events if event.get("event") == "mnemos.provider_failure"]
    expected_operations = (
        ["embed_batch", "embed"]
        if provider == "ollama" and operation == "embed_batch"
        else [operation]
    )
    assert [event["operation"] for event in failures] == expected_operations
    for event in failures:
        assert event["provider"] == provider
        assert event["error_type"] == type(error).__name__
        assert "error" not in event


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "openai", "openrouter"])
@pytest.mark.parametrize("error_class", [ValueError, MnemosConfigurationError])
async def test_generation_failure_logs_category_without_sensitive_message(
    monkeypatch, caplog, provider, error_class
):
    error = error_class(SECRET)
    calls = []

    async def fail(*args, **kwargs):
        calls.append(kwargs)
        raise error

    if provider == "ollama":
        target = llm.OllamaProvider()
        monkeypatch.setattr(llm, "call_with_async_retry", fail)
    else:
        target = llm.OpenAIProvider(
            api_key="fixture-only-primary",
            api_key_fallback="fixture-only-fallback",
            base_url=f"https://{provider}.invalid/v1",
        )
        monkeypatch.setattr(target, "_predict_with_key", fail)
    with pytest.raises(error_class) as caught:
        await target.predict(SECRET)
    assert caught.value is error
    assert len(calls) == (
        2 if provider == "openrouter" and error_class is MnemosConfigurationError else 1
    )
    _assert_safe_failure(caplog, provider, "predict", error)


@pytest.mark.parametrize("provider", ["ollama", "openai", "openrouter"])
@pytest.mark.parametrize("operation", ["embed", "embed_batch"])
@pytest.mark.parametrize("error_class", [ValueError, MnemosConfigurationError])
def test_embedding_failure_logs_category_without_sensitive_message(
    monkeypatch, caplog, provider, operation, error_class
):
    error = error_class(SECRET)

    def fail(*args, **kwargs):
        raise error

    if provider == "ollama":
        target = embeddings.OllamaEmbeddingProvider()
        monkeypatch.setattr(embeddings, "call_with_retry", fail)
    else:
        target = embeddings.OpenAIEmbeddingProvider(
            api_key="fixture-only-primary",
            api_key_fallback="fixture-only-fallback",
            base_url=f"https://{provider}.invalid/v1",
        )
        monkeypatch.setattr(target, f"_{operation}_with_key", fail)
    with pytest.raises(error_class) as caught:
        getattr(target, operation)([SECRET] if operation == "embed_batch" else SECRET)
    assert caught.value is error
    _assert_safe_failure(caplog, provider, operation, error)
