"""
tests/test_embeddings.py — Tests for production embedding providers.
"""

from __future__ import annotations

import asyncio

import pytest

from mnemos.utils import embeddings as embeddings_module


@pytest.mark.asyncio
async def test_embed_text_async_runs_simple_provider_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = embeddings_module.SimpleEmbeddingProvider(dim=8)

    async def _unexpected_to_thread(*args, **kwargs):
        raise AssertionError("SimpleEmbeddingProvider should not be offloaded to a worker thread.")

    monkeypatch.setattr(embeddings_module.asyncio, "to_thread", _unexpected_to_thread)

    vec = await embeddings_module.embed_text_async(provider, "local dev embedding")

    assert len(vec) == 8


@pytest.mark.asyncio
async def test_embed_text_async_offloads_network_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embedding": [0.1, 0.2, 0.3]}

    def _fake_post(url: str, *, json: dict[str, object], timeout: float) -> _DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse()

    async def _fake_to_thread(fn, *args, **kwargs):
        captured["fn_name"] = getattr(fn, "__name__", "<unknown>")
        return fn(*args, **kwargs)

    monkeypatch.setattr(embeddings_module.httpx, "post", _fake_post)
    monkeypatch.setattr(embeddings_module.asyncio, "to_thread", _fake_to_thread)

    provider = embeddings_module.OllamaEmbeddingProvider(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    vec = await embeddings_module.embed_text_async(provider, "hello world")

    assert vec == [0.1, 0.2, 0.3]
    assert captured["fn_name"] == "embed"
    assert captured["url"] == "http://localhost:11434/api/embed"


@pytest.mark.asyncio
async def test_embed_batch_async_runs_simple_provider_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = embeddings_module.SimpleEmbeddingProvider(dim=8)

    async def _unexpected_to_thread(*args, **kwargs):
        raise AssertionError("SimpleEmbeddingProvider batches should run inline.")

    monkeypatch.setattr(embeddings_module.asyncio, "to_thread", _unexpected_to_thread)

    vectors = await embeddings_module.embed_batch_async(provider, ["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(vector) == 8 for vector in vectors)


def test_ollama_embedding_provider_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embedding": [0.1, 0.2, 0.3]}

    def _fake_post(url: str, *, json: dict[str, object], timeout: float) -> _DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr(embeddings_module.httpx, "post", _fake_post)

    provider = embeddings_module.OllamaEmbeddingProvider(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    vec = provider.embed("hello world")

    assert vec == [0.1, 0.2, 0.3]
    assert provider.dim == 3
    assert captured["url"] == "http://localhost:11434/api/embed"
    assert captured["json"] == {"model": "nomic-embed-text", "input": "hello world"}


def test_openai_embedding_provider_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [0.5, 0.4, 0.3, 0.2]}]}

    def _fake_post(
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> _DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr(embeddings_module.httpx, "post", _fake_post)

    provider = embeddings_module.OpenAIEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
    )
    vec = provider.embed("memory test")

    assert vec == [0.5, 0.4, 0.3, 0.2]
    assert provider.dim == 4
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["json"] == {
        "model": "text-embedding-3-small",
        "input": "memory test",
    }
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer test-key"
