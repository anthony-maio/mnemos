"""
tests/test_embeddings.py — Tests for production embedding providers.
"""

from __future__ import annotations

import pytest

from mnemos.utils import embeddings as embeddings_module


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
