"""
tests/test_runtime.py — Tests for runtime env configuration helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mnemos.runtime as runtime_module
from mnemos.runtime import build_embedder_from_env, build_store_from_env, resolve_env_value
from mnemos.utils import OpenAIEmbeddingProvider, SimpleEmbeddingProvider, SQLiteStore


def test_resolve_env_value_supports_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)
    monkeypatch.setenv("MNEMOS_STORAGE", "sqlite")

    value = resolve_env_value(
        "MNEMOS_STORE_TYPE",
        default="memory",
        aliases=("MNEMOS_STORAGE",),
    )

    assert value == "sqlite"


def test_build_store_from_env_supports_db_path_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_alias.db"
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)
    monkeypatch.delenv("MNEMOS_SQLITE_PATH", raising=False)
    monkeypatch.setenv("MNEMOS_STORAGE", "sqlite")
    monkeypatch.setenv("MNEMOS_DB_PATH", str(db_path))

    store = build_store_from_env(default_store_type="memory")

    assert isinstance(store, SQLiteStore)
    assert store.db_path == str(db_path)
    store.close()


def test_build_embedder_from_env_defaults_to_simple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    embedder = build_embedder_from_env(default_provider="simple")
    assert isinstance(embedder, SimpleEmbeddingProvider)


def test_build_embedder_from_env_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "does-not-exist")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        build_embedder_from_env(default_provider="simple")


def test_build_embedder_from_env_requires_api_key_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("MNEMOS_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MNEMOS_OPENAI_API_KEY"):
        build_embedder_from_env(default_provider="simple")


def test_build_embedder_from_env_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MNEMOS_OPENAI_API_KEY", "dummy-key")
    embedder = build_embedder_from_env(default_provider="simple")
    assert isinstance(embedder, OpenAIEmbeddingProvider)


def test_build_embedder_from_env_openclaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")

    embedder = build_embedder_from_env(default_provider="simple")

    assert isinstance(embedder, OpenAIEmbeddingProvider)
    assert embedder.api_key == "claw-key"
    assert embedder.base_url == "https://api.openclaw.example/v1"


def test_build_store_from_env_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyQdrantStore:
        def __init__(
            self,
            *,
            url: str | None = None,
            api_key: str | None = None,
            path: str | None = None,
            collection_name: str = "mnemos_memory",
            vector_size: int | None = None,
        ) -> None:
            captured["url"] = url
            captured["api_key"] = api_key
            captured["path"] = path
            captured["collection_name"] = collection_name
            captured["vector_size"] = vector_size

    monkeypatch.setattr(runtime_module, "QdrantStore", DummyQdrantStore, raising=False)
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "qdrant")
    monkeypatch.setenv("MNEMOS_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MNEMOS_QDRANT_API_KEY", "qdrant-key")
    monkeypatch.setenv("MNEMOS_QDRANT_PATH", "/tmp/mnemos-qdrant")
    monkeypatch.setenv("MNEMOS_QDRANT_COLLECTION", "mnemos-test")
    monkeypatch.setenv("MNEMOS_EMBEDDING_DIM", "256")

    store = build_store_from_env(default_store_type="memory")

    assert isinstance(store, DummyQdrantStore)
    assert captured["url"] == "http://localhost:6333"
    assert captured["api_key"] == "qdrant-key"
    assert captured["path"] == "/tmp/mnemos-qdrant"
    assert captured["collection_name"] == "mnemos-test"
    assert captured["vector_size"] == 256
