"""
tests/test_runtime.py — Tests for runtime env configuration helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mnemos.runtime as runtime_module
from mnemos.runtime import build_embedder_from_env, build_store_from_env, resolve_env_value
from mnemos.types import MemoryChunk
from mnemos.utils import (
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleEmbeddingProvider,
    SQLiteStore,
)
from mnemos.utils.llm import UnavailableLLMProvider


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


def test_embedding_only_global_config_does_not_select_mock_llm(tmp_path: Path) -> None:
    config = tmp_path / "global.toml"
    config.write_text('[embedding]\nprovider = "simple"\ndim = 8\n', encoding="utf-8")
    engine = runtime_module.build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(config),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.sqlite"),
        },
        cwd=tmp_path,
    )
    try:
        assert isinstance(engine.llm, UnavailableLLMProvider)
    finally:
        engine.store.close()


def test_embedding_only_project_config_does_not_select_mock_llm(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config = project / ".mnemos" / "mnemos.toml"
    config.parent.mkdir(parents=True)
    config.write_text('[embedding]\nprovider = "simple"\ndim = 8\n', encoding="utf-8")
    engine = runtime_module.build_configured_engine(
        env={
            "MNEMOS_CONFIG_PATH": str(tmp_path / "missing-global.toml"),
            "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.sqlite"),
        },
        cwd=project,
    )
    try:
        assert isinstance(engine.llm, UnavailableLLMProvider)
    finally:
        engine.store.close()


@pytest.mark.parametrize("project_config", [False, True])
def test_configured_mock_provider_remains_explicitly_rejected(
    tmp_path: Path, project_config: bool
) -> None:
    config = tmp_path / ".mnemos" / "mnemos.toml" if project_config else tmp_path / "global.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[llm]\nprovider = "mock"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="allow_mock"):
        runtime_module.build_configured_engine(
            env={
                "MNEMOS_CONFIG_PATH": str(
                    tmp_path / "missing-global.toml" if project_config else config
                ),
                "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.sqlite"),
            },
            cwd=tmp_path,
        )


def test_build_store_from_env_defaults_to_sqlite_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_default.db"
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))

    store = build_store_from_env(default_store_type="sqlite")

    assert isinstance(store, SQLiteStore)
    assert store.db_path == str(db_path)
    store.close()


def test_build_embedder_from_env_defaults_to_simple(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    embedder = build_embedder_from_env(
        default_provider="simple",
        env={
            "MNEMOS_CONFIG_PATH": str(tmp_path / "missing-mnemos.toml"),
            "MNEMOS_LLM_PROVIDER": "mock",
        },
    )
    assert isinstance(embedder, SimpleEmbeddingProvider)


def test_build_embedder_from_env_respects_configured_dimension_for_reindex(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "mnemos_vec_dim.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    provider = SimpleEmbeddingProvider(dim=768)
    try:
        store.store(
            MemoryChunk(
                id="chunk-768",
                content="alpha memory",
                embedding=provider.embed("alpha memory"),
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )
    finally:
        store.close()

    embedder = build_embedder_from_env(
        default_provider="simple",
        env={
            "MNEMOS_STORE_TYPE": "sqlite",
            "MNEMOS_SQLITE_PATH": str(db_path),
            "MNEMOS_LLM_PROVIDER": "mock",
            "MNEMOS_EMBEDDING_PROVIDER": "simple",
            "MNEMOS_EMBEDDING_DIM": "64",
        },
    )

    assert isinstance(embedder, SimpleEmbeddingProvider)
    assert embedder.dim == 64


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


def test_build_embedder_from_env_infers_openai_from_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MNEMOS_OPENAI_API_KEY", "dummy-key")

    embedder = build_embedder_from_env(default_provider="simple")

    assert isinstance(embedder, OpenAIEmbeddingProvider)


def test_build_embedder_from_env_infers_ollama_from_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MNEMOS_OLLAMA_URL", "http://localhost:11434")

    embedder = build_embedder_from_env(default_provider="simple")

    assert isinstance(embedder, OllamaEmbeddingProvider)


def test_build_store_from_env_rejects_legacy_store_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "qdrant")
    with pytest.raises(Exception, match="storage.type"):
        build_store_from_env(default_store_type="memory")

    monkeypatch.setenv("MNEMOS_STORE_TYPE", "neo4j")
    with pytest.raises(Exception, match="storage.type"):
        build_store_from_env(default_store_type="memory")
