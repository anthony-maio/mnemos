"""Embedding resolution stays consistent across entrypoints and config sources."""

from __future__ import annotations

from pathlib import Path

import pytest

import mnemos.cli as cli
import mnemos.mcp_server as mcp_server
import mnemos.runtime as runtime
from mnemos.settings import load_settings
from mnemos.utils.embeddings import OllamaEmbeddingProvider, SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider


class StubFastembedProvider(SimpleEmbeddingProvider):
    """Inspect provider selection without downloading or running an ONNX model."""

    def __init__(self, model: str) -> None:
        super().__init__(dim=8)
        self.model = model


@pytest.fixture
def fresh_embedding_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "MNEMOS_LLM_PROVIDER",
        "MNEMOS_LLM_MODEL",
        "MNEMOS_EMBEDDING_PROVIDER",
        "MNEMOS_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "FastembedProvider", StubFastembedProvider)


@pytest.mark.parametrize("unrelated_config", [False, True])
def test_cli_mcp_and_service_share_fresh_embedding_defaults(
    fresh_embedding_config: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unrelated_config: bool,
) -> None:
    if unrelated_config:
        config = tmp_path / "storage-only.toml"
        config.write_text("[runtime]\ndebug = false\n", encoding="utf-8")
        monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config))

    cli_engine = cli._build_engine()
    service_engine = runtime.build_configured_engine(cwd=tmp_path)
    try:
        embedders = (cli_engine.embedder, mcp_server._build_embedder(), service_engine.embedder)
        for embedder in embedders:
            assert isinstance(embedder, StubFastembedProvider)
            assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert isinstance(service_engine.llm, UnavailableLLMProvider)
    finally:
        cli_engine.store.close()
        service_engine.store.close()


@pytest.mark.parametrize("source", ["env", "global", "project"])
def test_explicit_provider_and_model_survive_entrypoint_defaults(
    fresh_embedding_config: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
) -> None:
    if source == "env":
        monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("MNEMOS_EMBEDDING_MODEL", "custom-embed-model")
    else:
        config = (
            tmp_path / "global.toml" if source == "global" else tmp_path / ".mnemos" / "mnemos.toml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            '[embedding]\nprovider = "ollama"\nmodel = "custom-embed-model"\n',
            encoding="utf-8",
        )
        if source == "global":
            monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config))

    cli_engine = cli._build_engine()
    try:
        for embedder in (
            cli_engine.embedder,
            mcp_server._build_embedder(),
            runtime.build_embedder_from_env(default_provider="simple"),
        ):
            assert isinstance(embedder, OllamaEmbeddingProvider)
            assert embedder.model == "custom-embed-model"
    finally:
        cli_engine.store.close()


@pytest.mark.parametrize("explicit_model", [None, "custom-embed-model"])
def test_compatibility_provider_override_resolves_its_own_model(
    fresh_embedding_config: None,
    tmp_path: Path,
    explicit_model: str | None,
) -> None:
    env = {"MNEMOS_CONFIG_PATH": str(tmp_path / "absent.toml")}
    if explicit_model is not None:
        env["MNEMOS_EMBEDDING_MODEL"] = explicit_model
    embedder = runtime.build_embedder_from_env(default_provider="ollama", env=env, cwd=tmp_path)
    assert isinstance(embedder, OllamaEmbeddingProvider)
    assert embedder.model == (explicit_model or "nomic-embed-text")


@pytest.mark.parametrize(
    ("llm_provider", "embedding_provider", "embedding_model"),
    [
        ("mock", "fastembed", "BAAI/bge-small-en-v1.5"),
        ("ollama", "ollama", "nomic-embed-text"),
        ("openai", "openai", "text-embedding-3-small"),
        ("openclaw", "openclaw", "text-embedding-3-small"),
        ("openrouter", "openrouter", "text-embedding-3-small"),
    ],
)
def test_generated_profile_matches_canonical_provider_and_model(
    tmp_path: Path,
    llm_provider: str,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    env = cli._build_profile_env(
        profile="default",
        llm_provider=llm_provider,
        embedding_provider=None,
        model=None,
        sqlite_path=str(tmp_path / "profile.sqlite"),
    )
    assert env["MNEMOS_EMBEDDING_PROVIDER"] == embedding_provider
    resolved = load_settings(env=env, cwd=tmp_path, global_config_path=tmp_path / "absent.toml")
    assert resolved.settings.embedding.provider == embedding_provider
    assert resolved.settings.embedding.model == embedding_model


def test_generated_profile_preserves_explicit_embedding_provider() -> None:
    env = cli._build_profile_env(
        profile="default",
        llm_provider="openrouter",
        embedding_provider="simple",
        model="custom-llm-model",
        sqlite_path="example.sqlite",
    )
    assert env["MNEMOS_EMBEDDING_PROVIDER"] == "simple"
    assert env["MNEMOS_LLM_MODEL"] == "custom-llm-model"
