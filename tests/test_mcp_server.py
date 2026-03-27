"""
tests/test_mcp_server.py — Tests for MCP server runtime wiring helpers.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import ValidationError

import pytest

try:
    ExceptionGroup
except NameError:  # pragma: no cover - Python < 3.11 fallback
    from exceptiongroup import ExceptionGroup

from mnemos.mcp_server import (
    _build_config,
    _build_embedder,
    _build_llm_provider,
    _build_store,
    _format_startup_error,
)
from mnemos.utils import OpenAIEmbeddingProvider, OpenAIProvider, SQLiteStore


def test_mcp_build_store_supports_alias_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_mcp_alias.db"
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)
    monkeypatch.delenv("MNEMOS_SQLITE_PATH", raising=False)
    monkeypatch.setenv("MNEMOS_STORAGE", "sqlite")
    monkeypatch.setenv("MNEMOS_DB_PATH", str(db_path))

    store = _build_store()
    assert isinstance(store, SQLiteStore)
    assert store.db_path == str(db_path)
    store.close()


def test_mcp_build_embedder_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "invalid")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        _build_embedder()


def test_mcp_build_llm_provider_openclaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "openclaw/claude")

    llm = _build_llm_provider()

    assert isinstance(llm, OpenAIProvider)
    assert llm.api_key == "claw-key"
    assert llm.base_url == "https://api.openclaw.example/v1"
    assert llm.model == "openclaw/claude"


def test_mcp_build_embedder_infers_openclaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")

    embedder = _build_embedder()

    assert isinstance(embedder, OpenAIEmbeddingProvider)
    assert embedder.api_key == "claw-key"
    assert embedder.base_url == "https://api.openclaw.example/v1"


def test_mcp_build_config_reads_governance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_MEMORY_CAPTURE_MODE", "manual_only")
    monkeypatch.setenv("MNEMOS_MEMORY_RETENTION_TTL_DAYS", "14")
    monkeypatch.setenv("MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE", "200")

    config = _build_config()
    assert config.governance.capture_mode == "manual_only"
    assert config.governance.retention_ttl_days == 14
    assert config.governance.max_chunks_per_scope == 200


def test_mcp_builders_use_mnemos_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "mnemos.toml"
    db_path = tmp_path / "mcp-configured.db"
    config_path.write_text(
        f"""
[llm]
provider = "openrouter"
model = "openrouter/auto"

[embedding]
provider = "openrouter"
model = "text-embedding-3-small"

[storage]
type = "sqlite"
sqlite_path = "{db_path.as_posix()}"

[providers.openrouter]
api_key = "router-key"
base_url = "https://openrouter.ai/api/v1"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("MNEMOS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)

    llm = _build_llm_provider()
    embedder = _build_embedder()
    store = _build_store()

    assert isinstance(llm, OpenAIProvider)
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert isinstance(embedder, OpenAIEmbeddingProvider)
    assert isinstance(store, SQLiteStore)
    assert Path(store.db_path).resolve() == db_path.resolve()
    store.close()


def test_format_startup_error_for_legacy_store_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "mnemos.toml"
    config_path.write_text(
        """
[storage]
type = "neo4j"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("MNEMOS_CONFIG_PATH", str(config_path))

    with pytest.raises(ValidationError) as exc_info:
        _build_config()

    message = _format_startup_error(exc_info.value)

    assert "Mnemos MCP startup failed." in message
    assert str(config_path) in message
    assert "storage.type" in message
    assert "sqlite" in message
    assert "mnemos-cli doctor" in message


def test_format_startup_error_prefers_nested_actionable_message() -> None:
    nested = RuntimeError(
        "Mnemos MCP startup failed.\n"
        "Config path: C:\\test\\mnemos.toml\n"
        "Details: storage.type invalid"
    )
    wrapped = ExceptionGroup("outer", [nested])

    message = _format_startup_error(wrapped)

    assert message.startswith("Mnemos MCP startup failed.")
    assert "storage.type invalid" in message
    assert "ExceptionGroup" not in message
