"""
tests/test_mcp_server.py — Tests for MCP server runtime wiring helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.mcp_server import _build_embedder, _build_llm_provider, _build_store
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
