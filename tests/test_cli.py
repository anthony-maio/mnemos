"""
tests/test_cli.py — Tests for CLI runtime wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.cli import _build_engine
from mnemos.utils import OpenAIProvider, SimpleEmbeddingProvider, SQLiteStore


def test_build_engine_supports_storage_alias_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_alias.db"
    monkeypatch.delenv("MNEMOS_STORE_TYPE", raising=False)
    monkeypatch.delenv("MNEMOS_SQLITE_PATH", raising=False)
    monkeypatch.setenv("MNEMOS_STORAGE", "sqlite")
    monkeypatch.setenv("MNEMOS_DB_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")

    engine = _build_engine()
    assert isinstance(engine.store, SQLiteStore)
    assert engine.store.db_path == str(db_path)
    assert isinstance(engine.embedder, SimpleEmbeddingProvider)
    engine.store.close()


def test_build_engine_supports_openclaw_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_openclaw.db"
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "openclaw/claude")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))

    engine = _build_engine()

    assert isinstance(engine.llm, OpenAIProvider)
    assert engine.llm.api_key == "claw-key"
    assert engine.llm.base_url == "https://api.openclaw.example/v1"
    engine.store.close()
