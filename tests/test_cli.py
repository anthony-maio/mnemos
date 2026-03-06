"""
tests/test_cli.py — Tests for CLI runtime wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from argparse import Namespace

from mnemos.cli import _build_engine, _build_profile_env, _cmd_doctor, _cmd_profile
from mnemos.utils import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
    SimpleEmbeddingProvider,
    SQLiteStore,
)


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


def test_build_engine_infers_openclaw_embedder_from_llm_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_cli_openclaw_inferred.db"
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "claw-key")
    monkeypatch.setenv("MNEMOS_OPENCLAW_URL", "https://api.openclaw.example/v1")
    monkeypatch.delenv("MNEMOS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))

    engine = _build_engine()

    assert isinstance(engine.embedder, OpenAIEmbeddingProvider)
    assert engine.embedder.api_key == "claw-key"
    assert engine.embedder.base_url == "https://api.openclaw.example/v1"
    engine.store.close()


@pytest.mark.asyncio
async def test_cli_doctor_prints_report(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    await _cmd_doctor(
        Namespace(
            qdrant_chunk_threshold=5000,
            latency_p95_threshold_ms=250.0,
            observed_p95_ms=None,
        )
    )
    captured = capsys.readouterr().out
    assert '"profile": "starter"' in captured
    assert '"status": "degraded"' in captured


def test_build_profile_env_starter_defaults() -> None:
    env = _build_profile_env(
        profile="starter",
        llm_provider="openclaw",
        embedding_provider=None,
        model="openclaw/claude-sonnet",
        sqlite_path=".mnemos/memory.db",
        qdrant_path=".mnemos/qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_collection="mnemos_memory",
    )
    assert env["MNEMOS_STORE_TYPE"] == "sqlite"
    assert env["MNEMOS_SQLITE_PATH"] == ".mnemos/memory.db"
    assert env["MNEMOS_LLM_PROVIDER"] == "openclaw"
    assert env["MNEMOS_EMBEDDING_PROVIDER"] == "openclaw"


def test_build_profile_env_local_performance() -> None:
    env = _build_profile_env(
        profile="local-performance",
        llm_provider="openai",
        embedding_provider="openai",
        model="gpt-4o-mini",
        sqlite_path=".mnemos/memory.db",
        qdrant_path=".mnemos/qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_collection="mnemos_memory",
    )
    assert env["MNEMOS_STORE_TYPE"] == "qdrant"
    assert env["MNEMOS_QDRANT_PATH"] == ".mnemos/qdrant"
    assert env["MNEMOS_QDRANT_COLLECTION"] == "mnemos_memory"
    assert "MNEMOS_QDRANT_URL" not in env


@pytest.mark.asyncio
async def test_cli_profile_writes_dotenv(tmp_path: Path, capsys: Any) -> None:
    output_path = tmp_path / "mnemos.profile.env"
    await _cmd_profile(
        Namespace(
            profile="starter",
            format="dotenv",
            write=str(output_path),
            llm_provider="openclaw",
            embedding_provider="",
            model="",
            sqlite_path=".mnemos/memory.db",
            qdrant_path=".mnemos/qdrant",
            qdrant_url="http://localhost:6333",
            qdrant_collection="mnemos_memory",
        )
    )
    text = output_path.read_text(encoding="utf-8")
    assert "MNEMOS_STORE_TYPE=sqlite" in text
    assert "MNEMOS_SQLITE_PATH=.mnemos/memory.db" in text
    assert "MNEMOS_LLM_PROVIDER=openclaw" in text
    captured = capsys.readouterr().out
    assert "MNEMOS_STORE_TYPE=sqlite" in captured
