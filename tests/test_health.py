"""
tests/test_health.py — Readiness and profile checks for production onboarding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.types import MemoryChunk
from mnemos.health import detect_profile, run_health_checks
from mnemos.utils import SQLiteStore


def test_detect_profile_starter_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.delenv("MNEMOS_QDRANT_PATH", raising=False)
    assert detect_profile() == "starter"


def test_detect_profile_local_performance_embedded_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "qdrant")
    monkeypatch.setenv("MNEMOS_QDRANT_PATH", ".mnemos-qdrant")
    assert detect_profile() == "local-performance"


def test_detect_profile_scale_external_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "qdrant")
    monkeypatch.delenv("MNEMOS_QDRANT_PATH", raising=False)
    assert detect_profile() == "scale"


def test_health_fails_when_openclaw_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.delenv("MNEMOS_OPENCLAW_API_KEY", raising=False)
    monkeypatch.delenv("MNEMOS_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")

    report = run_health_checks()

    assert report["status"] == "not_ready"
    assert report["summary"]["fail"] >= 1


def test_health_reports_degraded_when_using_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")

    report = run_health_checks()

    assert report["status"] == "degraded"
    assert report["degraded_mode"] is True
    assert report["summary"]["warn"] >= 1


def test_health_reports_ready_for_openclaw_starter_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos.db"
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "test-key")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openclaw")

    report = run_health_checks()

    assert report["profile"] == "starter"
    assert report["status"] == "ready"
    assert report["summary"]["fail"] == 0


def test_health_does_not_recommend_qdrant_when_sqlite_below_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_below_threshold.db"
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "test-key")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_DOCTOR_QDRANT_CHUNK_THRESHOLD", "100")

    report = run_health_checks()

    assert report["upgrade_signals"]["threshold_exceeded"] is False
    assert not any("Upgrade path:" in rec for rec in report["recommendations"])


def test_health_recommends_qdrant_when_sqlite_chunk_threshold_exceeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_above_threshold.db"
    store = SQLiteStore(db_path=str(db_path))
    try:
        store.store(MemoryChunk(content="example memory", embedding=[0.1, 0.2, 0.3]))
    finally:
        store.close()

    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "test-key")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_DOCTOR_QDRANT_CHUNK_THRESHOLD", "1")

    report = run_health_checks()

    assert report["upgrade_signals"]["threshold_exceeded"] is True
    assert any("Upgrade path:" in rec for rec in report["recommendations"])


def test_health_reports_legacy_unscoped_sqlite_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "mnemos_legacy_scope.db"
    store = SQLiteStore(db_path=str(db_path))
    try:
        store.store(
            MemoryChunk(
                content="legacy chunk without scope",
                embedding=[0.1, 0.2, 0.3],
                metadata={},
            )
        )
    finally:
        store.close()

    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "openclaw")
    monkeypatch.setenv("MNEMOS_OPENCLAW_API_KEY", "test-key")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "openclaw")

    report = run_health_checks()

    assert report["status"] == "degraded"
    assert report["scope_isolation"]["legacy_unscoped_chunks"] == 1
    assert report["scope_isolation"]["ready"] is False
    assert any("legacy unscoped" in rec.lower() for rec in report["recommendations"])
