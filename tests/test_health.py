"""
tests/test_health.py — Readiness and profile checks for production onboarding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.health import detect_profile, run_health_checks


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
