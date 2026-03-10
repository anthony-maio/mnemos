"""
tests/test_control_plane.py — Onboarding control plane service tests.
"""

from __future__ import annotations

from pathlib import Path

from mnemos.control_plane import ControlPlaneService
from mnemos.types import MemoryChunk
from mnemos.utils import SQLiteStore


def test_control_plane_saves_global_settings(tmp_path: Path) -> None:
    global_config = tmp_path / "Mnemos" / "mnemos.toml"
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=global_config,
    )

    result = service.save_settings(
        {
            "llm": {"provider": "openrouter", "model": "openrouter/auto"},
            "embedding": {"provider": "openrouter", "model": "text-embedding-3-small"},
            "providers": {
                "openrouter": {
                    "api_key": "router-key",
                    "base_url": "https://openrouter.ai/api/v1",
                }
            },
            "storage": {"type": "sqlite", "sqlite_path": ".mnemos/memory.db"},
        },
        scope="global",
    )

    assert result["saved"] is True
    text = global_config.read_text(encoding="utf-8")
    assert 'provider = "openrouter"' in text
    assert 'api_key = "router-key"' in text


def test_control_plane_imports_existing_setup(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    codex_config = home_dir / ".codex" / "config.toml"
    codex_config.parent.mkdir(parents=True)
    codex_config.write_text(
        """
[mcp_servers.mnemos]
command = "mnemos-mcp"

[mcp_servers.mnemos.env]
MNEMOS_LLM_PROVIDER = "openrouter"
MNEMOS_EMBEDDING_PROVIDER = "openrouter"
MNEMOS_OPENROUTER_API_KEY = "router-key"
MNEMOS_STORE_TYPE = "sqlite"
MNEMOS_SQLITE_PATH = ".mnemos/memory.db"
""".strip(),
        encoding="utf-8",
    )

    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=home_dir,
        env={},
        global_config_path=tmp_path / "Mnemos" / "mnemos.toml",
    )

    imported = service.import_existing_setup()

    assert imported["settings"]["llm"]["provider"] == "openrouter"
    assert "codex" in imported["sources"]


def test_control_plane_runs_smoke_test_with_mock_profile(tmp_path: Path) -> None:
    global_config = tmp_path / "Mnemos" / "mnemos.toml"
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=global_config,
    )
    service.save_settings(
        {
            "llm": {"provider": "mock"},
            "embedding": {"provider": "simple", "dim": 64},
            "storage": {"type": "sqlite", "sqlite_path": str(tmp_path / "smoke.db")},
        },
        scope="global",
    )

    report = service.run_smoke_tests()

    assert report["status"] == "pass"
    assert report["steps"]["store"] == "pass"
    assert report["steps"]["retrieve"] == "pass"
    assert report["steps"]["consolidate"] == "pass"
    assert report["steps"]["health"] in {"pass", "warn"}


def test_control_plane_memory_detail_returns_inspection_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    global_config = tmp_path / "Mnemos" / "mnemos.toml"
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=global_config,
    )
    service.save_settings(
        {
            "llm": {"provider": "mock"},
            "embedding": {"provider": "simple", "dim": 64},
            "storage": {"type": "sqlite", "sqlite_path": str(db_path)},
        },
        scope="global",
    )

    store = SQLiteStore(str(db_path))
    chunk = MemoryChunk(
        id="chunk-123",
        content="Use uv for Python package management.",
        metadata={
            "scope": "project",
            "scope_id": "repo-alpha",
            "source": "surprisal_gate",
            "ingest_channel": "manual",
            "encoding_reason": "High surprisal.",
        },
    )
    store.store(chunk)
    store.close()

    payload = service.get_memory_detail("chunk-123")

    assert payload["id"] == "chunk-123"
    assert payload["scope"] == "project"
    assert payload["provenance"]["stored_by"] == "surprisal_gate"
    assert payload["history"] == []
