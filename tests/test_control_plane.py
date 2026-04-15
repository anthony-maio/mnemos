"""
tests/test_control_plane.py — Onboarding control plane service tests.
"""

from __future__ import annotations

import json
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


def test_control_plane_settings_view_exposes_sqlite_first_presets(tmp_path: Path) -> None:
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=tmp_path / "Mnemos" / "mnemos.toml",
    )

    view = service.get_settings_view()
    preset_ids = [preset["id"] for preset in view["presets"]]

    assert preset_ids == ["starter-sqlite", "stable-memory"]
    assert view["setup_story"]["primary_store"] == "sqlite"
    assert view["setup_story"]["advanced_store_types"] == ["memory"]
    assert "qdrant" not in json.dumps(view).lower()


def test_control_plane_memory_snapshot_includes_rollups_and_consolidation_summary(
    tmp_path: Path,
) -> None:
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
            "sleep": {
                "recall_gated_plasticity_enabled": True,
                "recall_min_supporting_episodes": 2,
                "recall_similarity_threshold": 0.82,
            },
        },
        scope="global",
    )

    store = SQLiteStore(str(db_path))
    try:
        store.store(
            MemoryChunk(
                id="global-1",
                content="Global preference",
                metadata={"scope": "global"},
            )
        )
        store.store(
            MemoryChunk(
                id="project-1",
                content="Use uv for Python tooling.",
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )
        store.store(
            MemoryChunk(
                id="workspace-1",
                content="Workspace deploys go through staging first.",
                metadata={"scope": "workspace", "scope_id": "ws-main"},
            )
        )
        store.store(
            MemoryChunk(
                id="sleep-1",
                content="Deploys run through fly.toml.",
                metadata={
                    "scope": "project",
                    "scope_id": "repo-alpha",
                    "source": "sleep_consolidation",
                    "consolidation_time": "2026-04-02T12:00:00+00:00",
                    "recall_supporting_episodes": 3,
                    "recall_support_score": 0.91,
                    "recall_similarity_threshold": 0.82,
                    "recall_min_supporting_episodes": 2,
                },
            )
        )
    finally:
        store.close()

    snapshot = service.get_memory_snapshot(limit=10)

    assert snapshot["count"] == 4
    assert snapshot["recent_count"] == 4
    assert snapshot["counts_by_scope"] == {"global": 1, "project": 2, "workspace": 1}
    assert snapshot["last_consolidation"]["present"] is True
    assert snapshot["last_consolidation"]["source"] == "sleep_consolidation"
    assert snapshot["last_consolidation"]["stored_chunk_count"] == 1
    assert snapshot["recall_gated"]["enabled"] is True
    assert snapshot["recall_gated"]["stored_chunk_count"] == 1
    assert snapshot["recall_gated"]["max_supporting_episodes"] == 3


def test_control_plane_dashboard_tracks_last_smoke_and_host_support(tmp_path: Path) -> None:
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
            "onboarding": {"preferred_host": "claude-code"},
        },
        scope="global",
    )

    initial_dashboard = service.get_dashboard_view()
    assert initial_dashboard["activity"]["last_smoke_test"]["present"] is False
    assert initial_dashboard["host"]["recommended"] == "claude-code"
    assert initial_dashboard["host"]["current"]["tier"] == "Tier 1"

    smoke = service.run_smoke_tests()
    dashboard = service.get_dashboard_view()

    assert smoke["status"] == "pass"
    assert dashboard["activity"]["last_smoke_test"]["present"] is True
    assert dashboard["activity"]["last_smoke_test"]["status"] == "pass"
    assert dashboard["next_actions"]
