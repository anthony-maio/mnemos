"""
tests/test_control_plane.py — Onboarding control plane service tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from mnemos.control_plane import ControlPlaneService
from mnemos.types import MemoryChunk
from mnemos.utils import SQLiteStore


class _FacadeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def request(self, operation: str, payload: dict, **_: object) -> dict:
        self.calls.append((operation, payload))
        if operation == "list":
            return {
                "status": "ready",
                "facts": [
                    {
                        "id": "fact-1",
                        "content": "Keep the test deterministic.",
                        "version": 2,
                        "updated_at": "2026-09-16T12:00:00+00:00",
                        "access_count": 1,
                        "metadata": {
                            "scope": "project",
                            "scope_id": "project-1",
                            "source": "explicit_remember",
                        },
                    }
                ],
                "jobs": {"pending": 1, "exhausted": 0},
            }
        if operation == "health":
            return {"status": "running", "provider_state": {"llm": {"status": "ready"}}}
        if operation == "retrieve":
            return {"facts": []}
        if operation == "maintenance":
            return {"status": "committed", "completed": 0}
        return {"status": "committed", "fact_id": "fact-1"}


def _facade_service(tmp_path: Path) -> tuple[ControlPlaneService, _FacadeClient]:
    client = _FacadeClient()
    service = ControlPlaneService(
        cwd=tmp_path, client=client, env={"MNEMOS_USER_PROFILE_ID": "alice"}
    )
    service._context_payload = lambda: {  # type: ignore[method-assign]
        "user_profile_id": "alice",
        "project_id": "project-1",
        "workspace_id": None,
        "session_id": "ui-test",
        "host": "manual",
    }
    return service, client


def test_control_plane_memory_uses_configured_client_facade(tmp_path: Path) -> None:
    service, client = _facade_service(tmp_path)

    snapshot = service.get_memory_snapshot()

    assert snapshot["count"] == 1
    assert snapshot["jobs"]["pending"] == 1
    assert [operation for operation, _ in client.calls] == ["list"]


def test_control_plane_requires_explicit_scope_identity(tmp_path: Path) -> None:
    service = ControlPlaneService(cwd=tmp_path, env={})

    snapshot = service.get_memory_snapshot()

    assert snapshot["status"] == "not_configured"
    assert "MNEMOS_USER_PROFILE_ID" in snapshot["error"]


def test_project_profile_selection_survives_ui_restart_without_copying_project_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("MNEMOS_USER_PROFILE_ID", raising=False)
    from mnemos.settings import load_settings

    project = tmp_path / "project"
    project_config = project / ".mnemos" / "mnemos.toml"
    project_config.parent.mkdir(parents=True)
    global_config = tmp_path / "config" / "mnemos.toml"
    registry = tmp_path / "scopes.json"
    service = ControlPlaneService(
        cwd=project,
        global_config_path=global_config,
        env={"MNEMOS_SCOPE_REGISTRY": str(registry), "UNRELATED_SECRET": "never-export"},
        client=_FacadeClient(),
    )
    service.save_settings({"llm": {"model": "global-model"}}, scope="global")
    project_config.write_text('[llm]\nmodel = "project-model"\n', encoding="utf-8")
    assert service.configure_project("selected-profile", True)["status"] == "committed"
    restarted = ControlPlaneService(
        cwd=project,
        global_config_path=global_config,
        env={"MNEMOS_SCOPE_REGISTRY": str(registry)},
        client=_FacadeClient(),
    )
    assert restarted._configured_profile() == "selected-profile"
    assert restarted._integration_env() == {
        "MNEMOS_USER_PROFILE_ID": "selected-profile",
        "MNEMOS_SCOPE_REGISTRY": str(registry),
    }
    global_only = load_settings(env={}, cwd=global_config.parent, global_config_path=global_config)
    assert global_only.settings.llm.model == "global-model"
    restarted.env["MNEMOS_USER_PROFILE_ID"] = "explicit-override"
    assert restarted._configured_profile() == "explicit-override"


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

    assert report["status"] == "fail"
    assert "MNEMOS_USER_PROFILE_ID" in report["error"]


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

    try:
        service.get_memory_detail("chunk-123")
    except ValueError as exc:
        assert "MNEMOS_USER_PROFILE_ID" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unscoped UI inspection was accepted")


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

    assert snapshot["status"] == "not_configured"
    assert snapshot["count"] == 0
    assert "MNEMOS_USER_PROFILE_ID" in snapshot["error"]


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
    assert initial_dashboard["host"]["current"]["tier"] == "Tier 2"

    smoke = service.run_smoke_tests()
    dashboard = service.get_dashboard_view()

    assert smoke["status"] == "fail"
    assert dashboard["activity"]["last_smoke_test"]["present"] is True
    assert dashboard["activity"]["last_smoke_test"]["status"] == "fail"
    assert dashboard["next_actions"]
