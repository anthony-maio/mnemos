"""
tests/test_ui_server.py — Local UI router tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from mnemos.control_plane import ControlPlaneService
from mnemos.types import MemoryChunk
from mnemos.ui_server import MnemosUiRouter
from mnemos.utils import SQLiteStore


class _FacadeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def request(self, operation: str, payload: dict, **_: object) -> dict:
        del payload
        self.calls.append(operation)
        if operation == "health":
            return {"status": "running", "provider_state": {"cognition": {}}}
        return {"status": "degraded", "error": "test facade"}


def test_ui_router_serves_index_and_settings(tmp_path: Path) -> None:
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=tmp_path / "Mnemos" / "mnemos.toml",
    )
    router = MnemosUiRouter(service)

    index_response = router.handle("GET", "/", None)
    assert index_response.status == 200
    assert index_response.content_type == "text/html; charset=utf-8"
    assert b"Mnemos Home" in index_response.body
    assert b"Starter SQLite" in index_response.body
    assert b"Qdrant" not in index_response.body

    settings_response = router.handle("GET", "/api/settings", None)
    assert settings_response.status == 200
    payload = json.loads(settings_response.body.decode("utf-8"))
    assert "settings" in payload
    assert "paths" in payload
    assert payload["presets"][0]["id"] == "starter-sqlite"


def test_ui_router_previews_cursor_integration(tmp_path: Path) -> None:
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=tmp_path / "Mnemos" / "mnemos.toml",
    )
    router = MnemosUiRouter(service)

    response = router.handle("POST", "/api/integrations/cursor/preview", b"{}")

    assert response.status == 200
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["host"] == "cursor"
    assert "MNEMOS_CONFIG_PATH" in payload["preview"]


def test_ui_router_serves_memory_detail_endpoint(tmp_path: Path) -> None:
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
    store.store(
        MemoryChunk(
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
    )
    store.close()

    router = MnemosUiRouter(service)
    response = router.handle("GET", "/api/memory/chunk-123", None)

    assert response.status == 503
    payload = json.loads(response.body.decode("utf-8"))
    assert "MNEMOS_USER_PROFILE_ID" in payload["error"]


def test_ui_router_serves_dashboard_and_retrieval_explanation(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    global_config = tmp_path / "Mnemos" / "mnemos.toml"
    client = _FacadeClient()
    service = ControlPlaneService(
        cwd=tmp_path / "repo",
        home=tmp_path / "home",
        env={},
        global_config_path=global_config,
        client=client,
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

    store = SQLiteStore(str(db_path))
    try:
        store.store(
            MemoryChunk(
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
        )
    finally:
        store.close()

    router = MnemosUiRouter(service)

    dashboard_response = router.handle("GET", "/api/dashboard", None)
    assert dashboard_response.status == 200
    dashboard = json.loads(dashboard_response.body.decode("utf-8"))
    assert dashboard["host"]["current"]["name"] == "claude-code"
    assert dashboard["memory"]["count"] == 0

    detail_response = router.handle(
        "GET",
        "/api/memory/chunk-123?query=python%20package%20management&current_scope=project&scope_id=repo-alpha",
        None,
    )
    assert detail_response.status == 503
    detail = json.loads(detail_response.body.decode("utf-8"))
    assert "MNEMOS_USER_PROFILE_ID" in detail["error"]
    assert client.calls == ["health"]


def test_ui_router_requires_csrf_for_http_mutations(tmp_path: Path) -> None:
    router = MnemosUiRouter(
        ControlPlaneService(
            cwd=tmp_path, env={}, global_config_path=tmp_path / "Mnemos" / "mnemos.toml"
        )
    )

    token = json.loads(router.handle("GET", "/api/csrf", None).body)["csrf_token"]
    denied = router.handle("POST", "/api/settings/global", b"{}", {"Host": "127.0.0.1"})
    allowed = router.handle(
        "POST", "/api/settings/global", b"{}", {"Host": "127.0.0.1", "X-Mnemos-CSRF": token}
    )

    assert denied.status == 403
    assert allowed.status == 200
    assert token not in allowed.body.decode("utf-8")
    assert router.handle("GET", "/api/csrf", None, {"Host": "attacker.example"}).status == 403
    assert router.handle("GET", "/api/settings", None, {"Host": "attacker.example"}).status == 403


def test_ui_router_rejects_non_loopback_binding(tmp_path: Path) -> None:
    from mnemos.ui_server import run_ui_server

    try:
        run_ui_server(service=ControlPlaneService(cwd=tmp_path), host="0.0.0.0", open_browser=False)
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-loopback UI binding was accepted")
