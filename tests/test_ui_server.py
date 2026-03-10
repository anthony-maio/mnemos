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
    assert b"Mnemos Control Plane" in index_response.body

    settings_response = router.handle("GET", "/api/settings", None)
    assert settings_response.status == 200
    payload = json.loads(settings_response.body.decode("utf-8"))
    assert "settings" in payload
    assert "paths" in payload


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

    assert response.status == 200
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["id"] == "chunk-123"
    assert payload["scope"] == "project"
