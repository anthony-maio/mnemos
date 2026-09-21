"""MCP tool safety metadata and read-only handler regressions."""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mnemos.config import MnemosConfig, SurprisalConfig
from mnemos.client import MnemosClient
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.mcp_server import (
    ServiceMCPContext,
    _create_service_mcp_server,
    _service_scope_context,
)
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import SQLiteStore

PROJECT_ID = "project-mcp-annotations"
PROFILE_ID = "profile-mcp-annotations"


def _start_scoped_service(tmp_path: Path) -> tuple[MnemosService, Path]:
    registry = ScopeRegistry(profiles=[PROFILE_ID])
    project = registry.register_project(tmp_path, user_profile_id=PROFILE_ID, enabled=True)
    project.id = PROJECT_ID
    registry.projects = {PROJECT_ID: project}
    registry_path = tmp_path / "scopes.json"
    registry.save(registry_path)

    config = MnemosConfig(surprisal=SurprisalConfig(threshold=0.0, min_content_length=0))
    config.modules.surprisal = False
    config.modules.sleep = False
    engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=config.surprisal.embedding_dim),
        store=SQLiteStore(str(tmp_path / "mcp-annotations.db")),
    )
    service = MnemosService(
        dispatcher=LocalMemoryService(engine, registry),
        descriptor_path=tmp_path / "runtime" / "runtime.json",
        lock_path=tmp_path / "runtime" / "runtime.lock",
        startup_timeout=3,
        dispatch_timeout=10,
    )
    service.start()
    return service, registry_path


def _service_env(service: MnemosService, registry_path: Path) -> dict[str, str]:
    return {
        "MNEMOS_RUNTIME_DESCRIPTOR": str(service.descriptor_path),
        "MNEMOS_SCOPE_REGISTRY": str(registry_path),
        "MNEMOS_USER_PROFILE_ID": PROFILE_ID,
        "MNEMOS_PROJECT_ROOT": str(registry_path.parent),
        "MNEMOS_SESSION_ID": "mcp-annotations-session",
        "MNEMOS_HOST": "manual",
        "MNEMOS_AUTO_START": "0",
    }


READ_ONLY = {
    "mnemos_job_status",
    "mnemos_forget_preview",
    "mnemos_stats",
    "mnemos_health",
    "mnemos_inspect",
    "mnemos_list",
    "mnemos_reviews",
    "mnemos_inspect_review",
}
REVIEW_DISCARDS = {"mnemos_discard_review"}
PROVIDER_WRITES = {"mnemos_store", "mnemos_retrieve"}
LOCAL_WRITES = {"mnemos_capture", "mnemos_feedback"}
DESTRUCTIVE_WRITES = {"mnemos_correct", "mnemos_forget", "mnemos_forget_related"}
MAINTENANCE = {"mnemos_consolidate"}


async def _registered_tools() -> dict[str, Any]:
    server = _create_service_mcp_server()
    return {tool.name: tool for tool in await server.list_tools()}


@pytest.mark.asyncio
async def test_fastmcp_tools_publish_truthful_safety_annotations() -> None:
    tools = await _registered_tools()

    assert (
        READ_ONLY
        | PROVIDER_WRITES
        | LOCAL_WRITES
        | DESTRUCTIVE_WRITES
        | MAINTENANCE
        | REVIEW_DISCARDS
        == set(tools)
    )

    for name in READ_ONLY:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is True
        assert annotations.openWorldHint is False

    for name in PROVIDER_WRITES:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is False
        assert annotations.openWorldHint is True

    for name in LOCAL_WRITES:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is False
        assert annotations.openWorldHint is False

    for name in DESTRUCTIVE_WRITES:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is True
        assert annotations.idempotentHint is False
        assert annotations.openWorldHint is False

    for name in REVIEW_DISCARDS:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is True
        assert annotations.idempotentHint is True
        assert annotations.openWorldHint is False

    annotations = tools["mnemos_consolidate"].annotations
    assert annotations is not None
    assert annotations.readOnlyHint is False
    assert annotations.destructiveHint is True
    assert annotations.idempotentHint is False
    assert annotations.openWorldHint is True


def _sqlite_snapshot(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    with sqlite3.connect(path) as connection:
        tables = [
            (row[0], row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        snapshot: dict[str, list[tuple[Any, ...]]] = {}
        for name, schema in tables:
            if isinstance(schema, str) and schema.upper().startswith("CREATE VIRTUAL TABLE"):
                # sqlite-vec is loaded by the service process.  The extension
                # is intentionally not loaded into this test process, so
                # retain the virtual table definition as its stable snapshot.
                snapshot[name] = [("__virtual_table__", schema)]
                continue
            quoted_name = '"' + name.replace('"', '""') + '"'
            rows = connection.execute(f"SELECT * FROM {quoted_name}").fetchall()
            snapshot[name] = sorted(rows, key=repr)
        return snapshot


def _tool_json(result: Any) -> dict[str, Any]:
    content, structured = result
    assert content
    text = getattr(content[0], "text", None)
    assert isinstance(text, str)
    payload = json.loads(text)
    assert isinstance(payload, dict)
    assert isinstance(structured, dict)
    return payload


@pytest.mark.asyncio
async def test_fastmcp_read_only_handlers_do_not_mutate_populated_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, registry_path = _start_scoped_service(tmp_path)
    client: MnemosClient | None = None
    stopped = False
    try:
        monkeypatch.setenv("MNEMOS_RUNTIME_DESCRIPTOR", str(service.descriptor_path))
        for key, value in _service_env(service, registry_path).items():
            monkeypatch.setenv(key, value)

        client = MnemosClient(auto_start=False)
        scope = _service_scope_context()
        context_payload = scope.model_dump(mode="json")
        remembered = await client.request(
            "remember",
            {
                "content": "The annotation regression uses a populated SQLite store.",
                "context": context_payload,
                "event_id": "mcp-annotations-remember",
            },
        )
        await client.request("maintenance", {})
        listed = await client.request(
            "list",
            {
                "limit": 20,
                "sort_by": "created_at",
                "context": context_payload,
                "allowed_scopes": ["project", "workspace", "global"],
            },
        )
        fact = listed["facts"][0]
        correction = await client.request(
            "correct",
            {
                "fact_id": fact["id"],
                "content": "The annotation regression uses a populated, revised SQLite store.",
                "expected_version": fact["version"],
                "context": context_payload,
                "event_id": "mcp-annotations-correction",
            },
        )
        await client.request("maintenance", {})
        revised = await client.request(
            "list",
            {
                "limit": 20,
                "sort_by": "created_at",
                "context": context_payload,
                "allowed_scopes": ["project", "workspace", "global"],
            },
        )
        fact_id = revised["facts"][0]["id"]
        expected_version = revised["facts"][0]["version"]
        assert expected_version == 2
        assert remembered["status"] == "committed"
        assert correction["status"] == "committed"

        server = _create_service_mcp_server()
        context = SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=ServiceMCPContext(
                    client=client,
                    scope=scope,
                )
            )
        )
        db_path = tmp_path / "mcp-annotations.db"
        expected_tables = _sqlite_snapshot(db_path)
        calls = {
            "mnemos_job_status": {"job_id": correction["job_id"]},
            "mnemos_forget_preview": {
                "fact_id": fact_id,
                "expected_version": expected_version,
                "limit": 100,
            },
            "mnemos_stats": {},
            "mnemos_health": {},
            "mnemos_inspect": {"chunk_id": fact_id},
            "mnemos_list": {"limit": 20, "sort_by": "created_at"},
        }
        for name, arguments in calls.items():
            tool = server._tool_manager.get_tool(name)
            assert tool is not None
            payload = _tool_json(await tool.run(arguments, context=context, convert_result=True))
            assert payload["schema_version"] == 1
            if name == "mnemos_job_status":
                assert payload["job"]["id"] == correction["job_id"]
                assert payload["job"]["status"] == "completed"
            elif name == "mnemos_forget_preview":
                assert payload["target"]["fact_id"] == fact_id
                assert payload["target"]["version"] == expected_version
                assert isinstance(payload["preview_digest"], str)
                assert payload["complete"] is True
            elif name in {"mnemos_stats", "mnemos_health"}:
                assert payload["status"] == "running"
                assert payload["store"]["total_chunks"] == 1
                assert payload["provider_state"]["cognition"]["status"] == "not_observed"
            elif name == "mnemos_inspect":
                assert payload["fact"]["id"] == fact_id
                assert payload["fact"]["version"] == expected_version
                assert payload["evidence"]
                assert len(payload["revisions"]) == 1
            elif name == "mnemos_list":
                assert any(item["id"] == fact_id for item in payload["facts"])
                assert payload["jobs"]["completed"] >= 2
            assert _sqlite_snapshot(db_path) == expected_tables, name

        await client.close()
        client = None
        service.stop()
        stopped = True
        reopened_store = SQLiteStore(str(db_path))
        try:
            assert _sqlite_snapshot(db_path) == expected_tables
        finally:
            reopened_store.close()
    finally:
        if client is not None:
            await client.close()
        if not stopped:
            service.stop()
