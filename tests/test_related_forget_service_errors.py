"""Authenticated transport behavior for reviewed forget selection errors."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from mnemos.client import MnemosClient, ServiceClientError
from mnemos.adapters.hermes_tools import dispatch_tool
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore
from tests.test_related_forget import _seed_overlap


def _runtime(tmp_path: Path) -> tuple[MnemosService, LocalMemoryService, Any]:
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="profile-a", enabled=True)
    context = registry.resolve(
        project,
        user_profile_id="profile-a",
        session_id="service-error-session",
        host="hermes",
    )
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False,
            mutable_rag=False,
            affective=False,
            sleep=False,
            spreading=False,
        )
    )
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=SimpleEmbeddingProvider(),
        store=store,
        scope_registry=registry,
        require_scope_context=True,
    )
    domain = LocalMemoryService(engine, registry)
    service = MnemosService(
        dispatcher=domain,
        descriptor_path=tmp_path / "runtime" / "descriptor.json",
        lock_path=tmp_path / "runtime" / "service.lock",
        startup_timeout=3,
        dispatch_timeout=10,
    )
    service.start()
    return service, domain, context


def _snapshot(domain: LocalMemoryService) -> dict[str, list[dict[str, Any]]]:
    connection = domain.engine.store._conn  # noqa: SLF001
    connection.row_factory = __import__("sqlite3").Row
    tables = {
        "facts": "SELECT id, content, version FROM memory_chunks ORDER BY id",
        "events": "SELECT id, content FROM episodic_events ORDER BY id",
        "evidence": "SELECT fact_id, source_event_id, excerpt FROM fact_evidence ORDER BY fact_id, source_event_id",
        "revisions": "SELECT fact_id, version, content FROM fact_revisions ORDER BY fact_id, version",
        "jobs": "SELECT id, status, payload, result FROM memory_jobs ORDER BY id",
        "suppressions": "SELECT fact_id, source_event_id, reason FROM memory_suppressions ORDER BY suppression_id",
    }
    return {name: [dict(row) for row in connection.execute(sql)] for name, sql in tables.items()}


@pytest.mark.asyncio
async def test_missing_source_selection_is_known_noncommitted_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, copy = await _seed_overlap(domain, context)
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
            },
            queue_on_failure=False,
        )
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == copy.id)
        before = _snapshot(domain)
        client.outbox.append(
            {
                "event_id": "queued-before-known-rejection",
                "context": context.model_dump(mode="json"),
                "content": "pending capture",
            }
        )
        client._outbox_drained = False  # noqa: SLF001
        related_payload = {
            "context": context.model_dump(mode="json"),
            "fact_id": target.id,
            "expected_version": target.version,
            "preview_digest": preview["preview_digest"],
            "preview_limit": preview["limit"],
            "selected_fact_ids": [copy.id],
            "selected_source_event_ids": [],
        }
        operations: list[str] = []
        original_dispatch = domain.dispatch

        async def counted_dispatch(operation, payload):
            operations.append(operation)
            return await original_dispatch(operation, payload)

        monkeypatch.setattr(domain, "dispatch", counted_dispatch)
        descriptor = service.descriptor
        assert descriptor is not None
        async with httpx.AsyncClient(trust_env=False) as raw:
            response = await raw.post(
                f"http://{descriptor['host']}:{descriptor['port']}/v1/dispatch",
                headers={"Authorization": f"Bearer {descriptor['token']}"},
                json={
                    "schema_version": 1,
                    "operation": "forget_related",
                    "payload": related_payload,
                },
            )
        assert response.status_code == 409
        result = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_related",
                {key: value for key, value in related_payload.items() if key != "context"},
            )
        )
        assert result == {
            "schema_version": 1,
            "status": "rejected",
            "committed": False,
            "error": "related_forget_source_selection_required",
            "message": "Review and select every required source event before applying related forget.",
        }
        assert candidate["required_source_event_ids"]
        assert _snapshot(domain) == before
        pending = client.outbox.pending()
        assert len(pending) == 1
        assert pending[0]["operation"] == "capture"
        assert pending[0]["payload"]["event_id"] == "queued-before-known-rejection"
        assert operations == ["forget_related", "forget_related"]
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_explicit_source_selection_still_commits(tmp_path: Path) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, copy = await _seed_overlap(domain, context)
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
            },
            queue_on_failure=False,
        )
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == copy.id)
        result = await client.request(
            "forget_related",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "preview_digest": preview["preview_digest"],
                "preview_limit": preview["limit"],
                "selected_fact_ids": [copy.id],
                "selected_source_event_ids": candidate["required_source_event_ids"],
            },
            queue_on_failure=False,
        )
        assert result["status"] == "committed"
        assert result["forgotten"] is True
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_stale_preview_limit_is_known_noncommitted_409(tmp_path: Path) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, copy = await _seed_overlap(domain, context)
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
            },
            queue_on_failure=False,
        )
        candidate = next(item for item in preview["candidates"] if item["fact_id"] == copy.id)
        before = _snapshot(domain)
        client.outbox.append(
            {
                "event_id": "queued-before-stale-rejection",
                "context": context.model_dump(mode="json"),
                "content": "pending capture",
            }
        )
        client._outbox_drained = False  # noqa: SLF001
        result = json.loads(
            await dispatch_tool(
                client,
                context,
                "mnemos_forget_related",
                {
                    "fact_id": target.id,
                    "expected_version": target.version,
                    "preview_digest": preview["preview_digest"],
                    "preview_limit": 20,
                    "selected_fact_ids": [copy.id],
                    "selected_source_event_ids": candidate["required_source_event_ids"],
                },
            )
        )
        assert result == {
            "schema_version": 1,
            "status": "rejected",
            "committed": False,
            "error": "related_forget_preview_stale",
            "message": "The related-forget preview is stale. Request a fresh preview and reuse its digest and limit.",
        }
        assert _snapshot(domain) == before
        pending = client.outbox.pending()
        assert len(pending) == 1
        assert pending[0]["payload"]["event_id"] == "queued-before-stale-rejection"
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_changed_target_version_is_known_stale_preview_409(tmp_path: Path) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, _copy = await _seed_overlap(domain, context)
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
            },
            queue_on_failure=False,
        )
        await domain.dispatch(
            "correct",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "content": "The project release label is a new value.",
            },
        )
        result = await client.request(
            "forget_related",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "preview_digest": preview["preview_digest"],
                "preview_limit": preview["limit"],
            },
            queue_on_failure=False,
        )
        assert result["status"] == "rejected"
        assert result["error"] == "related_forget_preview_stale"
        assert result["committed"] is False
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_out_of_preview_selection_rejection_does_not_leak_fact_identity(
    tmp_path: Path,
) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, _copy = await _seed_overlap(domain, context)
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
            },
            queue_on_failure=False,
        )
        foreign_id = "foreign-fact-id-that-must-not-appear"
        result = await client.request(
            "forget_related",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "preview_digest": preview["preview_digest"],
                "preview_limit": preview["limit"],
                "selected_fact_ids": [foreign_id],
                "selected_source_event_ids": [],
            },
            queue_on_failure=False,
        )
        assert result == {
            "schema_version": 1,
            "status": "rejected",
            "committed": False,
            "error": "related_forget_selection_invalid",
            "message": "Selected related-forget IDs must come from the reviewed preview.",
        }
        assert foreign_id not in json.dumps(result)
        assert _snapshot(domain)["suppressions"] == []
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_out_of_preview_source_rejection_preserves_snapshot_and_hermes_path(
    tmp_path: Path,
) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        _, target, copy = await _seed_overlap(domain, context)
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
            },
            queue_on_failure=False,
        )
        client.outbox.append(
            {
                "event_id": "queued-before-invalid-source",
                "context": context.model_dump(mode="json"),
                "content": "pending capture",
            }
        )
        client._outbox_drained = False  # noqa: SLF001
        before = _snapshot(domain)
        payload = {
            "fact_id": target.id,
            "expected_version": target.version,
            "preview_digest": preview["preview_digest"],
            "preview_limit": preview["limit"],
            "selected_fact_ids": [copy.id],
            "selected_source_event_ids": ["source-not-in-reviewed-preview"],
        }

        result = await client.request(
            "forget_related",
            {"context": context.model_dump(mode="json"), **payload},
            queue_on_failure=False,
        )
        assert result == {
            "schema_version": 1,
            "status": "rejected",
            "committed": False,
            "error": "related_forget_selection_invalid",
            "message": "Selected related-forget IDs must come from the reviewed preview.",
        }
        hermes_result = json.loads(
            await dispatch_tool(client, context, "mnemos_forget_related", payload)
        )
        assert hermes_result == result
        assert _snapshot(domain) == before
        pending = client.outbox.pending()
        assert len(pending) == 1
        assert pending[0]["payload"]["event_id"] == "queued-before-invalid-source"
    finally:
        await client.close()
        service.stop()
        domain.engine.store.close()


@pytest.mark.asyncio
async def test_unexpected_value_error_remains_degraded_500(tmp_path: Path) -> None:
    class Exploding:
        async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            _ = operation, payload
            raise ValueError("unexpected internal failure")

    service = MnemosService(
        dispatcher=Exploding(),
        descriptor_path=tmp_path / "runtime" / "descriptor.json",
        lock_path=tmp_path / "runtime" / "service.lock",
    )
    service.start()
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        result = await client.request(
            "forget_related", {"fact_id": "foreign", "expected_version": 1}, queue_on_failure=False
        )
        assert result["status"] == "degraded"
        assert result["committed"] is False
        assert "HTTP 500" in result["error"]
    finally:
        await client.close()
        service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,body",
    [
        (
            "forget_related",
            {
                "schema_version": 1,
                "status": "rejected",
                "committed": False,
                "error": "related_forget_source_selection_required",
                "message": "Review and select every required source event before applying related forget.",
                "private": "do-not-forward",
            },
        ),
        (
            "forget_related",
            {
                "schema_version": 1,
                "status": "rejected",
                "committed": False,
                "error": "related_forget_selection_invalid",
                "message": "Selected related-forget IDs must come from the reviewed preview.",
                "private": "do-not-forward",
            },
        ),
        (
            "forget_related",
            {
                "schema_version": True,
                "status": "rejected",
                "committed": False,
                "error": "related_forget_source_selection_required",
                "message": "Review and select every required source event before applying related forget.",
            },
        ),
        ("forget_related", {"schema_version": 1, "error": "unknown_private_error"}),
        ("forget_related", ["private response"]),
        (
            "inspect",
            {
                "schema_version": 1,
                "status": "rejected",
                "committed": False,
                "error": "related_forget_source_selection_required",
                "message": "Review and select every required source event before applying related forget.",
            },
        ),
        (
            "inspect",
            {
                "schema_version": 1,
                "status": "rejected",
                "committed": False,
                "error": "related_forget_selection_invalid",
                "message": "Selected related-forget IDs must come from the reviewed preview.",
            },
        ),
    ],
)
async def test_unrecognized_409_is_not_surfaced(tmp_path: Path, operation: str, body: Any) -> None:
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        outbox_path=tmp_path / "outbox.json",
        auto_start=False,
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(409, json=body, request=request))
    client._http_clients[asyncio.get_running_loop()] = httpx.AsyncClient(  # noqa: SLF001
        transport=transport
    )
    descriptor = {"host": "127.0.0.1", "port": 1, "token": "test"}
    try:
        with pytest.raises(ServiceClientError, match="HTTP 409"):
            await client._dispatch_http(operation, {}, descriptor)  # noqa: SLF001
    finally:
        await client.close()
