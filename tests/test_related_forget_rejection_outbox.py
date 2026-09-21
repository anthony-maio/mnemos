"""Known rejected selections must not replay otherwise valid queued captures."""

from datetime import datetime, timezone
import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
import httpx

from mnemos.adapters.hermes_tools import dispatch_tool
from mnemos.client import MnemosClient, ServiceClientError
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore
from tests.test_related_forget import _seed_overlap
from tests.test_related_forget_service_errors import _runtime, _snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["client", "hermes"])
async def test_invalid_selection_does_not_replay_valid_outbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str
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
        queued: dict[str, Any] = {
            "event_id": "valid-capture-awaiting-replay",
            "context": context.model_dump(mode="json"),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "role": "user",
            "content": "The local test archive uses cobalt boxes.",
            "source_kind": "hermes_memory_provider",
        }
        client.outbox.append(queued)
        client._outbox_drained = False
        pending_before = client.outbox.pending()
        before = _snapshot(domain)
        operations: list[str] = []
        captures: list[dict[str, Any]] = []
        original = domain.dispatch

        async def dispatch(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            operations.append(operation)
            result: dict[str, Any] = await original(operation, payload)
            if operation == "capture":
                captures.append(result)
            return result

        monkeypatch.setattr(domain, "dispatch", dispatch)
        selection: dict[str, Any] = {
            "fact_id": target.id,
            "expected_version": target.version,
            "preview_digest": preview["preview_digest"],
            "preview_limit": preview["limit"],
            "selected_fact_ids": [copy.id],
            "selected_source_event_ids": ["not-in-preview"],
        }
        if entrypoint == "hermes":
            result = json.loads(
                await dispatch_tool(client, context, "mnemos_forget_related", selection)
            )
        else:
            result = await client.request(
                "forget_related",
                {"context": context.model_dump(mode="json"), **selection},
                queue_on_failure=False,
            )
        assert result["status"] == "rejected"
        assert result["committed"] is False
        assert result["error"] == "related_forget_selection_invalid"
        assert operations == ["forget_related"]
        assert _snapshot(domain) == before
        assert client.outbox.pending() == pending_before

        # Prove the pending record can commit when replay is explicitly requested.
        replay = await client.replay_outbox()
        assert replay["pending"] == 0
        assert operations == ["forget_related", "capture"]
        assert len(captures) == 1 and captures[0]["status"] == "committed"
        stored = domain.engine.store.get_episodic_event(captures[0]["event_id"])
        assert stored is not None and stored.content == queued["content"]
    finally:
        await client.close()
        service.stop()
        cast(SQLiteStore, domain.engine.store).close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,code",
    [
        ("duplicate_fact", "related_forget_selection_invalid"),
        ("duplicate_source", "related_forget_selection_invalid"),
        ("incomplete", "related_forget_preview_incomplete"),
        ("dependency", "related_forget_selection_conflict"),
    ],
)
async def test_review_rejections_preserve_storage_and_valid_outbox(
    tmp_path: Path, kind: str, code: str
) -> None:
    service, domain, context = _runtime(tmp_path)
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    store = cast(SQLiteStore, domain.engine.store)
    try:
        assertion, target, copy = await _seed_overlap(domain, context)
        if kind == "incomplete":
            await domain.dispatch(
                "capture",
                {
                    "event_id": "second-related-copy",
                    "context": context.model_dump(mode="json"),
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "role": "user",
                    "content": f"A second captured copy: {assertion}",
                    "source_kind": "hermes_memory_provider",
                },
            )
            await domain.dispatch("maintenance", {})
        if kind == "dependency":
            first_preview = store.preview_related_forget(
                fact_id=target.id,
                expected_version=target.version,
                user_profile_id=context.user_profile_id,
                project_id=context.project_id,
            )
            source = next(
                item for item in first_preview["candidates"] if item["fact_id"] == copy.id
            )["required_source_event_ids"][0]
            independent = MemoryChunk(
                id="unselected-private-fact",
                content="Keep this independently reviewed note.",
                metadata={
                    "user_profile_id": context.user_profile_id,
                    "scope": "project",
                    "scope_id": context.project_id,
                    "source": "manual",
                },
            )
            store.store(independent)
            store._conn.execute(
                "INSERT INTO fact_evidence "
                "(evidence_id,fact_id,source_event_id,excerpt,source_ref,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    "private-source-link",
                    independent.id,
                    source,
                    independent.content,
                    "manual",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            store._conn.commit()
        preview = await client.request(
            "forget_preview",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "limit": 1 if kind == "incomplete" else 100,
            },
            queue_on_failure=False,
        )
        assert preview["complete"] is (kind != "incomplete")
        selected_facts = [copy.id]
        selected_sources = [
            source
            for candidate in preview["candidates"]
            if candidate["fact_id"] == copy.id
            for source in candidate["required_source_event_ids"]
        ]
        if kind == "duplicate_fact":
            selected_facts *= 2
        if kind == "duplicate_source":
            assert selected_sources
            selected_sources *= 2
        before = _snapshot(domain)
        client.outbox.append(
            {
                "event_id": "valid-pending-review-capture",
                "context": context.model_dump(mode="json"),
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "user",
                "content": "Keep the review queue intact.",
                "source_kind": "hermes_memory_provider",
            }
        )
        client._outbox_drained = False
        pending = client.outbox.pending()
        result = await client.request(
            "forget_related",
            {
                "context": context.model_dump(mode="json"),
                "fact_id": target.id,
                "expected_version": target.version,
                "preview_digest": preview["preview_digest"],
                "preview_limit": preview["limit"],
                "selected_fact_ids": selected_facts,
                "selected_source_event_ids": selected_sources,
            },
            queue_on_failure=False,
        )
        assert result["status"] == "rejected"
        assert result["committed"] is False
        assert result["error"] == code
        assert set(result) == {"schema_version", "status", "committed", "error", "message"}
        assert "unselected-private-fact" not in json.dumps(result)
        assert _snapshot(domain) == before
        assert client.outbox.pending() == pending
    finally:
        await client.close()
        service.stop()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [["private-error"], {"private": "detail"}, None, "unknown"])
async def test_malformed_rejection_code_is_not_forwarded(tmp_path: Path, error: Any) -> None:
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        outbox_path=tmp_path / "outbox.json",
        auto_start=False,
    )
    body = {
        "schema_version": 1,
        "status": "rejected",
        "committed": False,
        "error": error,
        "message": None,
    }
    client._http_clients[asyncio.get_running_loop()] = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(409, json=body))
    )
    try:
        with pytest.raises(ServiceClientError, match="HTTP 409"):
            await client._dispatch_http(
                "forget_related", {}, {"host": "127.0.0.1", "port": 1, "token": "test"}
            )
    finally:
        await client.close()
