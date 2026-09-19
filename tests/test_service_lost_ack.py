"""Durable capture replay when the service loses its HTTP acknowledgement."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest

from mnemos.claude_identity import scoped_event_id
from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


class NoEmbeddingCalls(SimpleEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        raise AssertionError("The provisional capture must not embed before replay.")


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_losses", [1, 2], ids=["one_lost_ack", "two_lost_acks"])
@pytest.mark.parametrize(
    "identity_status", ["normal", "provisional_unassociated"], ids=["extract", "capture_only"]
)
async def test_committed_capture_lost_ack_replays_once_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ack_losses: int,
    identity_status: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="profile-lost-ack", enabled=True)
    registry_path = tmp_path / "scopes.json"
    registry.save(registry_path)
    descriptor = tmp_path / "runtime.json"
    database = tmp_path / "memory.sqlite"
    outbox_path = tmp_path / "outbox.json"
    context = registry.resolve(
        project,
        user_profile_id="profile-lost-ack",
        session_id="session-lost-ack",
        host="claude-code",
    )
    content = f"The {identity_status} lost acknowledgement uses cedar-ack-731."
    source_ref = "prompt:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    event_id = (
        scoped_event_id(context, source_ref, "user", content)
        if identity_status == "provisional_unassociated"
        else "lost-ack-normal-event"
    )
    payload = {
        "identity_version": 2 if identity_status == "provisional_unassociated" else 1,
        "identity_status": identity_status,
        "event_id": event_id,
        "context": context.model_dump(mode="json"),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": content,
        "source_kind": (
            "claude_code_transcript"
            if identity_status == "provisional_unassociated"
            else "test_capture"
        ),
        "source_ref": source_ref,
    }
    expected_kind = "capture_only" if identity_status == "provisional_unassociated" else "extract"
    expected_status = "deferred" if identity_status == "provisional_unassociated" else "pending"

    def start() -> tuple[MnemosService, SQLiteStore]:
        current_registry = ScopeRegistry.load(registry_path)
        store = SQLiteStore(str(database))
        config = MnemosConfig(
            modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
        )
        engine = MnemosEngine(
            config=config,
            llm=UnavailableLLMProvider(),
            embedder=NoEmbeddingCalls(),
            store=store,
            scope_registry=current_registry,
            require_scope_context=True,
        )
        service = MnemosService(
            dispatcher=LocalMemoryService(engine, current_registry),
            descriptor_path=descriptor,
            lock_path=tmp_path / "runtime.lock",
            startup_timeout=2,
            dispatch_timeout=3,
        )
        service.start()
        return service, store

    service: MnemosService | None = None
    store: SQLiteStore | None = None
    restarted_service: MnemosService | None = None
    restarted_store: SQLiteStore | None = None
    client = MnemosClient(
        descriptor_path=descriptor,
        outbox_path=outbox_path,
        auto_start=False,
        request_timeout=2,
    )
    dropped = 0
    dispatcher_committed_before_drop = False
    import mnemos.service as service_module

    original_write_json = service_module._RequestHandler._write_json

    def drop_committed_response(handler: Any, status: int, body: Mapping[str, Any]) -> None:
        nonlocal dropped, dispatcher_committed_before_drop
        if (
            dropped < ack_losses
            and status == 200
            and (body.get("committed") is True or body.get("status") == "committed")
        ):
            dropped += 1
            dispatcher_committed_before_drop = True
            raise ConnectionAbortedError("simulated lost acknowledgement")
        original_write_json(handler, status, body)

    monkeypatch.setattr(service_module._RequestHandler, "_write_json", drop_committed_response)
    replay_client: MnemosClient | None = None
    try:
        service, store = start()
        queued = await client.capture(payload)
        assert dropped == 1
        assert dispatcher_committed_before_drop
        assert queued["status"] == "queued"
        assert queued["committed"] is False
        assert len(client.outbox.pending()) == 1
        assert len(store.list_episodic_events()) == 1
        assert len(store.list_jobs()) == 1
        assert store.list_jobs()[0].kind == expected_kind
        event_ids = [event.id for event in store.list_episodic_events()]
        job_ids = [job.id for job in store.list_jobs()]

        await client.close()
        service.stop()
        store.close()
        service = None
        store = None

        restarted_service, restarted_store = start()
        health = restarted_service.submit("health", {"context": context.model_dump(mode="json")})
        assert health["status"] == "running"
        replay_client = MnemosClient(
            descriptor_path=descriptor,
            outbox_path=outbox_path,
            auto_start=False,
            request_timeout=2,
        )
        assert [event.id for event in restarted_store.list_episodic_events()] == event_ids
        assert [job.id for job in restarted_store.list_jobs()] == job_ids
        replay = await replay_client.replay_outbox()
        assert replay["acknowledged"] == (1 if ack_losses == 1 else 0)
        assert replay["failed"] == (0 if ack_losses == 1 else 1)
        assert replay["pending"] == (0 if ack_losses == 1 else 1)
        assert [event.id for event in restarted_store.list_episodic_events()] == event_ids
        assert [job.id for job in restarted_store.list_jobs()] == job_ids
        assert restarted_store.list_jobs()[0].kind == expected_kind

        if ack_losses == 2:
            replay = await replay_client.replay_outbox()
            assert replay["acknowledged"] == 1
            assert replay["failed"] == 0
            assert replay["pending"] == 0
            assert [event.id for event in restarted_store.list_episodic_events()] == event_ids
            assert [job.id for job in restarted_store.list_jobs()] == job_ids

        duplicate = await replay_client.capture(payload)
        assert duplicate["status"] == "committed"
        assert duplicate["processing_status"] == expected_status
        assert duplicate["job_id"] == job_ids[0]
        assert duplicate["fact_eligible"] is (identity_status == "normal")
        assert [event.id for event in restarted_store.list_episodic_events()] == event_ids
        assert [job.id for job in restarted_store.list_jobs()] == job_ids
    finally:
        await client.close()
        if replay_client is not None:
            await replay_client.close()
        if service is not None:
            service.stop()
        if store is not None:
            store.close()
        if restarted_service is not None:
            restarted_service.stop()
        if restarted_store is not None:
            restarted_store.close()


@pytest.mark.asyncio
async def test_processed_correction_lost_ack_retries_after_service_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="hermes")
    descriptor = tmp_path / "runtime.json"
    database = tmp_path / "memory.sqlite"

    def start() -> tuple[MnemosService, SQLiteStore]:
        store = SQLiteStore(str(database))
        engine = MnemosEngine(
            config=MnemosConfig(
                modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
            ),
            llm=UnavailableLLMProvider(),
            embedder=SimpleEmbeddingProvider(),
            store=store,
        )
        current_service = MnemosService(
            dispatcher=LocalMemoryService(engine, registry),
            descriptor_path=descriptor,
            lock_path=tmp_path / "runtime.lock",
        )
        current_service.start()
        return current_service, store

    service, store = start()
    client = MnemosClient(
        descriptor_path=descriptor,
        outbox_path=tmp_path / "outbox.json",
        auto_start=False,
        request_timeout=5,
    )
    import mnemos.service as service_module

    original_write_json = service_module._RequestHandler._write_json
    lost_reply: dict[str, Any] = {}

    def drop_processed_correction(handler: Any, status: int, body: Mapping[str, Any]) -> None:
        job_id = body.get("job_id")
        job = store.get_job(job_id) if job_id else None
        if status == 200 and job is not None and job.kind == "correct" and not lost_reply:
            lost_reply.update(body)
            # The first correction has taken effect before the client knows
            # whether the service accepted it.
            service.submit("maintenance", {})
            assert store.get_job(job_id).status == "completed"
            raise ConnectionAbortedError("lost processed correction acknowledgement")
        original_write_json(handler, status, body)

    monkeypatch.setattr(service_module._RequestHandler, "_write_json", drop_processed_correction)
    try:
        service.submit("remember", {"context": context.model_dump(), "content": "Use React"})
        service.submit("maintenance", {})
        fact = store.get_all()[0]
        payload = {
            "context": context.model_dump(),
            "fact_id": fact.id,
            "expected_version": 1,
            "content": "Use Svelte",
            "event_id": "correction-lost-ack",
        }
        first = await client.request("correct", payload, queue_on_failure=False)
        assert first["status"] == "degraded"
        assert first["committed"] is False
        assert lost_reply["status"] == "committed"
        assert store.get(fact.id).version == 2
        assert client.outbox.pending() == []
        jobs = store.list_jobs()
        events = store.list_episodic_events()
        revisions = store.list_fact_revisions(fact.id)
        await client.close()
        service.stop()
        store.close()
        service, store = start()
        client = MnemosClient(
            descriptor_path=descriptor,
            outbox_path=tmp_path / "outbox.json",
            auto_start=False,
        )
        retried = await client.request("correct", payload, queue_on_failure=False)
        assert retried["status"] == "committed"
        assert retried["job_id"] == lost_reply["job_id"]
        assert retried["event_id"] == lost_reply["event_id"]
        assert retried["processing_status"] == "consolidated"
        assert store.list_jobs() == jobs
        assert store.list_episodic_events() == events
        assert store.list_fact_revisions(fact.id) == revisions
        assert store.get(fact.id).version == 2
        assert store.get(fact.id).content == "Use Svelte"
    finally:
        await client.close()
        service.stop()
        store.close()
