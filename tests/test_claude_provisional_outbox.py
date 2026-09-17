"""A provisional capture survives outage and retry without entering extraction."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path

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
        raise AssertionError("Provisional capture must not call the embedding provider")


@pytest.mark.asyncio
async def test_provisional_outbox_reject_retry_and_reopen(tmp_path: Path) -> None:
    registry = ScopeRegistry(profiles=["profile-a"])
    project = registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    context = registry.resolve(
        tmp_path, user_profile_id="profile-a", session_id="session-a", host="claude-code"
    )
    content = "The provisional outbox canary is cedar-4281."
    source_ref = "prompt:" + hashlib.sha256(content.encode()).hexdigest()
    payload = {
        "identity_version": 2,
        "identity_status": "provisional_unassociated",
        "event_id": scoped_event_id(context, source_ref, "user", content),
        "context": context.model_dump(mode="json"),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": content,
        "source_kind": "claude_code_transcript",
        "source_ref": source_ref,
    }
    descriptor = tmp_path / "runtime.json"
    client = MnemosClient(
        descriptor_path=descriptor,
        outbox_path=tmp_path / "outbox.json",
        auto_start=False,
        request_timeout=3,
    )
    database = tmp_path / "memory.db"

    def start() -> tuple[MnemosService, SQLiteStore]:
        store = SQLiteStore(str(database))
        engine = MnemosEngine(
            config=MnemosConfig(
                modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
            ),
            store=store,
            embedder=NoEmbeddingCalls(),
            llm=UnavailableLLMProvider(),
        )
        service = MnemosService(
            dispatcher=LocalMemoryService(engine, registry),
            descriptor_path=descriptor,
            lock_path=tmp_path / "runtime.lock",
        )
        service.start()
        return service, store

    service = None
    store = None
    try:
        queued = await client.capture(payload)
        assert queued["status"] == "queued"
        assert queued["committed"] is False
        assert (
            client.outbox.pending()[0]["payload"]["identity_status"] == "provisional_unassociated"
        )

        project.enabled = False
        service, store = start()
        rejected = await client.replay_outbox()
        assert rejected["acknowledged"] == 0
        assert rejected["pending"] == 1
        assert store.list_episodic_events() == []

        project.enabled = True
        replay = await client.replay_outbox()
        assert replay["acknowledged"] == 1
        assert replay["pending"] == 0
        await client.request("maintenance", {}, queue_on_failure=False)
        assert len(store.list_episodic_events(status="deferred")) == 1
        assert store.list_jobs()[0].status == "completed"
        assert store.get_all() == []
        await client.close()
        service.stop()
        store.close()
        service, store = start()

        # Simulate a lost acknowledgement by replaying the original payload.
        client.outbox.append(payload)
        replay = await client.replay_outbox()
        assert replay["acknowledged"] == 1
        assert replay["pending"] == 0
        acknowledged = await client.capture(payload)
        assert acknowledged["processing_status"] == "deferred"
        assert acknowledged["fact_eligible"] is False
        assert len(store.list_episodic_events()) == 1
        assert len(store.list_jobs()) == 1
        assert store.get_all() == []
    finally:
        await client.close()
        if service is not None:
            service.stop()
        if store is not None:
            store.close()
