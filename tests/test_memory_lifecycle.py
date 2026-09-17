from datetime import datetime, timedelta, timezone
import asyncio

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.utils.storage import SQLiteStore
from tests.test_memory_service import service


@pytest.mark.asyncio
async def test_forget_wins_over_an_already_running_extraction(tmp_path, monkeypatch):
    memory, context = service(tmp_path)
    try:
        for flag in type(memory.engine.config.modules).model_fields:
            setattr(memory.engine.config.modules, flag, False)
        await memory.dispatch(
            "remember", {"context": context.model_dump(), "content": "Use SQLite"}
        )
        await memory.maintenance.run_once(force=True)
        original = memory.engine.store.get_all()[0]
        memory.capture(
            CaptureEvent(
                event_id="in-flight",
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content="Use SQLite",
                source_kind="user_prompt",
            )
        )
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def delayed_embedding(embedder, content):
            entered.set()
            await proceed.wait()
            return embedder.embed(content)

        monkeypatch.setattr("mnemos.memory_service.embed_text_async", delayed_embedding)
        task = asyncio.create_task(memory.maintenance.run_once(force=True))
        await asyncio.wait_for(entered.wait(), timeout=2)
        await memory.dispatch(
            "forget",
            {
                "context": context.model_dump(),
                "fact_id": original.id,
                "expected_version": original.version,
            },
        )
        proceed.set()
        assert await asyncio.wait_for(task, timeout=2) == 1
        assert not memory.engine.store.get_all()
        assert any(
            (job.result or {}).get("decision") == "suppressed"
            for job in memory.engine.store.list_jobs()
        )
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_forget_survives_restart_replay_and_pending_correction(tmp_path):
    memory, context = service(tmp_path)
    payload = {"context": context.model_dump(), "event_id": "first", "content": "Use React"}
    await memory.dispatch("remember", payload)
    await memory.maintenance.run_once(force=True)
    chunk = memory.engine.store.get_all()[0]
    await memory.dispatch(
        "correct",
        {
            **payload,
            "event_id": "pending",
            "content": "Use Svelte",
            "fact_id": chunk.id,
            "expected_version": 1,
        },
    )
    await memory.dispatch(
        "forget", {"context": context.model_dump(), "fact_id": chunk.id, "expected_version": 1}
    )
    assert memory.engine.spreading_activation.get_node(chunk.id) is None
    assert not memory.engine.store.get_all()
    assert not memory.engine.store.list_episodic_events()
    assert not memory.engine.store.list_fact_evidence(chunk.id)
    assert not memory.engine.store.list_jobs(status="pending")
    registry, embedder, llm = memory.registry, memory.engine.embedder, memory.engine.llm
    memory.engine.store.close()
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    memory = LocalMemoryService(MnemosEngine(store=store, embedder=embedder, llm=llm), registry)
    try:
        assert (await memory.dispatch("remember", payload))["processing_status"] == "suppressed"
        event = CaptureEvent(
            event_id="new-auto",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content="Use React",
            source_kind="user_prompt",
        )
        assert memory.capture(event)["processing_status"] == "suppressed"
        assert not store.list_episodic_events()
        assert await memory.maintenance.run_once(force=True) == 0
        # An explicit fresh instruction can intentionally restore the assertion.
        await memory.dispatch("remember", {**payload, "event_id": "fresh-explicit"})
        assert await memory.maintenance.run_once(force=True) == 1
        assert len(store.get_all()) == 1
        assert store.get_all()[0].id != chunk.id
    finally:
        store.close()


@pytest.mark.asyncio
async def test_retention_only_prunes_processed_raw_evidence(tmp_path):
    memory, context = service(tmp_path)
    try:
        await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "event_id": "accepted", "content": "Use SQLite"},
        )
        await memory.maintenance.run_once(force=True)
        fact = memory.engine.store.get_all()[0]
        event = CaptureEvent(
            event_id="pending",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content="Unprocessed evidence",
            source_kind="user_prompt",
        )
        memory.capture(event)
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=1)
        assert memory.engine.store.prune_processed_evidence(before=cutoff) == 1
        assert len(memory.engine.store.list_episodic_events()) == 1
        assert memory.engine.store.get(fact.id).content == "Use SQLite"
        assert memory.engine.store.list_fact_evidence(fact.id)[0].excerpt == "Use SQLite"
        replay = await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "event_id": "accepted", "content": "Use SQLite"},
        )
        assert replay["processing_status"] == "consolidated"
        assert len(memory.engine.store.list_episodic_events()) == 1
        assert memory.engine.store.get(fact.id).observation_count == 1
    finally:
        memory.engine.store.close()
