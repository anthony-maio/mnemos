from datetime import datetime, timedelta, timezone

import pytest

from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


@pytest.mark.asyncio
@pytest.mark.parametrize("spreading", [False, True])
@pytest.mark.parametrize("inject_failure", [False, True])
async def test_correction_absorbs_captured_duplicate_and_forget_suppresses_both_sources(
    tmp_path, monkeypatch, spreading, inject_failure
):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="hermes")
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False, mutable_rag=False, affective=False, sleep=False, spreading=spreading
        )
    )

    def open_service():
        return LocalMemoryService(
            MnemosEngine(
                config=config,
                llm=UnavailableLLMProvider(),
                embedder=SimpleEmbeddingProvider(),
                store=SQLiteStore(str(tmp_path / "memory.sqlite")),
            ),
            registry,
        )

    memory = open_service()
    payloads = [
        dict(
            context=context.model_dump(),
            event_id=f"source-{i}",
            role="user",
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source_kind="hermes_memory_provider",
            content=content,
        )
        for i, content in enumerate(["Use cedar export format.", "Use birch export format."])
    ]
    try:
        receipts = [await memory.dispatch("capture", p) for p in payloads]
        assert await memory.maintenance.run_once(force=True) == 2
        target = next(f for f in memory.engine.store.get_all() if "cedar" in f.content)
        duplicate = next(f for f in memory.engine.store.get_all() if "birch" in f.content)
        repeated = duplicate.model_copy(deep=True, update={"id": "retired-captured-id"})
        memory.engine.store.upsert_canonical(repeated)
        foreign = duplicate.model_copy(
            deep=True,
            update={
                "id": "foreign-control",
                "metadata": {
                    **duplicate.metadata,
                    "scope_id": "another-project",
                    "project_id": "another-project",
                },
            },
        )
        memory.engine.store.upsert_canonical(foreign)
        foreign_snapshot = memory.engine.store.get(foreign.id).model_dump(mode="json")
        correction = await memory.dispatch(
            "correct",
            dict(
                context=context.model_dump(),
                event_id="correction",
                content=duplicate.content,
                fact_id=target.id,
                expected_version=1,
            ),
        )
        if inject_failure:
            store = memory.engine.store
            before = {f.id: f.model_dump(mode="json") for f in store.get_all()}
            original_upsert = store.upsert_canonical

            def fail_after_upsert(chunk):
                original_upsert(chunk)
                raise RuntimeError("injected after duplicate deletion")

            monkeypatch.setattr(store, "upsert_canonical", fail_after_upsert)
            assert await memory.maintenance.run_once(force=True) == 0
            assert {f.id: f.model_dump(mode="json") for f in store.get_all()} == before
            assert len(store.list_fact_evidence(target.id)) == 1
            assert len(store.list_fact_evidence(duplicate.id)) == 1
            assert not store.list_fact_revisions(target.id)
            monkeypatch.setattr(store, "upsert_canonical", original_upsert)
            memory.maintenance.clock = lambda: datetime.now(timezone.utc) + timedelta(seconds=3)
        assert await memory.maintenance.run_once(force=True) == 1
        assert len(memory.engine.store.get_all()) == 2
        assert memory.engine.store.get(foreign.id).model_dump(mode="json") == foreign_snapshot
        revised = memory.engine.store.get(target.id)
        assert revised.version == 2 and revised.content == duplicate.content
        assert revised.metadata["assertion_status"] == "asserted"
        assert revised.metadata["source"] != "captured_passthrough"
        assert memory.engine.store.get(duplicate.id) is None
        assert memory.engine.spreading_activation.get_node(duplicate.id) is None
        if spreading:
            assert (
                memory.engine.spreading_activation.get_node(target.id).content == duplicate.content
            )
        sources = {r["event_id"] for r in [*receipts, correction]}
        assert {
            e.source_event_id for e in memory.engine.store.list_fact_evidence(target.id)
        } == sources
        assert [r.content for r in memory.engine.store.list_fact_revisions(target.id)] == [
            target.content
        ]
        await memory.close()
        memory = open_service()
        assert memory.engine.store.get(target.id).version == 2
        preview = await memory.dispatch(
            "forget_preview",
            dict(context=context.model_dump(), fact_id=target.id, expected_version=2, limit=100),
        )
        assert preview["complete"] and not preview["candidates"]
        assert set(preview["target"]["source_event_ids"]) == sources
        await memory.dispatch(
            "forget_related",
            dict(
                context=context.model_dump(),
                **preview["apply_preconditions"],
                selected_fact_ids=[target.id],
                selected_source_event_ids=[s["source_event_id"] for s in preview["source_events"]],
            ),
        )
        for payload in payloads:
            assert (await memory.dispatch("capture", payload))["processing_status"] == "suppressed"
        await memory.maintenance.run_once(force=True)
        await memory.close()
        memory = open_service()
        assert [f.id for f in memory.engine.store.get_all()] == [foreign.id]
        assert memory.engine.store.get(foreign.id).model_dump(mode="json") == foreign_snapshot
        assert memory.engine.store.is_suppressed(fact_id=duplicate.id)
        assert memory.engine.store.is_suppressed(fact_id=repeated.id)
    finally:
        await memory.close()
