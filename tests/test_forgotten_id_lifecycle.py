"""A delayed assistant transcript must not revive a forgotten fact by ID."""

from datetime import datetime, timezone

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore
from tests.test_memory_service import service


@pytest.mark.asyncio
async def test_delayed_forget_narration_stays_suppressed_after_reopen(tmp_path):
    memory, context = service(tmp_path)
    for flag in type(memory.engine.config.modules).model_fields:
        setattr(memory.engine.config.modules, flag, False)
    old = "The project release label is cedar-ridge-731."
    new = "The project release label is maple-ridge-842."
    try:
        await memory.dispatch("remember", {"context": context.model_dump(), "content": old})
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        await memory.dispatch(
            "correct",
            {
                "context": context.model_dump(),
                "fact_id": target.id,
                "expected_version": 1,
                "content": new,
            },
        )
        await memory.maintenance.run_once(force=True)
        assert memory.engine.store.get(target.id).version == 2
        controls = []
        for name, profile in (("foreign-project", "local"), ("foreign-profile", "other")):
            project = tmp_path / name
            project.mkdir()
            memory.registry.register_project(project, user_profile_id=profile, enabled=True)
            foreign = memory.registry.resolve(
                project, user_profile_id=profile, session_id="control", host="hermes"
            )
            await memory.dispatch("remember", {"context": foreign.model_dump(), "content": new})
            await memory.maintenance.run_once(force=True)
            controls.append(
                next(
                    fact.model_dump()
                    for fact in memory.engine.store.get_all()
                    if fact.metadata["scope_id"] == foreign.project_id
                )
            )
        await memory.dispatch(
            "forget",
            {"context": context.model_dump(), "fact_id": target.id, "expected_version": 2},
        )
        event = CaptureEvent(
            event_id="delayed-native-assistant-row-40",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="assistant",
            content=(
                f"The asserted release-label fact has ID: {target.id}, Version: 2, "
                f'Content: "{new}". I will now forget it. '
                "An unrelated clause in this same assistant event is also suppressed."
            ),
            source_kind="hermes_memory_provider",
            source_ref="transcript:v2:40",
        )
        first = memory.capture(event)
        assert first["processing_status"] == "suppressed"
        assert first["job_id"] is None
        assert memory.engine.store.get_episodic_event(first["event_id"]) is None
        registry, config = memory.registry, memory.engine.config
        memory.engine.store.close()
        memory = LocalMemoryService(
            MnemosEngine(
                config=config,
                store=SQLiteStore(str(tmp_path / "memory.sqlite")),
                embedder=SimpleEmbeddingProvider(),
                llm=UnavailableLLMProvider(),
            ),
            registry,
        )
        replay = memory.capture(event)
        assert replay["processing_status"] == "suppressed"
        assert replay["event_id"] == first["event_id"]
        assert await memory.maintenance.run_once(force=True) == 0
        for control in controls:
            assert memory.engine.store.get(control["id"]).model_dump() == control
        assert memory.engine.store.get(target.id) is None

        unrelated = event.model_copy(
            update={"event_id": "unrelated-row", "content": "The build marker is slate-safe."}
        )
        assert memory.capture(unrelated)["processing_status"] == "pending"
        user = event.model_copy(update={"event_id": "new-user-row", "role": "user"})
        assert memory.capture(user)["processing_status"] == "pending"
        await memory.dispatch("remember", {"context": context.model_dump(), "content": new})
        await memory.maintenance.run_once(force=True)
        reteach = [
            fact
            for fact in memory.engine.store.get_all()
            if fact.content == new and fact.metadata["scope_id"] == context.project_id
        ]
        assert len(reteach) == 1
        assert reteach[0].id != target.id
        assert memory.capture(event)["processing_status"] == "suppressed"
    finally:
        memory.engine.store.close()
