from datetime import datetime, timezone
import json

import pytest

from mnemos.contracts import CaptureEvent, FactProposal
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.scopes import context_metadata
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import SQLiteStore


def service(tmp_path):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        llm=UnavailableLLMProvider(), embedder=SimpleEmbeddingProvider(), store=store
    )
    return LocalMemoryService(engine, registry), context


@pytest.mark.asyncio
async def test_project_enable_pause_persists_through_single_owner(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(registry_path))
    memory, context = service(tmp_path)
    try:
        paused = await memory.dispatch(
            "configure_project",
            {"path": str(tmp_path), "user_profile_id": "local", "enabled": False},
        )
        assert paused["project"]["id"] == context.project_id
        with pytest.raises(ValueError, match="not enabled"):
            memory.registry.validate_context(context)
        assert ScopeRegistry.load(registry_path).projects[context.project_id].enabled is False
        enabled = await memory.dispatch(
            "configure_project",
            {"path": str(tmp_path), "user_profile_id": "local", "enabled": True},
        )
        assert enabled["project"]["id"] == context.project_id
        assert memory.registry.validate_context(context) == context
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_cognition_probe_requires_input_specific_extraction_without_writes(tmp_path):
    from mnemos.utils.llm import LLMProvider

    class InputSpecificFixture(LLMProvider):
        wrong = False

        async def predict(self, prompt, **kwargs):
            source = json.loads(prompt.split("SOURCE EVENTS:\n", 1)[1])[0]
            return json.dumps(
                {
                    "decision": "facts",
                    "facts": [
                        {
                            "operation": "add",
                            "content": (
                                "An unrelated canned preference"
                                if self.wrong
                                else source["content"]
                            ),
                            "evidence_ids": [source["id"]],
                        }
                    ],
                }
            )

        async def classify(self, prompt, labels):
            return {}

    memory, context = service(tmp_path)
    try:
        payload = {"context": context.model_dump()}
        assert (await memory.dispatch("probe_cognition", payload))["status"] == "failed"
        fixture = InputSpecificFixture()
        memory.engine._llm = fixture
        probe = await memory.dispatch("probe_cognition", payload)
        assert probe["input_specific_extraction"] is True
        assert memory.engine.store.get_all() == []
        assert memory.engine.store.list_episodic_events() == []
        fixture.wrong = True
        assert (await memory.dispatch("probe_cognition", payload))[
            "input_specific_extraction"
        ] is False
        assert (await memory.dispatch("health", {}))["provider_state"]["cognition"][
            "status"
        ] == "failed"
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutable_enabled", [True, False])
async def test_automatic_correction_switch_changes_lifecycle_not_retrieval_query(
    tmp_path, mutable_enabled
):
    memory, context = service(tmp_path)
    try:
        memory.engine.config.modules.surprisal = False
        memory.engine.config.modules.affective = False
        memory.engine.config.modules.mutable_rag = mutable_enabled
        await memory.dispatch("remember", {"context": context.model_dump(), "content": "Use React"})
        await memory.maintenance.run_once(force=True)
        old = memory.engine.store.get_all()[0]
        event = CaptureEvent(
            event_id="migration",
            context=context,
            role="user",
            occurred_at=datetime.now(timezone.utc),
            content="We migrated from React to Svelte",
            source_kind="user_prompt",
        )
        captured = memory.capture(event)
        memory.engine._llm = MockLLMProvider(
            responses={
                "Extract durable facts": json.dumps(
                    {
                        "decision": "facts",
                        "facts": [
                            {
                                "content": "Use Svelte",
                                "operation": "correct",
                                "evidence_ids": [captured["event_id"]],
                                "target_fact_id": old.id,
                                "expected_version": 1,
                            }
                        ],
                    }
                )
            }
        )
        assert await memory.maintenance.run_once(force=True) == 1
        if mutable_enabled:
            assert memory.engine.store.get(old.id).content == "Use Svelte"
            assert memory.engine.store.get(old.id).version == 2
            assert len(memory.engine.store.get_all()) == 1
        else:
            assert memory.engine.store.get(old.id).content == "Use React"
            assert len(memory.engine.store.get_all()) == 2
            assert any(
                c.metadata.get("assertion_status") == "conflict"
                for c in memory.engine.store.get_all()
            )
            conflict = next(
                c
                for c in memory.engine.store.get_all()
                if c.metadata.get("assertion_status") == "conflict"
            )
            registry = memory.registry
            memory.engine.store.close()
            memory = LocalMemoryService(
                MnemosEngine(
                    llm=UnavailableLLMProvider(),
                    embedder=SimpleEmbeddingProvider(),
                    store=SQLiteStore(str(tmp_path / "memory.sqlite")),
                ),
                registry,
            )
            retrieved = await memory.dispatch(
                "retrieve",
                {"context": context.model_dump(), "query": "Use Svelte", "top_k": 5},
            )
            returned = next(f for f in retrieved["facts"] if f["id"] == conflict.id)
            assert returned["assertion_status"] == "conflict"
            assert (
                f"[{conflict.id} version 1; unresolved conflict] Use Svelte" in retrieved["context"]
            )
            assert retrieved["context_token_upper_bound"] == len(retrieved["context"].encode())
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_all_modules_disabled_retains_role_tagged_events_without_cognition(tmp_path):
    memory, context = service(tmp_path)
    try:
        for flag in type(memory.engine.config.modules).model_fields:
            setattr(memory.engine.config.modules, flag, False)
        event = CaptureEvent(
            event_id="response",
            context=context,
            role="assistant",
            occurred_at=datetime.now(timezone.utc),
            content="I suggest using SQLite",
            source_kind="assistant_response",
        )
        memory.capture(event)
        assert await memory.maintenance.run_once(force=True) == 1
        chunk = memory.engine.store.get_all()[0]
        assert chunk.content == "Assistant reported: I suggest using SQLite"
        assert chunk.cognitive_state is None
        assert chunk.metadata["encoding_diagnostics"]["surprisal_status"] == "disabled"
    finally:
        memory.engine.store.close()


def test_capture_is_idempotent_durable_and_does_not_require_a_model(tmp_path):
    memory, context = service(tmp_path)
    event = CaptureEvent(
        event_id="turn1",
        context=context,
        occurred_at=datetime.now(timezone.utc),
        role="user",
        content="For this project, use SQLite",
        source_kind="user_prompt",
    )
    first = memory.capture(event)
    second = memory.capture(event)
    assert first["status"] == second["status"] == "committed"
    assert first["event_id"] == second["event_id"]
    assert memory.engine.store.get_all() == []
    memory.engine.store.close()
    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert len(reopened.list_episodic_events()) == 1
        assert len(reopened.list_jobs()) == 1
        assert reopened.list_jobs()[0].payload["event_id"] == first["event_id"]
    finally:
        reopened.close()


def test_capture_rejects_secrets_and_disabled_projects_before_disk(tmp_path):
    memory, context = service(tmp_path)
    try:
        event = CaptureEvent(
            event_id="turn1",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content="api_key=neverpersist",
            source_kind="user_prompt",
        )
        assert memory.capture(event)["status"] == "rejected"
        assert memory.engine.store.list_episodic_events() == []
        assert memory.engine.store.list_jobs() == []
        memory.registry.projects[context.project_id].enabled = False
        with pytest.raises(ValueError, match="enabled"):
            memory.capture(event.model_copy(update={"content": "Safe content"}))
        assert memory.engine.store.list_episodic_events() == []
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_legacy_review_is_read_only_and_scoped_retrieval_excludes_legacy_rows(tmp_path):
    memory, context = service(tmp_path)
    try:
        for chunk_id, metadata in (
            ("a-foreign", {"user_profile_id": "other-profile", "scope": "project"}),
            ("b-legacy", {}),
            ("c-profile-key", {"profile_id": "other-profile"}),
            ("d-legacy", {}),
        ):
            memory.engine.store.store(MemoryChunk(id=chunk_id, content=chunk_id, metadata=metadata))

        reviewed = await memory.dispatch(
            "legacy_review", {"context": context.model_dump(), "limit": 2}
        )
        assert [item["id"] for item in reviewed["facts"]] == ["b-legacy", "d-legacy"]
        assert reviewed["read_only"] is True
        assert reviewed["migration_required"] is True
        assert reviewed["truncated"] is False
        assert reviewed["next_cursor"] is None

        retrieved = await memory.dispatch(
            "retrieve",
            {"context": context.model_dump(), "query": "b-legacy", "top_k": 5},
        )
        assert all(item["id"] not in {"b-legacy", "d-legacy"} for item in retrieved["facts"])
        assert memory.engine.store.get("b-legacy").content == "b-legacy"
    finally:
        memory.engine.store.close()


def test_fact_proposal_cannot_be_an_unversioned_correction():
    with pytest.raises(ValueError, match="expected version"):
        FactProposal(
            content="Now use Rust",
            evidence_ids=["source1"],
            operation="correct",
            target_fact_id="fact1",
        )
    with pytest.raises(ValueError):
        FactProposal(content="Use Rust", evidence_ids=[], operation="add")


@pytest.mark.asyncio
async def test_explicit_remember_and_versioned_correction_keep_identity(tmp_path):
    memory, context = service(tmp_path)
    try:
        await memory.dispatch(
            "remember",
            {
                "content": "Use React for the UI",
                "context": context.model_dump(),
                "event_id": "remember-1",
            },
        )
        assert await memory.maintenance.run_once(force=True) == 1
        first = memory.engine.store.get_all()[0]
        assert first.content == "Use React for the UI"
        assert memory.engine.store.list_fact_evidence(first.id)
        await memory.dispatch(
            "correct",
            {
                "fact_id": first.id,
                "expected_version": 1,
                "content": "Use Svelte for the UI",
                "context": context.model_dump(),
                "event_id": "correct-1",
            },
        )
        assert await memory.maintenance.run_once(force=True) == 1
        revised = memory.engine.store.get(first.id)
        assert revised.version == 2
        assert revised.content == "Use Svelte for the UI"
        assert memory.engine.spreading_activation.get_node(first.id).content == revised.content
        assert memory.engine.store.list_fact_revisions(first.id)[0].content == first.content
        with pytest.raises(ValueError, match="version conflict"):
            await memory.dispatch(
                "correct",
                {
                    "fact_id": first.id,
                    "expected_version": 1,
                    "content": "Stale update",
                    "context": context.model_dump(),
                },
            )
        await memory.dispatch(
            "remember",
            {"content": revised.content, "context": context.model_dump(), "event_id": "remember-2"},
        )
        assert await memory.maintenance.run_once(force=True) == 1
        assert len(memory.engine.store.get_all()) == 1
        assert memory.engine.store.get(first.id).version == 2
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "",
        "{}",
        '{"decision":"facts","facts":[]}',
        '{"decision":"facts","facts":[{"content":"Invented fact","evidence_ids":["missing"],"operation":"add"}]}',
    ],
)
async def test_bad_extraction_keeps_source_pending(tmp_path, response):
    from mnemos.utils.llm import MockLLMProvider

    memory, context = service(tmp_path)
    llm = MockLLMProvider(responses={"extract durable facts": response})
    memory.engine._llm = llm
    memory.engine.surprisal_gate._llm = llm
    memory.engine.config.surprisal.threshold = 0
    try:
        event = CaptureEvent(
            event_id="source",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content="For this project, use SQLite",
            source_kind="user_prompt",
        )
        captured = memory.capture(event)
        assert await memory.maintenance.run_once(force=True) == 0
        assert memory.engine.store.get_all() == []
        assert memory.engine.store.get_episodic_event(captured["event_id"]).status == "pending"
        assert memory.engine.store.list_jobs()[0].status == "pending"
    finally:
        memory.engine.store.close()
