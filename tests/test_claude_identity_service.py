"""Scoped Claude identity transitions against the real persistent store."""

from datetime import datetime, timezone
import hashlib
import json

import pytest

from mnemos.claude_identity import legacy_event_id, scoped_event_id
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


@pytest.fixture
def memory_contexts(tmp_path):
    registry = ScopeRegistry()
    contexts = []
    for name, profile in (("a", "local"), ("b", "local"), ("c", "other")):
        path = tmp_path / name
        path.mkdir()
        registry.register_project(path, user_profile_id=profile, enabled=True)
        contexts.append(
            registry.resolve(
                path, user_profile_id=profile, session_id="same-session", host="claude-code"
            )
        )
    config = MnemosConfig(
        modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
    )
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    engine = MnemosEngine(
        config=config, store=store, embedder=SimpleEmbeddingProvider(), llm=UnavailableLLMProvider()
    )
    memory = LocalMemoryService(engine, registry)
    try:
        yield memory, contexts
    finally:
        store.close()


def event(context, *, legacy=False, content="For this project, use SQLite", role="user"):
    ref = "native-prompt-1"
    identity = (
        legacy_event_id(context.session_id, ref, role, content)
        if legacy
        else scoped_event_id(context, ref, role, content)
    )
    return CaptureEvent(
        identity_version=1 if legacy else 2,
        event_id=identity,
        context=context,
        source_ref=ref,
        source_kind="claude_code_transcript",
        role=role,
        content=content,
        occurred_at=datetime.now(timezone.utc),
    )


def test_same_native_event_is_independent_across_projects_and_profiles(memory_contexts):
    memory, contexts = memory_contexts
    results = [memory.capture(event(context)) for context in contexts]
    assert len({r["event_id"] for r in results}) == 3
    assert all(r["status"] == "committed" for r in results)
    assert len(memory.engine.store.list_episodic_events()) == 3
    assert len(memory.engine.store.list_jobs()) == 3
    assert memory.capture(event(contexts[0]))["event_id"] == results[0]["event_id"]
    assert len(memory.engine.store.list_episodic_events()) == 3


def test_legacy_exact_replay_deduplicates_but_foreign_project_does_not(memory_contexts):
    memory, contexts = memory_contexts
    original = memory.capture(event(contexts[0], legacy=True))
    replay = memory.capture(event(contexts[0]))
    other = memory.capture(event(contexts[1]))
    assert replay["event_id"] == original["event_id"]
    assert other["event_id"] != original["event_id"]
    assert len(memory.engine.store.list_episodic_events()) == 2
    assert len(memory.engine.store.list_jobs()) == 2


@pytest.mark.asyncio
async def test_legacy_forgotten_source_stays_suppressed_after_restart(memory_contexts):
    memory, contexts = memory_contexts
    original = memory.capture(event(contexts[0], legacy=True, role="assistant"))
    assert await memory.maintenance.run_once(force=True) == 1
    fact = memory.engine.store.get_all()[0]
    # Baseline role labeling makes fact text differ from its source conversation.
    assert fact.content != event(contexts[0], role="assistant").content
    assert memory.engine.store.forget_fact(fact.id, expected_version=fact.version)
    assert memory.engine.store.get_episodic_event(original["event_id"]) is None
    db_path = memory.engine.store.db_path
    memory.engine.store.close()
    reopened = SQLiteStore(str(db_path))
    memory.engine._store = reopened
    try:
        for context in contexts[:2]:
            replay = memory.capture(event(context, role="assistant"))
            assert replay["processing_status"] == "suppressed"
            assert replay["reason"] == "legacy_source_suppressed"
        assert reopened.list_episodic_events() == []
        assert reopened.get_all() == []
        # Another profile never shares the old source identity.
        assert (
            memory.capture(event(contexts[2], role="assistant"))["processing_status"] == "pending"
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["content", "source_ref", "project", "host"])
def test_v2_identity_cannot_select_an_unrelated_legacy_tombstone(memory_contexts, change):
    memory, contexts = memory_contexts
    valid = event(contexts[0])
    old = event(contexts[0], legacy=True)
    old_key = hashlib.sha256(
        json.dumps(
            [contexts[0].user_profile_id, contexts[0].host, contexts[0].session_id, old.event_id]
        ).encode()
    ).hexdigest()
    memory.engine.store.record_suppression(source_event_id=old_key, reason="explicit_forget")
    updates = (
        {"content": "An unrelated fact"}
        if change == "content"
        else (
            {"source_ref": "different-native-ref"}
            if change == "source_ref"
            else {
                "context": (
                    contexts[1]
                    if change == "project"
                    else contexts[0].model_copy(update={"host": "codex"})
                )
            }
        )
    )
    result = memory.capture(valid.model_copy(update=updates))
    assert result["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


@pytest.mark.parametrize("content", ["Contact test@example.com", "Contact [REDACTED_EMAIL]"])
def test_redacted_v2_identity_fails_closed_without_new_source(memory_contexts, content):
    memory, contexts = memory_contexts
    result = memory.capture(event(contexts[0], content=content))
    assert result["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


@pytest.mark.asyncio
async def test_processed_legacy_replay_after_evidence_retention_stays_deduplicated(memory_contexts):
    from datetime import timedelta

    memory, contexts = memory_contexts
    original = memory.capture(event(contexts[0], legacy=True))
    assert await memory.maintenance.run_once(force=True) == 1
    assert (
        memory.engine.store.prune_processed_evidence(
            before=datetime.now(timezone.utc) + timedelta(days=1)
        )
        == 1
    )
    assert memory.engine.store.get_episodic_event(original["event_id"]) is None
    replay = memory.capture(event(contexts[0]))
    assert replay["event_id"] == original["event_id"]
    assert replay["processing_status"] == "consolidated"
    assert len(memory.engine.store.list_jobs()) == 1
    assert memory.engine.store.list_episodic_events() == []


def test_whitespace_variant_cannot_bypass_legacy_source_suppression(memory_contexts):
    memory, contexts = memory_contexts
    original = event(contexts[0], legacy=True)
    key = hashlib.sha256(
        json.dumps(
            [
                contexts[0].user_profile_id,
                contexts[0].host,
                contexts[0].session_id,
                original.event_id,
            ]
        ).encode()
    ).hexdigest()
    memory.engine.store.record_suppression(source_event_id=key, reason="explicit_forget")
    replay = event(contexts[0], content=original.content + " ")
    assert memory.capture(replay)["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


@pytest.mark.parametrize("reference", ["", "   "])
def test_blank_native_reference_is_rejected(memory_contexts, reference):
    memory, contexts = memory_contexts
    original = event(contexts[0])
    changed = original.model_copy(
        update={
            "source_ref": reference,
            "event_id": scoped_event_id(contexts[0], reference, original.role, original.content),
        }
    )
    assert memory.capture(changed)["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


@pytest.mark.parametrize("role", ["tool", "system"])
def test_claude_transcript_identity_rejects_nonconversation_roles(memory_contexts, role):
    memory, contexts = memory_contexts
    assert memory.capture(event(contexts[0], role=role))["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


def test_v2_source_cannot_be_reassigned_to_global_scope(memory_contexts):
    memory, contexts = memory_contexts
    assert (
        memory.capture(event(contexts[0]).model_copy(update={"scope": "global"}))["status"]
        == "rejected"
    )
    assert memory.engine.store.list_episodic_events() == []


def test_invalid_identity_is_rejected_even_when_its_source_is_suppressed(memory_contexts):
    memory, contexts = memory_contexts
    valid = event(contexts[0])
    key = hashlib.sha256(
        json.dumps(
            [
                contexts[0].user_profile_id,
                contexts[0].host,
                contexts[0].session_id,
                valid.event_id,
            ]
        ).encode()
    ).hexdigest()
    memory.engine.store.record_suppression(source_event_id=key, reason="explicit_forget")
    invalid = valid.model_copy(update={"content": "Changed content under a forgotten ID"})
    assert memory.capture(invalid)["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


@pytest.mark.asyncio
async def test_real_adapter_reused_native_prompt_commits_in_both_projects(
    memory_contexts, tmp_path
):
    from mnemos.adapters.claude_code import ClaudeCodeAdapter

    memory, contexts = memory_contexts

    class ServiceClient:
        async def capture(self, payload):
            return memory.capture(CaptureEvent.model_validate(payload))

    adapter = ClaudeCodeAdapter(
        client=ServiceClient(),
        registry=memory.registry,
        user_profile_id="local",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    for context in contexts[:2]:
        result = await adapter.handle(
            "UserPromptSubmit",
            {
                "session_id": context.session_id,
                "cwd": memory.registry.projects[context.project_id].root,
                "prompt": "Keep the project convention.",
                "prompt_id": "shared-native-prompt",
            },
            include_retrieval=False,
        )
        assert result.committed is True
        assert result.captured == 1
    assert len(memory.engine.store.list_episodic_events()) == 2
    assert len(memory.engine.store.list_jobs()) == 2
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 2


def test_v2_identity_requires_explicit_protocol_marker(memory_contexts):
    memory, contexts = memory_contexts
    value = event(contexts[0]).model_copy(update={"identity_version": 1})
    assert memory.capture(value)["status"] == "rejected"
    assert memory.engine.store.list_episodic_events() == []


def test_old_capture_schema_rejects_new_identity_protocol(memory_contexts):
    from pydantic import BaseModel, ConfigDict, ValidationError, create_model

    # Reproduce the old extra-forbid boundary without importing another checkout.
    class OldCaptureBoundary(BaseModel):
        model_config = ConfigDict(extra="forbid")

    fields = {
        name: (field.annotation, field)
        for name, field in CaptureEvent.model_fields.items()
        if name != "identity_version"
    }
    old_schema = create_model("OldCaptureSchema", __base__=OldCaptureBoundary, **fields)
    _, contexts = memory_contexts
    with pytest.raises(ValidationError, match="identity_version"):
        old_schema.model_validate(event(contexts[0]).model_dump())
