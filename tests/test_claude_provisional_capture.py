"""Durable, extraction-ineligible Claude hook fallback coverage."""

from datetime import datetime, timedelta, timezone
import hashlib

import pytest
from pydantic import ValidationError

from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.claude_identity import legacy_event_id, scoped_event_id
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.contracts import CaptureEvent
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ProjectIdentity, ScopeContext, ScopeRegistry, canonical_project_root
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def _service(tmp_path):
    registry = ScopeRegistry(
        projects={
            "project-a": ProjectIdentity(
                id="project-a",
                root=canonical_project_root(tmp_path),
                user_profile_id="profile-a",
                enabled=True,
            )
        }
    )
    context = registry.resolve(
        tmp_path,
        user_profile_id="profile-a",
        session_id="session-a",
        host="claude-code",
    )
    store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    config = MnemosConfig(
        modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
    )
    engine = MnemosEngine(
        config=config,
        store=store,
        embedder=SimpleEmbeddingProvider(),
        llm=UnavailableLLMProvider(),
    )
    return LocalMemoryService(engine, registry), context, store


def _event(
    context: ScopeContext,
    content: str,
    source_ref: str,
    *,
    identity_status: str = "normal",
) -> CaptureEvent:
    return CaptureEvent(
        identity_version=2,
        identity_status=identity_status,
        event_id=scoped_event_id(context, source_ref, "user", content),
        context=context,
        scope="project",
        occurred_at=datetime.now(timezone.utc),
        role="user",
        content=content,
        source_kind="claude_code_transcript",
        source_ref=source_ref,
    )


@pytest.mark.asyncio
async def test_provisional_capture_is_durable_capture_only_and_reopens(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        content = "The release canary is mnemos-fallback-7f4c."
        source_ref = "prompt:" + hashlib.sha256(content.encode()).hexdigest()
        event = _event(context, content, source_ref, identity_status="provisional_unassociated")

        accepted = memory.capture(event)
        assert accepted["status"] == "committed"
        assert accepted["processing_status"] == "deferred"
        assert accepted["identity_status"] == "provisional_unassociated"
        assert accepted["fact_eligible"] is False
        assert store.list_jobs()[0].kind == "capture_only"
        assert store.list_episodic_events()[0].status == "deferred"

        assert await memory.maintenance.run_once(force=True) == 1
        job = store.list_jobs()[0]
        assert job.status == "completed"
        assert job.result == {
            "decision": "identity_pending",
            "fact_eligible": False,
            "fact_ids": [],
        }
        assert store.list_episodic_events()[0].status == "deferred"
        assert store.get_all() == []

        replay = memory.capture(event)
        assert replay["event_id"] == accepted["event_id"]
        assert replay["processing_status"] == "deferred"
        assert replay["fact_eligible"] is False
        assert len(store.list_episodic_events()) == 1
    finally:
        store.close()

    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        assert len(reopened.list_episodic_events(status="deferred")) == 1
        assert reopened.list_jobs()[0].status == "completed"
        assert reopened.get_all() == []
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_transcript_uuid_is_only_extraction_eligible_event(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        content = "The release canary is mnemos-transcript-4c91."
        fallback_ref = "prompt:" + hashlib.sha256(content.encode()).hexdigest()
        fallback = _event(
            context, content, fallback_ref, identity_status="provisional_unassociated"
        )
        transcript = _event(context, content, "transcript-uuid-001")

        memory.capture(fallback)
        memory.capture(transcript)
        assert await memory.maintenance.run_once(force=True) == 2
        facts = store.get_all()
        assert len(facts) == 1
        evidence = store.list_fact_evidence(facts[0].id)
        assert len(evidence) == 1
        assert evidence[0].source_event_id == memory.capture(transcript)["event_id"]
        events = {event.metadata["source_ref"]: event for event in store.list_episodic_events()}
        assert events[fallback_ref].status == "deferred"
        assert events["transcript-uuid-001"].status == "consolidated"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_provisional_event_cannot_be_reclassified_into_extraction(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        content = "Do not extract this fallback event."
        source_ref = "prompt:" + hashlib.sha256(content.encode()).hexdigest()
        event = _event(context, content, source_ref, identity_status="provisional_unassociated")
        accepted = memory.capture(event)
        job_id = accepted["job_id"]
        with store._lock:
            store._conn.execute("UPDATE memory_jobs SET kind='extract' WHERE id=?", (job_id,))
            store._conn.commit()

        assert await memory.maintenance.run_once(force=True) == 0
        assert store.get_all() == []
        assert store.get_job(job_id).status == "pending"
        assert store.get_episodic_event(accepted["event_id"]).status == "deferred"
    finally:
        store.close()


def test_capture_only_commit_requires_its_own_event(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        first = _event(
            context,
            "First provisional event.",
            "prompt:first",
            identity_status="provisional_unassociated",
        )
        second = _event(
            context,
            "Second provisional event.",
            "prompt:second",
            identity_status="provisional_unassociated",
        )
        first_result = memory.capture(first)
        second_result = memory.capture(second)
        leased = store.claim_jobs(owner="guard", job_id=first_result["job_id"])[0]
        with pytest.raises(ValueError, match="cannot contain durable facts"):
            store.commit_fact_and_job(
                leased.id,
                owner="guard",
                lease_token=leased.lease_token,
                processed_event_ids=[first_result["event_id"]],
                result={"decision": "identity_pending", "fact_ids": ["fake"]},
            )
        with pytest.raises(ValueError, match="their capture event only"):
            store.commit_fact_and_job(
                leased.id,
                owner="guard",
                lease_token=leased.lease_token,
                processed_event_ids=[second_result["event_id"]],
                result={"decision": "identity_pending"},
            )
        assert store.fail_job(
            leased.id,
            owner="guard",
            lease_token=leased.lease_token,
            error="test cleanup",
        )
    finally:
        store.close()


def test_provisional_source_cannot_commit_fact_through_storage_backdoor(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        event = _event(
            context,
            "A provisional event must never become a fact.",
            "prompt:backdoor",
            identity_status="provisional_unassociated",
        )
        result = memory.capture(event)
        with store._lock:
            store._conn.execute(
                "UPDATE memory_jobs SET kind='extract' WHERE id=?", (result["job_id"],)
            )
            store._conn.commit()
        leased = store.claim_jobs(owner="guard", job_id=result["job_id"])[0]
        candidate = MemoryChunk(
            content=event.content,
            embedding=[1.0, 0.0, 0.0],
            metadata={
                "scope": "project",
                "scope_id": context.project_id,
                "user_profile_id": context.user_profile_id,
            },
        )
        with pytest.raises(ValueError, match="cannot commit durable facts"):
            store.commit_fact_and_job(
                leased.id,
                owner="guard",
                lease_token=leased.lease_token,
                chunks=[candidate],
                processed_event_ids=[result["event_id"]],
                result={"decision": "facts"},
            )
        assert store.get_all() == []
        assert store.fail_job(
            leased.id,
            owner="guard",
            lease_token=leased.lease_token,
            error="test cleanup",
        )
    finally:
        store.close()


def test_provisional_status_rejects_invalid_combinations(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        event = _event(
            context,
            "A provisional event",
            "prompt:invalid",
            identity_status="provisional_unassociated",
        )
        assert memory.capture(event, explicit=True)["status"] == "rejected"
        codex = context.model_copy(update={"host": "codex"})
        codex_event = event.model_copy(
            update={
                "context": codex,
                "event_id": scoped_event_id(codex, "prompt:invalid", "user", event.content),
            }
        )
        assert memory.capture(codex_event)["status"] == "rejected"
    finally:
        store.close()


def test_identity_status_is_strictly_typed():
    with pytest.raises(ValidationError):
        CaptureEvent(
            event_id="event",
            context=ScopeContext(
                user_profile_id="profile-a",
                project_id="project-a",
                workspace_id=None,
                session_id="session-a",
                host="claude-code",
            ),
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content="content",
            source_kind="claude_code_transcript",
            identity_status="unknown",
        )


def test_legacy_job_without_identity_status_replays_without_reclassification(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        content = "A legacy Claude event remains extraction eligible."
        source_ref = "native-prompt-legacy"
        current = _event(context, content, source_ref)
        legacy = current.model_copy(
            update={
                "identity_version": 1,
                "event_id": legacy_event_id(context.session_id, source_ref, "user", content),
            }
        )
        original = memory.capture(legacy)
        assert "identity_status" not in store.get_job(original["job_id"]).payload
        replay = memory.capture(current)
        assert replay["event_id"] == original["event_id"]
        assert replay["processing_status"] == "pending"
        assert replay["fact_eligible"] is True
    finally:
        store.close()


class _CaptureClient:
    def __init__(self):
        self.captures = []

    async def capture(self, payload):
        self.captures.append(payload)
        return {"status": "committed", "committed": True}

    async def retrieve(self, payload):
        return {"facts": []}


@pytest.mark.asyncio
async def test_adapter_marks_missing_occurrence_as_provisional(tmp_path):
    client = _CaptureClient()
    registry = ScopeRegistry(
        projects={
            "project-a": ProjectIdentity(
                id="project-a",
                root=canonical_project_root(tmp_path),
                user_profile_id="profile-a",
                enabled=True,
            )
        }
    )
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry,
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    result = await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "A repeated prompt with no native occurrence ID",
        },
    )
    assert result.status == "degraded"
    assert client.captures[0]["identity_status"] == "provisional_unassociated"
    await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "A prompt with an explicit native occurrence ID",
            "prompt_id": "native-occurrence-001",
        },
    )
    assert client.captures[1]["identity_status"] == "normal"


@pytest.mark.asyncio
async def test_deferred_retention_waits_for_job_and_does_not_recreate_raw_event(tmp_path):
    memory, context, store = _service(tmp_path)
    try:
        event = _event(
            context,
            "A provisional retention canary.",
            "prompt:retention",
            identity_status="provisional_unassociated",
        )
        memory.capture(event)
        cutoff = datetime.now(timezone.utc) + timedelta(days=1)
        assert store.prune_processed_evidence(before=cutoff) == 0
        assert await memory.maintenance.run_once(force=True) == 1
        assert store.prune_processed_evidence(before=cutoff) == 1
        replay = memory.capture(event)
        assert replay["processing_status"] == "deferred"
        assert replay["fact_eligible"] is False
        assert store.list_episodic_events() == []
        assert len(store.list_jobs()) == 1
        assert store.get_all() == []
    finally:
        store.close()


def test_capture_interrupt_rolls_back_event_and_job(tmp_path, monkeypatch):
    memory, context, store = _service(tmp_path)
    event = _event(
        context,
        "An interrupted capture.",
        "prompt:interrupt",
        identity_status="provisional_unassociated",
    )
    original = store._job_to_row

    def interrupted(job):
        raise KeyboardInterrupt("injected after event insert")

    try:
        monkeypatch.setattr(store, "_job_to_row", interrupted)
        with pytest.raises(KeyboardInterrupt):
            memory.capture(event)
        assert not store._conn.in_transaction
        assert store.list_episodic_events() == []
        assert store.list_jobs() == []
        monkeypatch.setattr(store, "_job_to_row", original)
        assert memory.capture(event)["status"] == "committed"
    finally:
        store.close()


def test_capture_only_commit_interrupt_rolls_back_and_preserves_lease(tmp_path, monkeypatch):
    memory, context, store = _service(tmp_path)
    event = _event(
        context,
        "An interrupted completion.",
        "prompt:commit-interrupt",
        identity_status="provisional_unassociated",
    )
    original = store._conn

    class InterruptedConnection:
        def __getattr__(self, name):
            return getattr(original, name)

        def execute(self, sql, parameters=()):
            if sql.startswith("UPDATE memory_jobs SET status='completed'"):
                raise KeyboardInterrupt("injected after event update")
            return original.execute(sql, parameters)

    try:
        captured = memory.capture(event)
        leased = store.claim_jobs(owner="interrupt", job_id=captured["job_id"])[0]
        monkeypatch.setattr(store, "_conn", InterruptedConnection())
        with pytest.raises(KeyboardInterrupt):
            store.commit_fact_and_job(
                leased.id,
                owner="interrupt",
                lease_token=leased.lease_token,
                processed_event_ids=[captured["event_id"]],
                result={"decision": "identity_pending"},
            )
        assert not original.in_transaction
        assert store.get_job(leased.id).status == "leased"
        assert store.get_episodic_event(captured["event_id"]).status == "deferred"
        assert store.get_all() == []
        monkeypatch.setattr(store, "_conn", original)
        assert store.commit_fact_and_job(
            leased.id,
            owner="interrupt",
            lease_token=leased.lease_token,
            processed_event_ids=[captured["event_id"]],
            result={"decision": "identity_pending"},
        )
    finally:
        monkeypatch.setattr(store, "_conn", original)
        store.close()
