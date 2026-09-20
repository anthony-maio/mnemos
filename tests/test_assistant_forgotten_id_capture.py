from datetime import datetime, timezone

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore
from tests.test_memory_service import service

FORGOTTEN_ID = "89c88200-5aeb-4061-956c-3144bc0f0544"


def assistant_event(context, content, event_id="assistant-after-forget"):
    return CaptureEvent(
        event_id=event_id,
        context=context,
        occurred_at=datetime.now(timezone.utc),
        role="assistant",
        content=content,
        source_kind="hermes_memory_provider",
        source_ref="transcript:v2:40",
    )


@pytest.mark.asyncio
async def test_assistant_quote_of_forgotten_id_is_suppressed(tmp_path):
    memory, context = service(tmp_path)
    try:
        store = memory.engine.store
        store.record_suppression(fact_id=FORGOTTEN_ID, reason="explicit_forget")
        captured = memory.capture(assistant_event(context, f"Forgotten fact ID: {FORGOTTEN_ID}."))
        assert captured["processing_status"] == "suppressed"
        assert store.list_episodic_events() == []
    finally:
        memory.engine.store.close()


def test_exact_id_matching_does_not_match_substring(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStore(str(path))
    try:
        store.record_suppression(fact_id=FORGOTTEN_ID, reason="explicit_forget")
        assert not store._contains_forgotten_id("prefix" + FORGOTTEN_ID + "suffix")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_user_capture_and_explicit_reteach_remain_allowed(tmp_path):
    memory, context = service(tmp_path)
    try:
        store = memory.engine.store
        store.record_suppression(fact_id=FORGOTTEN_ID, reason="explicit_forget")
        user = CaptureEvent(
            event_id="user-quoted-forgotten-id",
            context=context,
            occurred_at=datetime.now(timezone.utc),
            role="user",
            content=f"Please discuss {FORGOTTEN_ID}.",
            source_kind="hermes_memory_provider",
        )
        assert memory.capture(user)["status"] == "committed"
        explicit = await memory.dispatch(
            "remember",
            {
                "context": context.model_dump(),
                "content": f"The new explicit reference is {FORGOTTEN_ID}.",
            },
        )
        assert explicit["status"] == "committed"
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_real_forget_and_unrelated_assistant_control(tmp_path):
    memory, context = service(tmp_path)
    try:
        taught = await memory.dispatch(
            "remember",
            {"context": context.model_dump(), "content": "The release label is cedar-otter."},
        )
        assert taught["status"] == "committed"
        await memory.maintenance.run_once(force=True)
        target = memory.engine.store.get_all()[0]
        assert memory.engine.store.forget_fact(target.id, expected_version=target.version)
        blocked = memory.capture(
            assistant_event(
                context, f"The erased fact was {target.id}.", "blocked-after-real-forget"
            )
        )
        unrelated = memory.capture(
            assistant_event(context, "This unrelated assistant control remains valid.", "unrelated")
        )
        assert blocked["processing_status"] == "suppressed"
        assert unrelated["status"] == "committed"
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_capture_before_forget_commit_after_forget_is_suppressed(tmp_path):
    memory, context = service(tmp_path)
    try:
        store = memory.engine.store
        event = assistant_event(context, f"Assistant quoted {FORGOTTEN_ID}.", "racing-event")
        assert memory.capture(event)["status"] == "committed"
        job = store.claim_jobs(owner="race-owner", limit=1)[0]
        store.record_suppression(fact_id=FORGOTTEN_ID, reason="explicit_forget")
        candidate = MemoryChunk(
            content="A generated candidate from the racing assistant event.",
            metadata={
                "scope": "project",
                "scope_id": context.project_id,
                "user_profile_id": context.user_profile_id,
            },
        )
        assert store.commit_fact_and_job(
            job.id,
            owner="race-owner",
            lease_token=job.lease_token,
            chunks=[candidate],
            evidence=[],
            processed_event_ids=[job.payload["event_id"]],
            result={"decision": "facts"},
        )
        assert store.get_all() == []
    finally:
        memory.engine.store.close()


def test_tombstone_matching_survives_restart(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = SQLiteStore(str(path))
    first.record_suppression(fact_id=FORGOTTEN_ID, reason="explicit_forget")
    first.close()
    second = SQLiteStore(str(path))
    try:
        assert second._contains_forgotten_id(f"quoted: {FORGOTTEN_ID}")
        assert not second._contains_forgotten_id("unrelated project control")
    finally:
        second.close()
