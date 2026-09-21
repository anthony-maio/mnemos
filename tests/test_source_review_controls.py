from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from mnemos.contracts import CaptureEvent
from mnemos.scopes import ScopeContext
from mnemos.jobs import DurableJob
from mnemos.memory_service import LocalMemoryService
from mnemos.utils.storage import SQLiteStore

from tests.test_source_bound_service import PipelineProvider, service


async def create_review(
    tmp_path: Path,
) -> tuple[LocalMemoryService, ScopeContext, SQLiteStore, DurableJob, dict[str, Any]]:
    memory, context, store = service(tmp_path)
    memory.engine.config.sleep.extraction_mode = "source_bound"
    memory.engine.config.modules.surprisal = False
    memory.engine.config.modules.affective = False
    memory.engine._llm = PipelineProvider("review")
    content = "The marker changed, but no replacement was approved."
    captured = memory.capture(
        CaptureEvent(
            event_id="review-event",
            context=context,
            role="user",
            occurred_at=datetime.now(timezone.utc),
            content=content,
            source_kind="user_prompt",
        )
    )
    job = store.claim_jobs(owner="worker")[0]
    await memory.process_job(job)
    return memory, context, store, job, captured


def review_payload(context: ScopeContext, **extra: object) -> dict[str, object]:
    return {"context": context.model_dump(), **extra}


@pytest.mark.asyncio
async def test_reviews_inspect_discard_and_restart(tmp_path: Path) -> None:
    memory, context, store, job, captured = await create_review(tmp_path)
    try:
        listed = await memory.dispatch("reviews", review_payload(context))
        assert listed["schema_version"] == 1
        assert len(listed["reviews"]) == 1
        row = listed["reviews"][0]
        assert row["job_id"] == job.id
        assert row["candidate_index"] == 0
        assert row["state"] == "open"
        assert "content" not in row["reference"]

        inspected = await memory.dispatch(
            "inspect_review",
            review_payload(context, job_id=job.id, candidate_index=0),
        )
        assert inspected["schema_version"] == 1
        assert inspected["review"]["state"] == "open"
        assert inspected["review"]["excerpt"] == (
            "The marker changed, but no replacement was approved."
        )

        discarded = await memory.dispatch(
            "discard_review",
            review_payload(context, job_id=job.id, candidate_index=0),
        )
        assert discarded == {"schema_version": 1, "status": "discarded"}
        repeated = await memory.dispatch(
            "discard_review",
            review_payload(context, job_id=job.id, candidate_index=0),
        )
        assert repeated == {"schema_version": 1, "status": "unchanged"}
        after = await memory.dispatch(
            "inspect_review",
            review_payload(context, job_id=job.id, candidate_index=0),
        )
        assert after["review"]["state"] == "discarded"
        assert after["review"]["excerpt"] is None
        assert store.get_all() == []
    finally:
        store.close()

    from mnemos.utils.storage import SQLiteStore

    reopened_store = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        listed_after_restart = reopened_store.list_source_reviews(
            user_profile_id=context.user_profile_id,
            scope="project",
            scope_id=context.project_id,
        )
        assert listed_after_restart[0]["state"] == "discarded"
    finally:
        reopened_store.close()


@pytest.mark.asyncio
async def test_open_review_pins_source_until_discard_then_prune_releases_it(
    tmp_path: Path,
) -> None:
    memory, context, store, job, _ = await create_review(tmp_path)
    try:
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        store._conn.execute(
            "UPDATE episodic_events SET created_at=? WHERE id=?",
            (old, job.payload["event_id"]),
        )
        store._conn.commit()
        assert (
            store.prune_processed_evidence(before=datetime.now(timezone.utc) + timedelta(days=1))
            == 0
        )
        await memory.dispatch(
            "discard_review", review_payload(context, job_id=job.id, candidate_index=0)
        )
        assert (
            store.prune_processed_evidence(before=datetime.now(timezone.utc) + timedelta(days=1))
            == 1
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stale_source_is_concealed_and_invalidated(tmp_path: Path) -> None:
    memory, context, store, job, _ = await create_review(tmp_path)
    try:
        store._conn.execute(
            "UPDATE episodic_events SET content=? WHERE id=?",
            ("Changed after review.", job.payload["event_id"]),
        )
        store._conn.commit()
        before = list(store._conn.iterdump())
        changes = store._conn.total_changes
        listed = await memory.dispatch("reviews", review_payload(context))
        assert listed["reviews"][0]["state"] == "invalidated_source"
        inspected = await memory.dispatch(
            "inspect_review", review_payload(context, job_id=job.id, candidate_index=0)
        )
        assert store._conn.total_changes == changes
        assert list(store._conn.iterdump()) == before
        assert inspected["review"]["state"] == "invalidated_source"
        assert inspected["review"]["excerpt"] is None
        assert "Changed after review." not in json.dumps(inspected)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_review_controls_reject_foreign_scope_and_malformed_requests(
    tmp_path: Path,
) -> None:
    memory, context, store, job, _ = await create_review(tmp_path)
    try:
        foreign_path = tmp_path / "foreign"
        foreign_path.mkdir()
        memory.registry.register_project(foreign_path, user_profile_id="foreign", enabled=True)
        foreign = memory.registry.resolve(
            foreign_path, user_profile_id="foreign", session_id="s", host="codex"
        )
        with pytest.raises(ValueError):
            await memory.dispatch(
                "inspect_review",
                review_payload(foreign, job_id=job.id, candidate_index=0),
            )
        with pytest.raises(ValueError):
            await memory.dispatch(
                "inspect_review", review_payload(context, job_id=job.id, candidate_index=-1)
            )
        with pytest.raises(ValueError):
            await memory.dispatch(
                "inspect_review",
                review_payload(
                    context,
                    job_id=job.id,
                    candidate_index=True,
                ),
            )
        with pytest.raises(ValueError):
            await memory.dispatch("reviews", review_payload(context, scope_id=context.project_id))
        with pytest.raises(ValueError):
            await memory.dispatch(
                "inspect_review", review_payload(context, job_id="missing", candidate_index=0)
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_disabled_project_cannot_inspect_reviews(tmp_path: Path) -> None:
    memory, context, store, _, _ = await create_review(tmp_path)
    try:
        assert context.project_id is not None
        project = memory.registry.projects[context.project_id]
        project.enabled = False
        with pytest.raises(ValueError):
            await memory.dispatch("reviews", review_payload(context))
    finally:
        store.close()
