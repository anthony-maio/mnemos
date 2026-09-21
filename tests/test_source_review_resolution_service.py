"""Explicit clarification must atomically resolve a review through normal jobs."""

from pathlib import Path
from typing import Any

import pytest

from tests.test_source_review_controls import create_review, review_payload


@pytest.mark.asyncio
async def test_explicit_resolution_add_uses_new_source_and_survives_restart(tmp_path: Path) -> None:
    memory, context, store, review_job, _ = await create_review(tmp_path)
    try:
        response = await memory.dispatch(
            "resolve_review",
            review_payload(
                context,
                job_id=review_job.id,
                candidate_index=0,
                event_id="clarification-1",
                operation="add",
                content="The approved project marker is spruce-window-92.",
            ),
        )
        assert response["processing_status"] == "pending"
        assert store.get_all() == []
        job = store.claim_jobs(owner="resolver")[0]
        await memory.process_job(job)
        facts = store.get_all()
        assert len(facts) == 1
        assert facts[0].content == "The approved project marker is spruce-window-92."
        review = (
            await memory.dispatch(
                "inspect_review",
                review_payload(
                    context,
                    job_id=review_job.id,
                    candidate_index=0,
                ),
            )
        )["review"]
        assert review["state"] == "resolved"
        assert review["excerpt"] is None
        assert review["resolved_fact_id"] == facts[0].id
        assert review["resolved_version"] == facts[0].version
        assert review["resolution_event_id"] == response["event_id"]
        rows = store._conn.execute(
            "SELECT source_event_id, excerpt FROM fact_evidence WHERE fact_id=?", (facts[0].id,)
        ).fetchall()
        assert rows == [(response["event_id"], facts[0].content)]
    finally:
        store.close()
    from mnemos.utils.storage import SQLiteStore

    reopened = SQLiteStore(str(tmp_path / "memory.sqlite"))
    try:
        review = reopened.inspect_source_review(
            review_job.id,
            0,
            user_profile_id=context.user_profile_id,
            scope="project",
            scope_id=context.project_id,
        )
        assert review["state"] == "resolved"
        assert review["resolved_fact_id"] == facts[0].id
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_discard_between_resolution_enqueue_and_commit_rolls_back(tmp_path: Path) -> None:
    memory, context, store, review_job, _ = await create_review(tmp_path)
    try:
        await memory.dispatch(
            "resolve_review",
            review_payload(
                context,
                job_id=review_job.id,
                candidate_index=0,
                event_id="clarification-race",
                operation="add",
                content="The approved project marker is spruce-window-92.",
            ),
        )
        job = store.claim_jobs(owner="resolver")[0]
        await memory.dispatch(
            "discard_review",
            review_payload(
                context,
                job_id=review_job.id,
                candidate_index=0,
            ),
        )
        with pytest.raises(ValueError):
            await memory.process_job(job)
        assert store.get_all() == []
        pending = store.get_job(job.id)
        assert pending is not None and pending.status != "completed"
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"content": "   "},
        {"role": "assistant"},
        {"scope_id": "foreign"},
        {"candidate_index": True},
        {"operation": "correct"},
        {"target_fact_id": "foreign"},
    ],
)
async def test_resolution_rejects_malformed_requests_without_capture(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    memory, context, store, review_job, _ = await create_review(tmp_path)
    try:
        payload = review_payload(
            context,
            job_id=review_job.id,
            candidate_index=0,
            event_id="clarification-bad",
            operation="add",
            content="A new explicit fact.",
        )
        payload.update(changes)
        before = list(store._conn.iterdump())
        with pytest.raises(ValueError):
            await memory.dispatch("resolve_review", payload)
        assert list(store._conn.iterdump()) == before
    finally:
        store.close()


@pytest.mark.asyncio
async def test_unresolved_review_can_explicitly_correct_with_automatic_mutation_disabled(
    tmp_path: Path,
) -> None:
    memory, context, store, review_job, _ = await create_review(tmp_path)
    try:
        memory.engine.config.modules.mutable_rag = False
        await memory.dispatch(
            "remember",
            review_payload(
                context,
                event_id="old-marker",
                content="The approved project marker is cedar-window-92.",
            ),
        )
        await memory.process_job(store.claim_jobs(owner="seed")[0])
        old = store.get_all()[0]
        response = await memory.dispatch(
            "resolve_review",
            review_payload(
                context,
                job_id=review_job.id,
                candidate_index=0,
                event_id="correct-marker",
                operation="correct",
                target_fact_id=old.id,
                expected_version=old.version,
                content="The approved project marker is spruce-window-92.",
            ),
        )
        await memory.process_job(store.claim_jobs(owner="resolver")[0])
        facts = store.get_all()
        assert len(facts) == 1 and facts[0].id == old.id
        assert facts[0].version == old.version + 1
        assert facts[0].content == "The approved project marker is spruce-window-92."
        review = (
            await memory.dispatch(
                "inspect_review", review_payload(context, job_id=review_job.id, candidate_index=0)
            )
        )["review"]
        assert review["state"] == "resolved"
        assert review["resolved_fact_id"] == old.id
        assert review["resolved_version"] == facts[0].version
        assert review["resolution_event_id"] == response["event_id"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_foreign_project_cannot_resolve_review(tmp_path: Path) -> None:
    memory, context, store, review_job, _ = await create_review(tmp_path)
    try:
        other = tmp_path / "other"
        other.mkdir()
        memory.registry.register_project(
            other, user_profile_id=context.user_profile_id, enabled=True
        )
        foreign = memory.registry.resolve(
            other, user_profile_id=context.user_profile_id, session_id="other", host="codex"
        )
        before = list(store._conn.iterdump())
        with pytest.raises(ValueError):
            await memory.dispatch(
                "resolve_review",
                review_payload(
                    foreign,
                    job_id=review_job.id,
                    candidate_index=0,
                    event_id="foreign-resolution",
                    operation="add",
                    content="The approved project marker is spruce-window-92.",
                ),
            )
        assert list(store._conn.iterdump()) == before
    finally:
        store.close()
