"""Durable source-review references remain atomic, scoped, and forgettable."""

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import pytest

from mnemos.jobs import DurableJob, FactEvidence
from mnemos.source_review import SourceReviewReference
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import SQLiteStore


def source(event_id: str = "source-1", *, scope_id: str = "project-1") -> EpisodicEvent:
    content = "The marker changed, but no replacement is accepted."
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role="user",
        content=content,
        normalized_content=content.casefold(),
        scope="project",
        scope_id=scope_id,
        source="manual",
        metadata={"user_profile_id": "profile-1", "project_id": scope_id},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def target(*, fact_id: str = "fact-1", version: int = 1) -> MemoryChunk:
    return MemoryChunk(
        id=fact_id,
        content="The marker is cedar.",
        version=version,
        metadata={
            "user_profile_id": "profile-1",
            "scope": "project",
            "scope_id": "project-1",
        },
    )


def review(
    event: EpisodicEvent,
    *,
    candidate_index: int = 0,
    target_id: str | None = None,
    version: int | None = None,
) -> SourceReviewReference:
    return SourceReviewReference(
        candidate_index=candidate_index,
        source_id=event.id,
        source_fingerprint=event.event_fingerprint,
        source_content_sha256=hashlib.sha256(event.content.encode()).hexdigest(),
        source_role=event.role,
        source_channel=event.source,
        occurred_at=event.occurred_at.isoformat(),
        user_profile_id="profile-1",
        scope="project",
        scope_id=event.scope_id,
        start=0,
        end=len(event.content),
        source_kind="correction_or_rejection",
        operation="conflict" if target_id else "unresolved",
        target_fact_id=target_id,
        expected_version=version,
    )


def prepare(
    store: SQLiteStore,
    event: EpisodicEvent,
    *,
    payload: dict[str, object] | None = None,
    lease_seconds: int = 300,
) -> tuple[DurableJob, DurableJob]:
    store.append_episodic_event(event)
    job = store.enqueue_job(
        DurableJob(
            kind="extract",
            payload={"event_id": event.id, **(payload or {})},
        )
    )
    return job, store.claim_jobs(owner="review-worker", lease_seconds=lease_seconds)[0]


def commit_review(
    store: SQLiteStore,
    claimed: DurableJob,
    *,
    references: list[SourceReviewReference],
    chunks: list[MemoryChunk] | None = None,
    evidence: list[FactEvidence] | None = None,
    processed_event_ids: list[str] | None = None,
    result: dict[str, object] | None = None,
    now: datetime | None = None,
) -> bool:
    return store.commit_fact_and_job(
        claimed.id,
        owner="review-worker",
        lease_token=claimed.lease_token,
        chunks=chunks or [],
        evidence=evidence or [],
        processed_event_ids=processed_event_ids or ["source-1"],
        review_references=references,
        result=result or {"decision": "review_pending"},
        now=now,
    )


def test_review_survives_restart_without_copying_source_text(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    event = source()
    _, claimed = prepare(store, event)
    assert commit_review(store, claimed, references=[review(event)])
    raw = store._conn.execute("SELECT reference FROM source_review_records").fetchone()[0]
    assert event.content not in raw
    store.close()

    reopened = SQLiteStore(db_path=str(db_path))
    try:
        rows = reopened.list_source_reviews(
            user_profile_id="profile-1", scope="project", scope_id="project-1"
        )
        assert rows[0]["state"] == "open"
        assert rows[0]["reference"]["source_id"] == event.id
    finally:
        reopened.close()


def test_mixed_fact_and_review_commit_is_atomic(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "mixed.sqlite"))
    try:
        event = source()
        chunk = target(fact_id="new-fact")
        job, claimed = prepare(store, event)
        assert commit_review(
            store,
            claimed,
            references=[review(event)],
            chunks=[chunk],
            evidence=[
                FactEvidence(fact_id=chunk.id, source_event_id=event.id, excerpt=event.content)
            ],
            result={"decision": "facts"},
        )
        assert store.get(chunk.id) is not None
        assert store.list_source_reviews(
            user_profile_id="profile-1", scope="project", scope_id="project-1"
        )
        completed = store.get_job(job.id)
        assert completed is not None and completed.status == "completed"
    finally:
        store.close()


def test_expired_lease_does_not_persist_reviews(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "expired.sqlite"))
    try:
        event = source()
        _, claimed = prepare(store, event, lease_seconds=1)
        assert claimed.lease_until is not None
        committed = commit_review(
            store,
            claimed,
            references=[review(event)],
            processed_event_ids=[event.id],
            now=claimed.lease_until + timedelta(seconds=1),
        )
        assert committed is False
        assert (
            store.list_source_reviews(
                user_profile_id="profile-1", scope="project", scope_id="project-1"
            )
            == []
        )
    finally:
        store.close()


def test_stale_target_rolls_back_job_and_review(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "stale.sqlite"))
    try:
        event = source()
        fact = target(version=2)
        store.store(fact)
        pending_chunk = target(fact_id="pending-fact")
        # The target exists only in the review reference, exercising target
        # dependency discovery independent of the processing job payload.
        _, claimed = prepare(store, event)
        with pytest.raises(ValueError):
            commit_review(
                store,
                claimed,
                references=[review(event, target_id=fact.id, version=1)],
                chunks=[pending_chunk],
                evidence=[
                    FactEvidence(
                        fact_id=pending_chunk.id,
                        source_event_id=event.id,
                        excerpt=event.content,
                    )
                ],
            )
        assert (
            store.list_source_reviews(
                user_profile_id="profile-1", scope="project", scope_id="project-1"
            )
            == []
        )
        assert store.get(pending_chunk.id) is None
        current = store.get_job(claimed.id)
        assert current is not None and current.status == "leased"
    finally:
        store.close()


def test_source_scope_and_mutation_are_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "source-guard.sqlite"))
    try:
        event = source()
        _, claimed = prepare(store, event)
        foreign = review(event).model_copy(update={"scope_id": "other-project"})
        with pytest.raises(ValueError, match="scope|partition"):
            commit_review(store, claimed, references=[foreign])
        changed_content = "Changed source."
        changed_reference = review(event).model_copy(
            update={
                "source_content_sha256": hashlib.sha256(changed_content.encode()).hexdigest(),
                "end": len(changed_content),
            }
        )
        with pytest.raises(ValueError, match="source|changed"):
            commit_review(store, claimed, references=[changed_reference])
    finally:
        store.close()


def test_duplicate_candidate_indices_are_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "duplicates.sqlite"))
    try:
        event = source()
        _, claimed = prepare(store, event)
        with pytest.raises(ValueError, match="unique"):
            commit_review(store, claimed, references=[review(event), review(event)])
    finally:
        store.close()


def test_open_review_pins_source_against_processed_pruning(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "prune.sqlite"))
    try:
        event = source()
        _, claimed = prepare(store, event)
        assert commit_review(store, claimed, references=[review(event)])
        assert (
            store.prune_processed_evidence(
                before=datetime.now(timezone.utc) + timedelta(days=1), limit=100
            )
            == 0
        )
        assert store.get_episodic_event(event.id) is not None
    finally:
        store.close()


def test_target_forget_invalidates_targeted_review(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "forget-target.sqlite"))
    try:
        event = source()
        fact = target()
        store.store(fact)
        # The target exists only in the review reference, exercising target
        # dependency discovery independent of the processing job payload.
        _, claimed = prepare(store, event)
        assert commit_review(
            store,
            claimed,
            references=[review(event, target_id=fact.id, version=fact.version)],
        )
        assert store.forget_fact(fact.id, expected_version=fact.version)
        rows = store.list_source_reviews(
            user_profile_id="profile-1", scope="project", scope_id="project-1"
        )
        assert rows[0]["state"] == "invalidated_target"
        assert store.get_episodic_event(event.id) is not None
        assert (
            store.list_source_reviews(
                user_profile_id="other-profile", scope="project", scope_id="project-1"
            )
            == []
        )
        with pytest.raises(ValueError, match="scope identity"):
            store.list_source_reviews(user_profile_id="profile-1", scope="project", scope_id=None)
    finally:
        store.close()


def test_shared_source_is_preserved_but_source_suppression_invalidates_review(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "shared-source.sqlite"))
    try:
        event = source()
        first = target(fact_id="fact-1")
        second = target(fact_id="fact-2")
        store.store(first)
        store.store(second)
        _, claimed = prepare(store, event, payload={"target_fact_id": first.id})
        assert commit_review(
            store,
            claimed,
            references=[review(event, target_id=first.id, version=first.version)],
        )
        # Add a second fact dependency on the same source, preserving it when
        # the first fact is forgotten through the reviewed cleanup path.
        store._conn.execute(
            "INSERT INTO fact_evidence (evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (
                "evidence-second",
                second.id,
                event.id,
                event.content,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store._conn.commit()
        with store._lock:
            store._conn.execute("BEGIN IMMEDIATE")
            try:
                assert store._forget_fact_locked(
                    first.id, expected_version=first.version, preserve_shared_sources=True
                )
                store._conn.commit()
            except BaseException:
                store._conn.rollback()
                raise
        assert store.get_episodic_event(event.id) is not None
        rows = store.list_source_reviews(
            user_profile_id="profile-1", scope="project", scope_id="project-1"
        )
        assert rows[0]["state"] == "invalidated_source"
    finally:
        store.close()
