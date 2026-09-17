from datetime import datetime, timezone

import pytest

from mnemos.jobs import DurableJob, FactEvidence
from mnemos.maintenance import MaintenanceScheduler
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import (
    ConflictCanonicalCollisionError,
    InMemoryStore,
    SQLiteStore,
)


def _metadata(status: str) -> dict[str, str]:
    return {
        "scope": "project",
        "scope_id": "project-a",
        "user_profile_id": "profile-a",
        "assertion_status": status,
    }


def _chunk(content: str, *, status: str, chunk_id: str) -> MemoryChunk:
    return MemoryChunk(
        id=chunk_id,
        content=content,
        embedding=[1.0, 0.0],
        metadata=_metadata(status),
    )


def _event(event_id: str = "source-1") -> EpisodicEvent:
    now = datetime.now(timezone.utc)
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=event_id,
        role="user",
        content="The source says the assertion is unresolved.",
        normalized_content="the source says the assertion is unresolved.",
        scope="project",
        scope_id="project-a",
        source="test",
        metadata={"user_profile_id": "profile-a"},
        created_at=now,
        occurred_at=now,
    )


def _job(event_id: str = "source-1", job_id: str = "job-1") -> DurableJob:
    return DurableJob(
        id=job_id,
        kind="extract",
        max_attempts=3,
        payload={"event_id": event_id},
    )


def test_inmemory_rejects_conflict_collision_without_mutating_asserted_row():
    store = InMemoryStore()
    existing = _chunk("Use SQLite", status="asserted", chunk_id="asserted")
    store.upsert_canonical(existing)

    with pytest.raises(ConflictCanonicalCollisionError) as error:
        store.upsert_canonical(_chunk("  Use   SQLite  ", status="conflict", chunk_id="conflict"))
    assert "SQLite" not in str(error.value)

    retained = store.get(existing.id)
    assert retained is not None
    assert retained.metadata["assertion_status"] == "asserted"
    assert retained.version == 1
    assert retained.metadata.get("merged_duplicate_ids") is None
    assert store.get("conflict") is None


def test_inmemory_allows_new_and_repeat_conflict_rows():
    store = InMemoryStore()
    first = store.upsert_canonical(_chunk("Use Redis", status="conflict", chunk_id="first"))
    repeated = store.upsert_canonical(_chunk("Use Redis", status="conflict", chunk_id="repeat"))

    assert first.id == repeated.id == "first"
    assert repeated.metadata["assertion_status"] == "conflict"
    assert repeated.metadata["merged_duplicate_ids"] == ["repeat"]

    new_conflict = store.upsert_canonical(
        _chunk("Use PostgreSQL", status="conflict", chunk_id="new")
    )
    assert new_conflict.id == "new"
    assert len(store.get_all()) == 2


@pytest.mark.asyncio
async def test_sqlite_collision_rolls_back_batch_and_marks_retryable_job(tmp_path):
    path = tmp_path / "collision.sqlite"
    store = SQLiteStore(str(path))
    existing = _chunk("Use SQLite", status="asserted", chunk_id="asserted")
    store.upsert_canonical(existing)
    source = _event()
    store.append_episodic_event(source)
    store.enqueue_job(_job())

    async def process(job: DurableJob) -> None:
        valid = _chunk("Use PostgreSQL", status="asserted", chunk_id="valid")
        conflict = _chunk("Use SQLite", status="conflict", chunk_id="conflict")
        evidence = [
            FactEvidence(fact_id=valid.id, source_event_id=source.id, excerpt="valid"),
            FactEvidence(fact_id=conflict.id, source_event_id=source.id, excerpt="conflict"),
        ]
        store.commit_fact_and_job(
            job.id,
            owner=job.lease_owner or "",
            lease_token=job.lease_token,
            chunks=[valid, conflict],
            evidence=evidence,
            processed_event_ids=[source.id],
            result={"decision": "facts"},
        )

    scheduler = MaintenanceScheduler(store, process)
    assert await scheduler.run_once(force=True) == 0

    failed_job = store.get_job("job-1")
    retained_source = store.get_episodic_event(source.id)
    retained_fact = store.get(existing.id)
    assert failed_job is not None
    assert failed_job.status == "pending"
    assert failed_job.last_error == "conflict_canonical_collision"
    assert retained_source is not None and retained_source.status == "pending"
    assert retained_fact is not None
    assert retained_fact.metadata["assertion_status"] == "asserted"
    assert store.get("valid") is None
    assert store.get("conflict") is None
    assert store.list_fact_evidence(existing.id) == []
    store.close()

    reopened = SQLiteStore(str(path))
    try:
        assert reopened.get(existing.id).metadata["assertion_status"] == "asserted"
        assert reopened.get_episodic_event(source.id).status == "pending"
        assert reopened.get_job("job-1").status == "pending"
    finally:
        reopened.close()


def test_sqlite_missing_assertion_status_rejects_conflict_after_reopen(tmp_path):
    path = tmp_path / "legacy-status.sqlite"
    store = SQLiteStore(str(path))
    existing = _chunk("Use SQLite", status="asserted", chunk_id="asserted")
    existing.metadata.pop("assertion_status")
    store.upsert_canonical(existing)
    store.close()

    reopened = SQLiteStore(str(path))
    try:
        before = reopened.get("asserted")
        assert before is not None
        assert "assertion_status" not in before.metadata
        with pytest.raises(ConflictCanonicalCollisionError):
            reopened.upsert_canonical(
                _chunk(" Use  SQLite ", status="conflict", chunk_id="conflict")
            )
        after = reopened.get("asserted")
        assert after is not None
        assert after.model_dump() == before.model_dump()
        assert reopened.get("conflict") is None
    finally:
        reopened.close()


def test_sqlite_collision_after_reopen_preserves_existing_evidence(tmp_path):
    path = tmp_path / "evidence-preserved.sqlite"
    store = SQLiteStore(str(path))
    existing = _chunk("Use SQLite", status="asserted", chunk_id="asserted")
    store.upsert_canonical(existing)
    source = _event("source-evidence")
    store.append_episodic_event(source)
    store.enqueue_job(_job(source.id, "job-evidence"))
    claimed = store.claim_jobs(owner="worker", job_id="job-evidence")[0]
    store.commit_fact_and_job(
        claimed.id,
        owner="worker",
        lease_token=claimed.lease_token,
        chunks=[_chunk("Use SQLite", status="asserted", chunk_id="proposal")],
        evidence=[FactEvidence(fact_id="proposal", source_event_id=source.id, excerpt="original")],
        processed_event_ids=[source.id],
        result={"decision": "facts"},
    )
    original_evidence = store.list_fact_evidence(existing.id)
    assert len(original_evidence) == 1
    store.close()

    reopened = SQLiteStore(str(path))
    try:
        before = reopened.get(existing.id)
        assert before is not None
        collision_source = _event("source-collision")
        reopened.append_episodic_event(collision_source)
        reopened.enqueue_job(_job(collision_source.id, "job-collision"))
        collision_job = reopened.claim_jobs(owner="worker", job_id="job-collision")[0]
        with pytest.raises(ConflictCanonicalCollisionError):
            reopened.commit_fact_and_job(
                collision_job.id,
                owner="worker",
                lease_token=collision_job.lease_token,
                chunks=[_chunk(" Use   SQLite ", status="conflict", chunk_id="conflict")],
                evidence=[
                    FactEvidence(
                        fact_id="conflict",
                        source_event_id=collision_source.id,
                        excerpt="must not attach",
                    )
                ],
                processed_event_ids=[collision_source.id],
                result={"decision": "facts"},
            )
        after = reopened.get(existing.id)
        assert after is not None
        assert after.model_dump() == before.model_dump()
        assert reopened.list_fact_evidence(existing.id) == original_evidence
        assert reopened.get("conflict") is None
        assert reopened.get_episodic_event(collision_source.id).status == "pending"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_sqlite_allows_targetless_and_repeated_conflicts(tmp_path):
    store = SQLiteStore(str(tmp_path / "valid-conflicts.sqlite"))
    try:
        first = store.upsert_canonical(_chunk("Use Redis", status="conflict", chunk_id="first"))
        new_conflict = store.upsert_canonical(
            _chunk("Use PostgreSQL", status="conflict", chunk_id="new")
        )
        assert new_conflict.id == "new"

        source = _event("source-repeat")
        store.append_episodic_event(source)
        store.enqueue_job(_job(source.id, "job-repeat"))
        claimed = store.claim_jobs(owner="worker", job_id="job-repeat")[0]
        repeated = _chunk("Use Redis", status="conflict", chunk_id="repeat")
        store.commit_fact_and_job(
            claimed.id,
            owner="worker",
            lease_token=claimed.lease_token,
            chunks=[repeated],
            evidence=[
                FactEvidence(fact_id=repeated.id, source_event_id=source.id, excerpt="same"),
                FactEvidence(fact_id=repeated.id, source_event_id=source.id, excerpt="duplicate"),
            ],
            processed_event_ids=[source.id],
            result={"decision": "facts"},
        )
        assert first.id == "first"
        assert store.get(first.id).metadata["assertion_status"] == "conflict"
        assert len(store.list_fact_evidence(first.id)) == 1
        assert store.get_episodic_event(source.id).status == "consolidated"
    finally:
        store.close()
