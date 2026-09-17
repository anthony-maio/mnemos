"""Durable job and transactional evidence regressions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemos.jobs import DurableJob, FactEvidence, FactRevision
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore


def _chunk(content: str = "Use Python", chunk_id: str = "fact-1") -> MemoryChunk:
    embedder = SimpleEmbeddingProvider(dim=16)
    return MemoryChunk(
        id=chunk_id,
        content=content,
        embedding=embedder.embed(content),
        metadata={"scope": "project", "scope_id": "repo", "user_profile_id": "profile-a"},
    )


def test_job_lease_expiry_is_recoverable_and_exhaustion_is_visible(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "jobs.sqlite"))
    try:
        job = store.enqueue_job(
            DurableJob(kind="extract", payload={"event_id": "event-1"}, max_attempts=3)
        )
        now = datetime.now(timezone.utc)
        claimed = store.claim_jobs(owner="worker-a", limit=1, now=now)
        assert claimed[0].id == job.id
        assert claimed[0].status == "leased"
        assert claimed[0].attempt_count == 1

        recovered = store.claim_jobs(owner="worker-b", limit=1, now=now + timedelta(minutes=10))
        assert recovered[0].lease_owner == "worker-b"

        assert store.fail_job(
            recovered[0].id,
            owner="worker-b",
            lease_token=recovered[0].lease_token,
            error="provider unavailable",
            now=now + timedelta(minutes=10),
        )
        assert store.list_jobs(status="pending")

        pending = store.list_jobs(status="pending")[0]
        final_claim = store.claim_jobs(owner="worker-c", limit=1, now=now + timedelta(minutes=20))[
            0
        ]
        assert store.fail_job(
            final_claim.id,
            owner="worker-c",
            lease_token=final_claim.lease_token,
            error="provider unavailable",
            now=now + timedelta(minutes=21),
        )
        assert store.list_jobs(status="exhausted")[0].id == job.id
    finally:
        store.close()


def test_fact_commit_and_job_completion_are_atomic(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "atomic.sqlite"))
    try:
        job = store.enqueue_job(DurableJob(kind="extract", max_attempts=2))
        claimed = store.claim_jobs(owner="worker-a", limit=1)[0]
        chunk = _chunk()
        evidence = FactEvidence(fact_id=chunk.id, excerpt="Use Python", source_event_id="event-1")
        revision = FactRevision(fact_id=chunk.id, version=1, content=chunk.content)

        committed = store.commit_fact_and_job(
            claimed.id,
            owner="worker-a",
            lease_token=claimed.lease_token,
            chunk=chunk,
            evidence=[evidence],
            revision=revision,
        )
        assert committed is True
        assert store.get(chunk.id) is not None
        assert store.list_jobs(status="completed")[0].id == job.id
        assert store.list_fact_evidence(chunk.id)[0].source_event_id == "event-1"
        assert store.list_fact_revisions(chunk.id)[0].version == 1
    finally:
        store.close()


def test_stale_same_owner_lease_token_cannot_mutate_reclaimed_job(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "fencing.sqlite"))
    try:
        job = store.enqueue_job(DurableJob(kind="extract", max_attempts=3))
        start = datetime.now(timezone.utc)
        first = store.claim_jobs(owner="same-worker", now=start, lease_seconds=5)[0]
        second = store.claim_jobs(
            owner="same-worker", now=start + timedelta(seconds=6), lease_seconds=30
        )[0]
        assert first.lease_token != second.lease_token
        assert not store.complete_job(
            job.id,
            owner="same-worker",
            lease_token=first.lease_token,
            now=start + timedelta(seconds=7),
        )
        assert store.complete_job(
            job.id,
            owner="same-worker",
            lease_token=second.lease_token,
            now=start + timedelta(seconds=7),
        )
    finally:
        store.close()


def test_atomic_batch_rolls_back_after_mid_commit_failure_and_survives_reopen(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rollback.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    job = store.enqueue_job(DurableJob(kind="extract"))
    now = datetime.now(timezone.utc)
    claimed = store.claim_jobs(owner="worker", now=now)[0]
    good = _chunk("first", "fact-1")
    bad_evidence = FactEvidence(fact_id="missing-fact", excerpt="should fail")
    with pytest.raises(Exception):
        store.commit_fact_and_job(
            job.id,
            owner="worker",
            lease_token=claimed.lease_token,
            chunks=[good],
            evidence=[bad_evidence],
            now=now,
        )
    store.close()
    reopened = SQLiteStore(db_path=str(db_path))
    try:
        assert reopened.get(good.id) is None
        assert reopened.list_fact_evidence("missing-fact") == []
        assert reopened.list_jobs(status="leased")[0].lease_token == claimed.lease_token
    finally:
        reopened.close()


def test_capture_event_and_job_rolls_back_and_reopens_cleanly(tmp_path: Path) -> None:
    db_path = tmp_path / "capture-rollback.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        conflicting_job = store.enqueue_job(
            DurableJob(id="job-1", kind="extract", payload={"event_id": "old"})
        )
        event = EpisodicEvent(
            id="event-new",
            event_fingerprint="finger-new",
            role="user",
            content="new payload",
            normalized_content="new payload",
            source="codex",
            metadata={
                "source_event_id": "source-new",
                "user_profile_id": "profile",
                "session_id": "session",
            },
        )
        with pytest.raises(ValueError, match="Job identity"):
            store.capture_event_and_job(
                event,
                DurableJob(id=conflicting_job.id, kind="extract", payload={"event_id": "new"}),
            )
    finally:
        store.close()
    reopened = SQLiteStore(db_path=str(db_path))
    try:
        assert reopened.get_episodic_event(event.id) is None
        assert len(reopened.list_jobs()) == 1
        assert reopened.list_jobs()[0].payload == {"event_id": "old"}
    finally:
        reopened.close()


def test_atomic_no_fact_decision_processes_events(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "no-fact.sqlite"))
    try:
        event = EpisodicEvent(
            id="event-1",
            event_fingerprint="finger-1",
            role="user",
            content="ordinary request",
            normalized_content="ordinary request",
            source="codex",
        )
        store.append_episodic_event(event)
        job = store.enqueue_job(DurableJob(kind="extract"))
        claimed = store.claim_jobs(owner="worker")[0]
        assert store.commit_fact_and_job(
            job.id,
            owner="worker",
            lease_token=claimed.lease_token,
            processed_event_ids=[event.id],
            result={"decision": "no_durable_facts"},
        )
        assert store.get_episodic_event(event.id).status == "consolidated"
        assert store.list_jobs(status="completed")[0].result["decision"] == "no_durable_facts"
    finally:
        store.close()


def test_exhausted_job_can_be_manually_retried(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "manual-retry.sqlite"))
    try:
        job = store.enqueue_job(DurableJob(kind="extract", max_attempts=1))
        claimed = store.claim_jobs(owner="worker")[0]
        assert store.fail_job(
            job.id, owner="worker", lease_token=claimed.lease_token, error="failed"
        )
        assert store.list_jobs(status="exhausted")[0].attempt_count == 1
        assert store.retry_job(job.id)
        retried = store.list_jobs(status="pending")[0]
        assert retried.attempt_count == 0
        assert retried.last_error is None
    finally:
        store.close()
