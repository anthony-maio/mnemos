from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from mnemos.jobs import DurableJob
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import SQLiteStore


def make_event() -> EpisodicEvent:
    content = "The marker is cedar."
    return EpisodicEvent(
        id="source-1",
        event_fingerprint="fingerprint-1",
        role="user",
        content=content,
        normalized_content=content.casefold(),
        scope="project",
        scope_id="project-1",
        source="manual",
        metadata={"user_profile_id": "profile-1", "project_id": "project-1"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def prepare(store: SQLiteStore) -> tuple[EpisodicEvent, DurableJob]:
    event = make_event()
    store.append_episodic_event(event)
    job = store.enqueue_job(DurableJob(kind="extract", payload={"event_id": event.id}))
    return event, store.claim_jobs(owner="worker")[0]


def commit_empty(
    store: SQLiteStore,
    claimed: DurableJob,
    event: EpisodicEvent,
    *,
    result: dict[str, object] | None = None,
    chunks: list[MemoryChunk] | None = None,
    now: datetime | None = None,
) -> bool:
    return store.commit_fact_and_job(
        claimed.id,
        owner="worker",
        lease_token=claimed.lease_token,
        chunks=chunks,
        processed_event_ids=[event.id],
        expected_source_event=event,
        result=result or {"decision": "no_durable_facts"},
        now=now,
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("content", "Changed source."),
        ("role", "assistant"),
        ("scope", "workspace"),
        ("scope_id", "workspace-1"),
        ("event_fingerprint", "changed-fingerprint"),
        ("metadata", json.dumps({"user_profile_id": "other-profile"})),
        ("status", "deferred"),
    ],
)
def test_changed_source_is_rejected_without_fact_or_job_write(
    tmp_path: Path, column: str, value: str
) -> None:
    store = SQLiteStore(db_path=str(tmp_path / f"{column}.sqlite"))
    try:
        event, claimed = prepare(store)
        store._conn.execute(f"UPDATE episodic_events SET {column}=? WHERE id=?", (value, event.id))
        store._conn.commit()
        with pytest.raises(ValueError, match="Expected source event changed"):
            commit_empty(store, claimed, event)
        current_job = store.get_job(claimed.id)
        current_event = store.get_episodic_event(event.id)
        assert current_job is not None and current_job.status == "leased"
        assert current_event is not None and current_event.status != "consolidated"
    finally:
        store.close()


def test_missing_source_is_rejected_atomically(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "missing.sqlite"))
    try:
        event, claimed = prepare(store)
        store._conn.execute("DELETE FROM episodic_events WHERE id=?", (event.id,))
        store._conn.commit()
        with pytest.raises(ValueError, match="Expected source event changed"):
            commit_empty(store, claimed, event)
        current_job = store.get_job(claimed.id)
        assert current_job is not None and current_job.status == "leased"
    finally:
        store.close()


def test_valid_snapshot_commits_empty_result(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "valid.sqlite"))
    try:
        event, claimed = prepare(store)
        assert commit_empty(store, claimed, event)
        current_job = store.get_job(claimed.id)
        assert current_job is not None and current_job.status == "completed"
    finally:
        store.close()


def test_snapshot_failure_rolls_back_fact_and_job(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "rollback.sqlite"))
    try:
        event, claimed = prepare(store)
        store._conn.execute(
            "UPDATE episodic_events SET content=? WHERE id=?", ("Changed.", event.id)
        )
        store._conn.commit()
        chunk = MemoryChunk(
            content="A durable fact.",
            metadata={"scope": "project", "scope_id": "project-1", "user_profile_id": "profile-1"},
        )
        with pytest.raises(ValueError):
            commit_empty(
                store,
                claimed,
                event,
                chunks=[chunk],
                result={"decision": "facts"},
            )
        assert store.get(chunk.id) is None
        current_job = store.get_job(claimed.id)
        assert current_job is not None and current_job.status == "leased"
    finally:
        store.close()


def test_expired_lease_returns_false_before_snapshot_write(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "expired.sqlite"))
    try:
        event, claimed = prepare(store)
        assert claimed.lease_until is not None
        assert (
            commit_empty(
                store,
                claimed,
                event,
                now=claimed.lease_until + timedelta(seconds=1),
            )
            is False
        )
        current_job = store.get_job(claimed.id)
        assert current_job is not None and current_job.status == "leased"
    finally:
        store.close()
