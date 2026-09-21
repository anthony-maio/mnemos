"""Review retention follows suppression across schema upgrades and restore."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from mnemos.backup import create_backup, restore_backup
from mnemos.jobs import DurableJob
from mnemos.scopes import ScopeRegistry
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore
from tests.test_source_review import reference, source


def test_reference_only_target_dependency_is_invalidated_on_forget(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "target-only.sqlite"))
    try:
        event = source()
        store.append_episodic_event(event)
        target = MemoryChunk(
            id="target",
            content="The old rule",
            embedding=[1.0, 0.0],
            metadata={"scope": "project", "scope_id": "project-a", "user_profile_id": "owner"},
        )
        store.store(target)
        store.enqueue_job(DurableJob(kind="extract", payload={"event_id": event.id}))
        job = store.claim_jobs(owner="worker")[0]
        assert store.commit_fact_and_job(
            job.id,
            owner="worker",
            lease_token=job.lease_token,
            processed_event_ids=[event.id],
            review_references=[
                reference(
                    event,
                    operation="conflict",
                    target_fact_id=target.id,
                    expected_version=target.version,
                )
            ],
            result={"decision": "review_pending"},
        )
        assert (
            store.list_source_reviews(
                user_profile_id="other", scope="project", scope_id="project-a"
            )
            == []
        )
        assert (
            store.list_source_reviews(user_profile_id="owner", scope="project", scope_id="other")
            == []
        )
        assert store.forget_fact(target.id, expected_version=target.version)
        assert store.get_episodic_event(event.id) is not None
        rows = store.list_source_reviews(
            user_profile_id="owner", scope="project", scope_id="project-a"
        )
        assert rows[0]["state"] == "invalidated_target"
    finally:
        store.close()


def test_failure_after_review_insert_rolls_back_every_write(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "rollback.sqlite"))
    try:
        event = source()
        store.append_episodic_event(event)
        store.enqueue_job(DurableJob(kind="extract", payload={"event_id": event.id}))
        job = store.claim_jobs(owner="worker")[0]
        chunk = MemoryChunk(id="new-fact", content="A new fact", embedding=[1.0, 0.0])
        with pytest.raises(ValueError, match="missing episodic event"):
            store.commit_fact_and_job(
                job.id,
                owner="worker",
                lease_token=job.lease_token,
                chunks=[chunk],
                processed_event_ids=[event.id, "missing-event"],
                review_references=[reference(event)],
                result={"decision": "facts"},
            )
        assert store.get(chunk.id) is None
        assert store._conn.execute("SELECT COUNT(*) FROM source_review_records").fetchone() == (0,)
        current_event = store.get_episodic_event(event.id)
        current_job = store.get_job(job.id)
        assert current_event is not None and current_event.status == "pending"
        assert current_job is not None and current_job.status == "leased"
    finally:
        store.close()


def test_restore_preserves_review_then_applies_newer_source_suppression(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "current.sqlite"))
    try:
        event = source()
        store.append_episodic_event(event)
        store.enqueue_job(DurableJob(kind="extract", payload={"event_id": event.id}))
        job = store.claim_jobs(owner="worker")[0]
        assert store.commit_fact_and_job(
            job.id,
            owner="worker",
            lease_token=job.lease_token,
            processed_event_ids=[event.id],
            review_references=[reference(event)],
            result={"decision": "review_pending"},
        )
        archive = tmp_path / "snapshot.zip"
        create_backup(store, archive, ScopeRegistry())
        first_path = tmp_path / "first.sqlite"
        restore_backup(archive, first_path)
        first = SQLiteStore(str(first_path))
        try:
            assert (
                first.list_source_reviews(
                    user_profile_id="owner", scope="project", scope_id="project-a"
                )[0]["state"]
                == "open"
            )
            assert (
                first.prune_processed_evidence(
                    before=datetime.now(timezone.utc) + timedelta(days=1)
                )
                == 0
            )
        finally:
            first.close()
        store.record_suppression(source_event_id=event.id, reason="explicit_forget")
        second_path = tmp_path / "second.sqlite"
        restore_backup(archive, second_path, current_store=store)
        second = SQLiteStore(str(second_path))
        try:
            assert second.get_episodic_event(event.id) is None
            rows = second.list_source_reviews(
                user_profile_id="owner", scope="project", scope_id="project-a"
            )
            assert rows[0]["state"] == "invalidated_source"
            assert event.content not in str(rows)
            assert second._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            second.close()
    finally:
        store.close()


def test_schema_eight_upgrade_preserves_data_and_pre_upgrade_backup(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    store = SQLiteStore(str(path))
    try:
        store.store(MemoryChunk(id="existing", content="Keep this fact", embedding=[1.0, 0.0]))
        for name in ("review_source_deleted", "review_target_deleted", "review_source_suppressed"):
            store._conn.execute(f"DROP TRIGGER {name}")
        store._conn.execute("DROP TABLE source_review_records")
        store._conn.execute("UPDATE mnemos_meta SET value='8' WHERE key='schema_version'")
        store._conn.commit()
    finally:
        store.close()
    upgraded = SQLiteStore(str(path))
    try:
        existing = upgraded.get("existing")
        assert existing is not None and existing.content == "Keep this fact"
        assert upgraded.get_stats()["schema_version"] == 9
        assert (
            upgraded.list_source_reviews(
                user_profile_id="owner", scope="project", scope_id="project-a"
            )
            == []
        )
    finally:
        upgraded.close()
    backups = list(tmp_path.glob("old.sqlite.pre-schema-9-*.sqlite"))
    assert len(backups) == 1
    connection = sqlite3.connect(backups[0])
    try:
        assert connection.execute(
            "SELECT value FROM mnemos_meta WHERE key='schema_version'"
        ).fetchone() == ("8",)
        assert connection.execute(
            "SELECT content FROM memory_chunks WHERE id='existing'"
        ).fetchone() == ("Keep this fact",)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='source_review_records'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()
