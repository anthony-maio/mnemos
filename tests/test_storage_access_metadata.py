"""SQLite access metadata side-table contract tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemos.types import MemoryChunk, ReconsolidationJob
from mnemos.utils.storage import SQLiteStore

STAMP = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _chunk(chunk_id: str = "fact") -> MemoryChunk:
    return MemoryChunk(
        id=chunk_id,
        content=f"content for {chunk_id}",
        embedding=[1.0, 0.0, 0.0],
        metadata={"scope": "project", "scope_id": "repo"},
        created_at=STAMP,
        updated_at=STAMP,
    )


def test_v7_migration_seeds_and_reopens_access_metadata(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    store = SQLiteStore(str(path))
    store.store(_chunk())
    store._conn.execute("DELETE FROM memory_chunk_access")
    store._conn.execute(
        """UPDATE memory_chunks
           SET access_count=4, last_retrieved_at=?, reconsolidation_due_at=?
           WHERE id=?""",
        ("2026-09-16T10:00:00+00:00", "2026-09-18T10:00:00+00:00", "fact"),
    )
    store._conn.execute("DROP TRIGGER memory_chunk_access_ai")
    store._conn.execute("DROP TABLE memory_chunk_access")
    store._conn.execute("DELETE FROM mnemos_meta WHERE key='migration_8'")
    store._conn.execute("UPDATE mnemos_meta SET value='7' WHERE key='schema_version'")
    store._conn.commit()
    store.close()

    reopened = SQLiteStore(str(path))
    try:
        fact = reopened.get("fact")
        assert fact is not None
        assert fact.access_count == 4
        assert fact.last_retrieved_at == datetime(2026, 9, 16, 10, tzinfo=timezone.utc)
        assert fact.reconsolidation_due_at == datetime(2026, 9, 18, 10, tzinfo=timezone.utc)
        assert reopened._conn.execute(
            "SELECT access_count FROM memory_chunk_access WHERE chunk_id='fact'"
        ).fetchone() == (4,)
        assert reopened.migration_backup is not None
        assert Path(reopened.migration_backup).exists()
        backup = sqlite3.connect(reopened.migration_backup)
        try:
            assert backup.execute(
                "SELECT access_count, last_retrieved_at, reconsolidation_due_at "
                "FROM memory_chunks WHERE id='fact'"
            ).fetchone() == (
                4,
                "2026-09-16T10:00:00+00:00",
                "2026-09-18T10:00:00+00:00",
            )
        finally:
            backup.close()
    finally:
        reopened.close()


def test_clear_due_does_not_resurrect_legacy_value_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "clear_due.sqlite"
    store = SQLiteStore(str(path))
    try:
        due = datetime(2026, 9, 18, 10, tzinfo=timezone.utc)
        fact = _chunk()
        fact.reconsolidation_due_at = due
        store.store(fact)
        store.mark_retrieved("fact", retrieved_at=STAMP, reconsolidation_due_at=due)
        assert store.get("fact").reconsolidation_due_at == due  # type: ignore[union-attr]
        store.mark_retrieved_many(["fact"], retrieved_at=STAMP)
        assert store.get("fact").reconsolidation_due_at is None  # type: ignore[union-attr]
    finally:
        store.close()

    reopened = SQLiteStore(str(path))
    try:
        assert reopened.get("fact").reconsolidation_due_at is None  # type: ignore[union-attr]
    finally:
        reopened.close()


def test_reopen_retrieval_increments_once_each_time(tmp_path: Path) -> None:
    path = tmp_path / "reopen.sqlite"
    first = SQLiteStore(str(path))
    first.store(_chunk())
    first.close()

    second = SQLiteStore(str(path))
    assert second.mark_retrieved_many(["fact"], retrieved_at=STAMP) == 1
    second.close()

    third = SQLiteStore(str(path))
    try:
        assert third.mark_retrieved_many(["fact"], retrieved_at=STAMP) == 1
        assert third.get("fact").access_count == 2  # type: ignore[union-attr]
    finally:
        third.close()


def test_store_and_reconsolidation_clear_preserve_side_table_fields(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "store.sqlite"))
    try:
        fact = _chunk()
        fact.access_count = 3
        fact.last_retrieved_at = datetime(2026, 9, 15, 10, tzinfo=timezone.utc)
        fact.reconsolidation_due_at = datetime(2026, 9, 18, 10, tzinfo=timezone.utc)
        store.store(fact)
        rewritten = store.get("fact")
        assert rewritten is not None
        rewritten.content = "rewritten content"
        store.store(rewritten)
        preserved = store.get("fact")
        assert preserved is not None
        assert preserved.access_count == 3
        assert preserved.last_retrieved_at == fact.last_retrieved_at
        assert preserved.reconsolidation_due_at == fact.reconsolidation_due_at

        store.enqueue_reconsolidation_job(
            ReconsolidationJob(chunk_id="fact", context="review", due_at=STAMP)
        )
        assert store.delete_reconsolidation_job("fact") is True
        assert store.get("fact").reconsolidation_due_at is None  # type: ignore[union-attr]
    finally:
        store.close()


def test_touch_updates_access_side_table_and_content_updated_at(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "touch.sqlite"))
    try:
        store.store(_chunk())
        touched_at = datetime(2026, 9, 18, 12, tzinfo=timezone.utc)
        assert store.touch("fact", access_count=7, updated_at=touched_at) is True
        fact = store.get("fact")
        assert fact is not None
        assert fact.access_count == 7
        assert fact.updated_at == touched_at
        assert store.touch("missing", access_count=4, updated_at=STAMP) is False
    finally:
        store.close()


def test_mark_retrieved_many_deduplicates_and_ignores_missing_ids(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "ids.sqlite"))
    try:
        store._conn.execute(
            """INSERT INTO memory_chunks
               (id, content, embedding, metadata, salience, cognitive_state,
                created_at, updated_at, access_count, version, canonical_key,
                user_profile_id, surprisal_score, surprisal_status, cognitive_state_status,
                last_observed_at, observation_count, last_retrieved_at, reconsolidation_due_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            store._chunk_to_row(_chunk()),
        )
        store._conn.commit()
        assert store._conn.execute("SELECT COUNT(*) FROM memory_chunk_access").fetchone() == (1,)
        assert store.mark_retrieved_many(["fact", "fact", "missing"], retrieved_at=STAMP) == 1
        assert store.get("fact").access_count == 1  # type: ignore[union-attr]
    finally:
        store.close()


def test_delete_cascades_access_side_row(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "delete.sqlite"))
    try:
        store.store(_chunk())
        store.mark_retrieved_many(["fact"], retrieved_at=STAMP)
        assert store.delete("fact") is True
        assert (
            store._conn.execute(
                "SELECT 1 FROM memory_chunk_access WHERE chunk_id='fact'"
            ).fetchone()
            is None
        )
    finally:
        store.close()


def test_mark_retrieved_many_rolls_back_side_insert_on_mid_operation_failure(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "rollback.sqlite"))
    try:
        store.store(_chunk())
        store._conn.execute("DELETE FROM memory_chunk_access")
        store._conn.commit()

        def deny_access_update(action: int, table: str | None, *_args: object) -> int:
            if action == sqlite3.SQLITE_UPDATE and table == "memory_chunk_access":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store._conn.set_authorizer(deny_access_update)
        with pytest.raises(sqlite3.DatabaseError):
            store.mark_retrieved_many(["fact"], retrieved_at=STAMP)
        store._conn.set_authorizer(lambda *args: sqlite3.SQLITE_OK)
        assert (
            store._conn.execute(
                "SELECT 1 FROM memory_chunk_access WHERE chunk_id='fact'"
            ).fetchone()
            is None
        )
        assert store.get("fact").access_count == 0  # type: ignore[union-attr]
    finally:
        store.close()


def test_mark_retrieved_many_respects_outer_transaction_rollback(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "outer-rollback.sqlite"))
    try:
        store.store(_chunk())
        store._conn.execute("BEGIN IMMEDIATE")
        store._transaction_active = True
        assert store.mark_retrieved_many(["fact"], retrieved_at=STAMP) == 1
        store._conn.rollback()
        store._transaction_active = False
        assert store.get("fact").access_count == 0  # type: ignore[union-attr]
    finally:
        store.close()


def test_reopen_does_not_reseed_existing_access_rows(tmp_path: Path) -> None:
    path = tmp_path / "no-reseed.sqlite"
    initial = SQLiteStore(str(path))
    initial.store(_chunk())
    initial.mark_retrieved_many(["fact"], retrieved_at=STAMP)
    initial.close()

    class TracedSQLiteStore(SQLiteStore):
        def _backup_before_migration(self) -> None:
            self._migration_sql: list[str] = []
            self._conn.set_trace_callback(self._migration_sql.append)
            super()._backup_before_migration()

    reopened = TracedSQLiteStore(str(path))
    try:
        assert not any(
            "INSERT OR IGNORE INTO MEMORY_CHUNK_ACCESS" in statement.upper()
            and "FROM MEMORY_CHUNKS" in statement.upper()
            for statement in reopened._migration_sql
        )
        assert reopened.get("fact").access_count == 1  # type: ignore[union-attr]
    finally:
        reopened.close()


def test_failed_nested_access_update_preserves_outer_transaction(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "nested-failure.sqlite"))
    try:
        store.store(_chunk())
        store._conn.execute("BEGIN IMMEDIATE")
        store._transaction_active = True
        store._conn.execute("INSERT INTO mnemos_meta(key, value) VALUES ('outer', 'kept')")

        def deny_access_update(action: int, table: str | None, *_args: object) -> int:
            if action == sqlite3.SQLITE_UPDATE and table == "memory_chunk_access":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store._conn.set_authorizer(deny_access_update)
        with pytest.raises(sqlite3.DatabaseError):
            store.mark_retrieved_many(["fact"], retrieved_at=STAMP)
        store._conn.set_authorizer(lambda *args: sqlite3.SQLITE_OK)
        assert store._conn.execute(
            "SELECT value FROM mnemos_meta WHERE key='outer'"
        ).fetchone() == ("kept",)
        store._conn.rollback()
        store._transaction_active = False
    finally:
        store.close()


def test_mark_retrieved_many_does_not_update_wide_chunk_row(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "trace.sqlite"))
    statements: list[str] = []
    try:
        store.store(_chunk())
        store._conn.set_trace_callback(statements.append)
        assert store.mark_retrieved_many(["fact"], retrieved_at=STAMP) == 1
    finally:
        store.close()
    updates = [
        statement.upper()
        for statement in statements
        if statement.lstrip().upper().startswith("UPDATE")
    ]
    assert any("MEMORY_CHUNK_ACCESS" in statement for statement in updates)
    assert all("UPDATE MEMORY_CHUNKS" not in statement for statement in updates)
