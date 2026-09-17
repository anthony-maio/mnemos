"""Atomic SQLite schema startup and migration failure regressions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def _schema_tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_fresh_schema_failure_rolls_back_every_object_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    path = tmp_path / "fresh.sqlite"
    original_apply = SQLiteStore._apply_migrations

    def fail_after_schema_ddl(store: SQLiteStore) -> None:
        original_apply(store)
        raise failure("injected migration failure")

    monkeypatch.setattr(SQLiteStore, "_apply_migrations", fail_after_schema_ddl)
    with pytest.raises(failure, match="injected migration failure"):
        SQLiteStore(str(path))

    # Before the explicit transaction, the DDL before the first migration DML
    # survived this same failure point. A retry must see an empty database.
    assert _schema_tables(path) == set()

    monkeypatch.undo()
    store = SQLiteStore(str(path))
    try:
        assert store._meta_value("schema_version") == "8"  # noqa: SLF001
        assert "memory_chunks" in _schema_tables(path)
        assert (
            store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        )  # noqa: SLF001
        assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # noqa: SLF001
        store.store(MemoryChunk(id="retry", content="Retry fact", embedding=[1.0, 0.0]))
        assert store.get("retry").content == "Retry fact"
    finally:
        store.close()


def test_migration_failure_preserves_prior_data_and_verified_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "legacy.sqlite"
    store = SQLiteStore(str(path))
    store.store(MemoryChunk(id="kept", content="Prior durable fact", embedding=[1.0, 0.0]))
    store._conn.execute("DROP TRIGGER memory_chunks_au")  # noqa: SLF001
    store._conn.execute("""CREATE TRIGGER memory_chunks_au
        AFTER UPDATE OF content ON memory_chunks
        BEGIN
            SELECT NEW.content;
        END""")  # noqa: SLF001
    store._conn.execute(
        "UPDATE mnemos_meta SET value='4' WHERE key='schema_version'"
    )  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    store.close()

    with sqlite3.connect(path) as connection:
        before_schema = connection.execute(
            "SELECT name, type, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

    def fail_before_migration_dml(_store: SQLiteStore) -> None:
        raise RuntimeError("injected pre-DML migration failure")

    monkeypatch.setattr(SQLiteStore, "_apply_migrations", fail_before_migration_dml)
    with pytest.raises(RuntimeError, match="injected pre-DML migration failure"):
        SQLiteStore(str(path))

    with sqlite3.connect(path) as connection:
        after_schema = connection.execute(
            "SELECT name, type, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        assert after_schema == before_schema
        assert connection.execute(
            "SELECT value FROM mnemos_meta WHERE key='schema_version'"
        ).fetchone() == ("4",)
        assert connection.execute(
            "SELECT content FROM memory_chunks WHERE id='kept'"
        ).fetchone() == ("Prior durable fact",)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)

    backups = sorted(tmp_path.glob("legacy.sqlite.pre-schema-8-*.sqlite"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert (
            backup.execute(
                "SELECT name, type, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
            == before_schema
        )
        assert backup.execute(
            "SELECT value FROM mnemos_meta WHERE key='schema_version'"
        ).fetchone() == ("4",)
        assert backup.execute("SELECT content FROM memory_chunks WHERE id='kept'").fetchone() == (
            "Prior durable fact",
        )
        assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
