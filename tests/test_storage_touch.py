"""
tests/test_storage_touch.py — Fast access-count persistence helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import InMemoryStore, SQLiteStore


@pytest.mark.parametrize("legacy_trigger", [False, True])
def test_sqlite_touch_does_not_reindex_unchanged_content(tmp_path, legacy_trigger) -> None:
    """Regression test for https://github.com/anthony-maio/mnemos/pull/7.

    Access metadata must not write to the full-text index. Existing databases
    must get the same behavior when their schema is initialized again.
    """
    path = str(tmp_path / "touch_fts.sqlite")
    store = SQLiteStore(path)
    chunk = MemoryChunk(id="alpha", content="original searchable memory")
    store.store(chunk)
    if legacy_trigger:
        store._conn.execute("DROP TRIGGER memory_chunks_au")
        store._conn.execute("""
            CREATE TRIGGER memory_chunks_au AFTER UPDATE ON memory_chunks
            BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            """)
        store._conn.commit()
        store.close()
        store = SQLiteStore(path)

    try:
        for access_count in (None, 9):
            before = store._conn.total_changes
            assert store.touch("alpha", access_count=access_count)
            assert store._conn.total_changes - before == 1

        touched = store.get("alpha")
        assert touched is not None
        assert touched.access_count == 9
        assert store._conn.execute(
            "SELECT content FROM memory_fts WHERE memory_fts MATCH 'original'"
        ).fetchall() == [(chunk.content,)]

        touched.metadata["note"] = "metadata update"
        before = store._conn.total_changes
        store.store(touched)
        assert store._conn.total_changes - before == 1

        touched.content = "replacement searchable memory"
        store.store(touched)
        assert (
            store._conn.execute(
                "SELECT content FROM memory_fts WHERE memory_fts MATCH 'original'"
            ).fetchall()
            == []
        )
        assert store._conn.execute(
            "SELECT content FROM memory_fts WHERE memory_fts MATCH 'replacement'"
        ).fetchall() == [(touched.content,)]
    finally:
        store.close()


@pytest.mark.parametrize("store_factory", [InMemoryStore, SQLiteStore])
def test_store_touch_persists_access_count_and_updated_at(store_factory, tmp_path) -> None:
    if store_factory is SQLiteStore:
        store = store_factory(str(tmp_path / "touch.sqlite"))
    else:
        store = store_factory()

    chunk = MemoryChunk(
        id="alpha",
        content="project memory",
        embedding=[1.0, 0.0, 0.0],
        metadata={"scope": "project", "scope_id": "repo-alpha"},
    )
    store.store(chunk)

    touched_at = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
    assert store.touch(chunk.id, access_count=3, updated_at=touched_at) is True

    touched = store.get(chunk.id)
    assert touched is not None
    assert touched.access_count == 3
    assert touched.updated_at == touched_at

    close = getattr(store, "close", None)
    if callable(close):
        close()
