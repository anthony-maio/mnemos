"""Graph rollback must preserve the caller's pending content and full-text edit."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def indexed(connection: sqlite3.Connection, term: str) -> list[int]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?", (term,)
        ).fetchall()
    ]


@pytest.mark.parametrize("commit", [False, True], ids=["caller-rollback", "caller-commit"])
def test_graph_failure_preserves_caller_content_and_full_text_transaction(
    tmp_path: Path, commit: bool
) -> None:
    path = tmp_path / "shared.sqlite"
    store = SQLiteStore(str(path))
    observer = None
    try:
        for identifier in ("source", "target", "other"):
            content = "baselinelexeme" if identifier == "source" else identifier
            store.store(MemoryChunk(id=identifier, content=content, embedding=[1.0, 0.0]))
        store.replace_graph_neighbors("source", {"target": 0.8})
        rowid = store._conn.execute(
            "SELECT rowid FROM memory_chunks WHERE id = 'source'"
        ).fetchone()[0]
        before = store.get_graph_edges()
        store._conn.execute("""CREATE TEMP TRIGGER interrupt_graph BEFORE INSERT ON memory_edges
               WHEN NEW.target_id = 'other'
               BEGIN SELECT RAISE(ABORT, 'interrupted graph write'); END""")
        store._conn.commit()
        observer = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)

        store._conn.execute("BEGIN IMMEDIATE")
        store._conn.execute(
            "UPDATE memory_chunks SET content = 'pendinglexeme' WHERE id = 'source'"
        )
        with pytest.raises(sqlite3.IntegrityError, match="interrupted graph write"):
            store.replace_graph_neighbors("source", {"target": 0.3, "other": 0.4})

        assert store.get_graph_edges() == before
        assert store._conn.in_transaction
        assert indexed(store._conn, "pendinglexeme") == [rowid]
        assert indexed(store._conn, "baselinelexeme") == []
        assert indexed(observer, "baselinelexeme") == [rowid]
        assert indexed(observer, "pendinglexeme") == []

        if commit:
            store._conn.commit()
        else:
            store._conn.rollback()
        expected, absent = (
            ("pendinglexeme", "baselinelexeme") if commit else ("baselinelexeme", "pendinglexeme")
        )
        assert not store._conn.in_transaction
        assert store.get_graph_edges() == before
        assert indexed(observer, expected) == [rowid]
        assert indexed(observer, absent) == []
        assert (
            observer.execute("SELECT content FROM memory_chunks WHERE id = 'source'").fetchone()[0]
            == expected
        )
        assert observer.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        if observer is not None:
            observer.close()
        store.close()
