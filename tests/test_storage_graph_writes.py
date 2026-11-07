"""Stale graph writers must preserve facts and transaction boundaries."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def seed(store: SQLiteStore) -> None:
    for identifier in ("source", "target", "other"):
        store.store(MemoryChunk(id=identifier, content=identifier, embedding=[1.0, 0.0]))
    store.replace_graph_neighbors("source", {"target": 0.8})


def test_deleted_source_from_another_process_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    stale_client = SQLiteStore(str(path))
    try:
        seed(stale_client)
        remembered_neighbors = stale_client.get_graph_edges()["source"]
        # This client stays open while an independent process deletes its source.
        # Its cached graph is now stale even though the database is consistent.
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from mnemos.utils.storage import SQLiteStore; "
                "s=SQLiteStore(sys.argv[1]); assert s.delete('source'); s.close()",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert completed.returncode == 0
        stale_client.replace_graph_neighbors("source", remembered_neighbors)
        assert stale_client.get("source") is None
        assert "source" not in stale_client.get_graph_edges()
        assert stale_client._conn.in_transaction is False
        assert stale_client._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # A successful return must release the write lock for another client.
        other_client = SQLiteStore(str(path))
        try:
            other_client.store(MemoryChunk(id="after", content="Later write"))
        finally:
            other_client.close()
        assert stale_client.get("after") is not None
    finally:
        stale_client.close()


def fail_second_edge(store: SQLiteStore) -> None:
    store._conn.execute("""CREATE TEMP TRIGGER fail_graph_insert BEFORE INSERT ON memory_edges
           WHEN NEW.target_id = 'other'
           BEGIN SELECT RAISE(ABORT, 'interrupted graph write'); END""")
    store._conn.commit()


def test_failed_neighbor_replacement_rolls_back_delete_and_partial_insert(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "rollback.sqlite"))
    try:
        seed(store)
        fail_second_edge(store)
        before = store.get_graph_edges()
        with pytest.raises(sqlite3.IntegrityError, match="interrupted graph write"):
            store.replace_graph_neighbors("source", {"target": 0.3, "other": 0.4})
        assert store.get_graph_edges() == before
        assert store._conn.in_transaction is False
        store._conn.execute("DROP TRIGGER fail_graph_insert")
        store.replace_graph_neighbors("source", {"target": 0.5})
        assert store.get_graph_edges() == {"source": {"target": 0.5}}
    finally:
        store.close()


@pytest.mark.parametrize("source", ["source", "missing"])
def test_neighbor_replacement_does_not_commit_caller_transaction(
    tmp_path: Path, source: str
) -> None:
    store = SQLiteStore(str(tmp_path / "outer.sqlite"))
    try:
        seed(store)
        before = store.get_graph_edges()
        store._conn.execute("BEGIN IMMEDIATE")
        store._conn.execute("INSERT INTO mnemos_meta(key, value) VALUES ('caller', 'pending')")
        store.replace_graph_neighbors(source, {"target": 0.2})
        assert store._conn.in_transaction is True
        store._conn.rollback()
        assert store._meta_value("caller") is None
        assert store.get_graph_edges() == before
    finally:
        store.close()


def test_failed_neighbor_replacement_preserves_caller_transaction(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "outer-failure.sqlite"))
    try:
        seed(store)
        fail_second_edge(store)
        before = store.get_graph_edges()
        store._conn.execute("BEGIN IMMEDIATE")
        store._conn.execute("INSERT INTO mnemos_meta(key, value) VALUES ('caller', 'pending')")
        with pytest.raises(sqlite3.IntegrityError, match="interrupted graph write"):
            store.replace_graph_neighbors("source", {"target": 0.3, "other": 0.4})
        assert store.get_graph_edges() == before
        assert store._conn.in_transaction is True
        assert store._meta_value("caller") == "pending"
        store._conn.commit()
        assert store._meta_value("caller") == "pending"
    finally:
        store.close()
