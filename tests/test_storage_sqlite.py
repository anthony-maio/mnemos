"""
tests/test_storage_sqlite.py — SQLiteStore graph persistence and schema tests.
"""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def test_failed_vector_write_rolls_back_fact_and_leaves_connection_usable(tmp_path):
    store = SQLiteStore(str(tmp_path / "atomic.sqlite"))
    try:
        store.store(MemoryChunk(id="original", content="Keep original", embedding=[1.0, 0.0]))
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.store(MemoryChunk(id="bad", content="Never committed", embedding=[1.0, 0.0, 0.0]))
        assert store.get("bad") is None
        assert store._conn.in_transaction is False
        store.store(MemoryChunk(id="valid", content="Still works", embedding=[0.0, 1.0]))
        assert len(store.get_all()) == 2
    finally:
        store.close()


def _make_chunk(chunk_id: str, scope_id: str = "repo-alpha") -> MemoryChunk:
    base = 0.1 if chunk_id == "alpha" else 0.2
    return MemoryChunk(
        id=chunk_id,
        content=f"memory for {chunk_id}",
        embedding=[base, base + 0.1, base + 0.2],
        metadata={"scope": "project", "scope_id": scope_id},
    )


def test_sqlite_store_initializes_graph_schema_and_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "mnemos_graph.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        tables = {
            row[0]
            for row in store._conn.execute(  # noqa: SLF001 - schema verification
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        stats = store.get_stats()
        schema_version = store._conn.execute(  # noqa: SLF001 - schema verification
            "SELECT value FROM mnemos_meta WHERE key = 'schema_version'"
        ).fetchone()
    finally:
        store.close()

    assert "memory_chunks" in tables
    assert "memory_edges" in tables
    assert "retrieval_feedback_events" in tables
    assert "memory_fts" in tables
    assert schema_version == ("7",)
    assert stats["total_chunks"] == 0
    assert stats["related_edges"] == 0
    assert stats["fts_enabled"] is True
    assert stats["schema_version"] == 7
    assert stats["sqlite_vec_enabled"] is True


def test_sqlite_store_graph_neighbors_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "round_trip.sqlite"))
    try:
        alpha = _make_chunk("alpha")
        beta = _make_chunk("beta")
        store.store(alpha)
        store.store(beta)

        store.replace_graph_neighbors(alpha.id, {beta.id: 0.87})

        edge_map = store.get_graph_edges()
    finally:
        store.close()

    assert edge_map == {"alpha": {"beta": 0.87}}


def test_sqlite_store_delete_cascades_graph_edges(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "cascade.sqlite"))
    try:
        alpha = _make_chunk("alpha")
        beta = _make_chunk("beta")
        store.store(alpha)
        store.store(beta)
        store.replace_graph_neighbors(alpha.id, {beta.id: 0.87})
        store.replace_graph_neighbors(beta.id, {alpha.id: 0.87})

        deleted = store.delete(alpha.id)
        edge_map = store.get_graph_edges()
        stats = store.get_stats()
    finally:
        store.close()

    assert deleted is True
    assert edge_map == {}
    assert stats["related_edges"] == 0


def test_sqlite_store_stats_report_related_edges(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "stats.sqlite"))
    try:
        alpha = _make_chunk("alpha")
        beta = _make_chunk("beta")
        store.store(alpha)
        store.store(beta)
        store.replace_graph_neighbors(alpha.id, {beta.id: 0.91})
        store.replace_graph_neighbors(beta.id, {alpha.id: 0.91})

        stats = store.get_stats()
    finally:
        store.close()

    assert stats["total_chunks"] == 2
    assert stats["chunks_with_embeddings"] == 2
    assert stats["related_edges"] == 1


def test_sqlite_store_fts_tracks_chunk_content(tmp_path: Path) -> None:
    db_path = tmp_path / "fts.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        store.store(_make_chunk("alpha"))
    finally:
        store.close()

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT content FROM memory_fts WHERE memory_fts MATCH 'alpha'"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("memory for alpha",)]


def test_sqlite_store_retrieve_uses_sqlite_vec_ordering(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "vec.sqlite"))
    try:
        store.store(
            MemoryChunk(
                id="alpha",
                content="alpha memory",
                embedding=[1.0, 0.0, 0.0],
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )
        store.store(
            MemoryChunk(
                id="beta",
                content="beta memory",
                embedding=[0.8, 0.2, 0.0],
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )
        store.store(
            MemoryChunk(
                id="gamma",
                content="gamma memory",
                embedding=[0.0, 1.0, 0.0],
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )

        results = store.retrieve([1.0, 0.0, 0.0], top_k=2)
        stats = store.get_stats()
    finally:
        store.close()

    assert [chunk.id for chunk in results] == ["alpha", "beta"]
    assert stats["sqlite_vec_enabled"] is True
