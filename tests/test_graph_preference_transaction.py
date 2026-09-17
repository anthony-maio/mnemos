"""Preference rows, projection, and completion markers share one commit."""

import json

import pytest

from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore


def test_projection_failure_rolls_back_preferences_edges_and_markers(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "atomic-graph.sqlite"))
    try:
        engine = MnemosEngine(store=store, embedder=SimpleEmbeddingProvider(dim=2))
        for identifier in ("a", "b"):
            fact = MemoryChunk(
                id=identifier,
                content=identifier,
                embedding=[1.0, 0.0],
                metadata={"scope": "project", "scope_id": "p", "user_profile_id": "u"},
            )
            store.store(fact)
            engine.refresh_graph_chunk(fact)
        before_preferences = store.get_graph_preferences()
        before_edges = store.get_graph_edges()
        before_status = store.graph_derivation_status()
        before_meta = store._conn.execute(
            "SELECT key,value FROM mnemos_meta ORDER BY key"
        ).fetchall()
        assert before_status["ready"]

        def interrupted_projection(now):
            # Simulate interruption after a partial projection write, not
            # merely before the transaction has changed anything.
            store._conn.execute("DELETE FROM memory_edges WHERE edge_type='SEMANTIC_SIMILAR'")
            raise RuntimeError("projection interrupted")

        monkeypatch.setattr(store, "_project_graph_edges_locked", interrupted_projection)
        with pytest.raises(RuntimeError, match="projection interrupted"):
            store.replace_graph_preferences_batch({"a": {}}, builder_version="unfinished")

        assert store.get_graph_preferences() == before_preferences
        assert store.get_graph_edges() == before_edges
        assert store.graph_derivation_status() == before_status
        assert (
            store._conn.execute("SELECT key,value FROM mnemos_meta ORDER BY key").fetchall()
            == before_meta
        )
    finally:
        store.close()


def test_wrong_builder_marker_cannot_promote_unrebuilt_sources(tmp_path):
    path = str(tmp_path / "wrong-builder.sqlite")
    store = SQLiteStore(path)
    try:
        engine = MnemosEngine(store=store, embedder=SimpleEmbeddingProvider(dim=2))
        for identifier in ("a", "b", "c"):
            fact = MemoryChunk(
                id=identifier,
                content=identifier,
                embedding=[1.0, 0.0],
                metadata={"scope": "project", "scope_id": "p", "user_profile_id": "u"},
            )
            store.store(fact)
            engine.refresh_graph_chunk(fact)
        marker = json.loads(store._meta_value("graph_preference_sources"))
        marker["version"] = "obsolete-builder"
        store._conn.execute(
            "UPDATE mnemos_meta SET value=? WHERE key='graph_preference_sources'",
            (json.dumps(marker),),
        )
        store._conn.commit()
    finally:
        store.close()

    store = SQLiteStore(path)
    try:
        engine = MnemosEngine(store=store, embedder=SimpleEmbeddingProvider(dim=2))
        assert store.graph_derivation_status()["pending_count"] == 3
        assert store.get_graph_preferences(completed_only=True) == {}
        assert engine.rebuild_graph_batch(limit=1) == 1
        assert store.graph_derivation_status()["marked_sources"] == 1
        assert store.graph_derivation_status()["ready"] is False
        while engine.rebuild_graph_batch(limit=1):
            pass
        assert store.graph_derivation_status()["ready"] is True
        assert store.get_graph_edges()["a"]["b"] == pytest.approx(1.0)
    finally:
        store.close()
