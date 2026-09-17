"""Regression coverage for graph rebuild restart state."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.config import MnemosConfig, ModuleConfig, SpreadingConfig
from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore


def _config() -> MnemosConfig:
    return MnemosConfig(
        modules=ModuleConfig(spreading=True),
        spreading=SpreadingConfig(
            auto_connect_threshold=0.5,
            max_neighbors_per_node=1,
            startup_hydration_limit=100,
            startup_auto_connect=False,
        ),
    )


def _fact(identifier: str, embedding: list[float]) -> MemoryChunk:
    return MemoryChunk(
        id=identifier,
        content=f"fact {identifier}",
        embedding=embedding,
        metadata={
            "scope": "project",
            "scope_id": "project-a",
            "project_id": "project-a",
            "user_profile_id": "profile-a",
        },
    )


def _build_initial_graph(path: Path) -> None:
    facts = [
        _fact("a", [1.0, 0.0]),
        _fact("b", [0.99, 0.1]),
        _fact("c", [0.0, 1.0]),
    ]
    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(),
            embedder=SimpleEmbeddingProvider(dim=2),
            store=store,
        )
        for fact in facts:
            store.store(fact)
            engine.refresh_graph_chunk(fact)
        assert store.graph_derivation_status()["ready"] is True
        assert store.get_graph_preferences()["a"]["b"] == pytest.approx(0.9949372, abs=2e-6)
        store._conn.execute("DELETE FROM mnemos_meta WHERE key='graph_preference_sources'")
        store._conn.commit()
    finally:
        store.close()


def test_interrupted_rebuild_restart_drains_before_new_neighbor_insertion(tmp_path: Path) -> None:
    """A completed rebuild must retain all source preferences after restart."""
    path = tmp_path / "interrupted-rebuild.sqlite"
    _build_initial_graph(path)

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(),
            embedder=SimpleEmbeddingProvider(dim=2),
            store=store,
        )
        assert store.graph_rebuild_cursor() == ""
        assert engine.rebuild_graph_batch(limit=1) == 1
        assert store.graph_rebuild_cursor() == "a"
    finally:
        store.close()

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(),
            embedder=SimpleEmbeddingProvider(dim=2),
            store=store,
        )
        assert store.graph_rebuild_cursor() == "a"
        while engine.rebuild_graph_batch(limit=1):
            pass
        assert store.graph_derivation_status()["ready"] is True

        arriving = _fact("d", [1.0, 0.0])
        store.store(arriving)
        engine.refresh_graph_chunk(arriving)

        preferences = store.get_graph_preferences()
        # Independent nearest-neighbor oracle for max_neighbors=1: a and d
        # are identical, so each must select the other after insertion.
        assert preferences["a"]["d"] == pytest.approx(1.0, abs=2e-6)
        assert preferences["d"]["a"] == pytest.approx(1.0, abs=2e-6)
    finally:
        store.close()
