"""Regression coverage for the persisted deterministic graph builder."""

from __future__ import annotations

import json
from pathlib import Path

from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore


def _fact(identifier: str, vector: list[float]) -> MemoryChunk:
    return MemoryChunk(
        id=identifier,
        content=f"fact {identifier}",
        embedding=vector,
        metadata={
            "scope": "project",
            "scope_id": "p",
            "user_profile_id": "profile",
        },
    )


def _run_order(
    facts: list[MemoryChunk],
    tmp_path: Path,
    query_embedding: list[float] | None = None,
    *,
    incremental: bool = False,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(str(tmp_path / "graph.sqlite"))
    try:
        engine = MnemosEngine(embedder=SimpleEmbeddingProvider(dim=2), store=store)
        if incremental:
            for fact in facts:
                store.store(fact)
                store._conn.commit()
                engine.refresh_graph_chunk(fact)
        else:
            for fact in facts:
                store.store(fact)
            store._conn.commit()
            for fact in facts:
                engine.refresh_graph_chunk(fact)
        query = query_embedding or facts[0].embedding or []
        results = engine.spreading_activation.retrieve(
            query,
            top_k=len(facts),
            current_scope="project",
            scope_id="p",
            user_profile_id="profile",
            allowed_scopes=("project",),
        )
        return store.get_graph_edges(), [item.id for item in results]
    finally:
        store.close()


def test_persisted_graph_is_identical_for_forward_and_reverse_insertion(tmp_path: Path) -> None:
    facts = [
        _fact("a", [1.0, 0.0]),
        _fact("b", [0.99, 0.1]),
        _fact("c", [0.1, 0.99]),
        _fact("d", [0.7, 0.7]),
    ]
    forward, forward_results = _run_order(facts, tmp_path / "forward", [1.0, 0.0])
    reverse, reverse_results = _run_order(list(reversed(facts)), tmp_path / "reverse", [1.0, 0.0])
    assert forward == reverse
    assert forward_results == reverse_results


def test_correction_and_delete_refill_preferences_without_stale_targets(tmp_path: Path) -> None:
    path = tmp_path / "correction.sqlite"
    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(embedder=SimpleEmbeddingProvider(dim=2), store=store)
        facts = [_fact("a", [1.0, 0.0]), _fact("b", [0.99, 0.1]), _fact("c", [0.1, 0.99])]
        for fact in facts:
            store.store(fact)
        store._conn.commit()
        for fact in facts:
            engine.refresh_graph_chunk(fact)

        corrected = _fact("b", [-1.0, 0.0])
        store.store(corrected)
        store._conn.commit()
        engine.refresh_graph_chunk(corrected)
        assert all("b" not in targets for targets in store.get_graph_edges().values())
        assert store.graph_derivation_status()["ready"]

        assert store.delete("c")
        assert not store.graph_derivation_status()["ready"]
        assert "c" not in store.get_graph_edges()
    finally:
        store.close()


def test_preference_rows_do_not_appear_as_traversable_neighbors(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "edge-types.sqlite"))
    try:
        for fact in (_fact("a", [1.0, 0.0]), _fact("b", [0.99, 0.1])):
            store.store(fact)
        store._conn.commit()
        store.replace_graph_preferences("a", {"b": 0.9})

        def scope_filter(chunk: MemoryChunk) -> bool:
            return chunk.id in {"a", "b"}

        scope_filter._mnemos_scope_filter = {
            "user_profile_id": "profile",
            "project_id": "p",
            "allowed_scopes": ("project",),
        }
        assert store.get_graph_neighbors("a", filter_fn=scope_filter) == []
    finally:
        store.close()


def test_invalid_preference_weight_is_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "invalid-preference.sqlite"))
    try:
        store.store(_fact("a", [1.0, 0.0]))
        store.store(_fact("b", [0.0, 1.0]))
        store._conn.commit()
        try:
            store.replace_graph_preferences("a", {"b": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid graph preference weight was accepted")
        assert store.get_graph_preferences() == {}
    finally:
        store.close()


def test_empty_preferences_are_a_ready_restart_state(tmp_path: Path) -> None:
    path = tmp_path / "empty-preferences.sqlite"
    store = SQLiteStore(str(path))
    fact = _fact("isolated", [1.0, 0.0])
    try:
        engine = MnemosEngine(embedder=SimpleEmbeddingProvider(dim=2), store=store)
        store.store(fact)
        store._conn.commit()
        engine.refresh_graph_chunk(fact)
        assert store.graph_derivation_status()["ready"]
    finally:
        store.close()

    restarted_store = SQLiteStore(str(path))
    try:
        restarted = MnemosEngine(embedder=SimpleEmbeddingProvider(dim=2), store=restarted_store)
        assert restarted_store.graph_derivation_status()["ready"]
        assert restarted.spreading_activation.get_stats()["graph_derivation_status"] == "ready"
    finally:
        restarted_store.close()


def test_malformed_preference_marker_is_pending_without_hot_path_crash(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "malformed-marker.sqlite"))
    fact = _fact("isolated", [1.0, 0.0])
    try:
        engine = MnemosEngine(embedder=SimpleEmbeddingProvider(dim=2), store=store)
        store.store(fact)
        store._conn.commit()
        engine.refresh_graph_chunk(fact)
        raw = store._conn.execute(
            "SELECT value FROM mnemos_meta WHERE key='graph_preference_sources'"
        ).fetchone()[0]
        marker = json.loads(raw)
        marker["sources"][fact.id] = []
        store._conn.execute(
            "UPDATE mnemos_meta SET value=? WHERE key='graph_preference_sources'",
            (json.dumps(marker, sort_keys=True),),
        )
        store._conn.commit()
        status = store.graph_derivation_status()
        assert status["ready"] is False
        assert status["pending_count"] == 1
    finally:
        store.close()


def test_fixture_observed_vectors_converge_without_sorting_results(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "graph-insertion-development-v1.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    facts = [
        MemoryChunk(
            id=item["id"],
            content=item["content"],
            embedding=item["embedding"],
            metadata={
                "scope": item["scope"],
                "scope_id": item["project_id"],
                "workspace_id": item.get("workspace_id"),
                "user_profile_id": item["user_profile_id"],
            },
        )
        for item in payload["case"]["facts"]
    ]
    query = payload["case"]["query"]["embedding"]
    forward, forward_results = _run_order(
        facts, tmp_path / "fixture-forward", query, incremental=True
    )
    reverse, reverse_results = _run_order(
        list(reversed(facts)), tmp_path / "fixture-reverse", query, incremental=True
    )
    assert forward == reverse
    assert forward_results == reverse_results
