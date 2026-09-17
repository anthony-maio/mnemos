from __future__ import annotations

import pytest

from mnemos.config import SpreadingConfig
from mnemos.modules.spreading import SpreadingActivation
from mnemos.utils.embeddings import SimpleEmbeddingProvider


def _graph() -> SpreadingActivation:
    return SpreadingActivation(
        SimpleEmbeddingProvider(dim=3),
        SpreadingConfig(decay_rate=0.2, activation_threshold=0.01, max_hops=8),
    )


def _add(graph: SpreadingActivation, node_id: str) -> None:
    graph.add_node(
        node_id,
        embedding=[1.0, 0.0, 0.0],
        node_id=node_id,
        metadata={"scope": "project", "scope_id": "repo", "user_profile_id": "u"},
    )


def test_versioned_edges_round_trip_traverse_beyond_startup_page() -> None:
    original = _graph()
    for node_id in ("a", "b", "c", "d"):
        _add(original, node_id)
    original.add_edge("a", "b", 0.9)
    original.add_edge("b", "c", 0.8)
    original.add_edge("c", "d", 0.7)
    persisted = original.export_edges()

    restored = _graph()
    for node_id in ("a", "b", "c", "d"):
        _add(restored, node_id)
    assert restored.hydrate_edges(persisted) == 3

    activated = restored.activate(
        "a",
        current_scope="project",
        project_id="repo",
        user_profile_id="u",
        allowed_scopes=("project",),
        max_hops=3,
    )
    assert activated["d"] == pytest.approx(0.9 * 0.8 * 0.8 * 0.8 * 0.8 * 0.7)


def test_incompatible_persisted_contract_is_rejected_before_edge_mutation() -> None:
    graph = _graph()
    _add(graph, "a")
    _add(graph, "b")
    graph.add_edge("a", "b", 0.4)
    before = dict(graph.get_node("a").neighbors)

    incompatible = {
        **graph.persistence_contract(),
        "traversal_version": "1",
        "edges": {"a": {"b": 0.9}},
    }
    with pytest.raises(ValueError, match="incompatible persisted graph contract"):
        graph.hydrate_edges(incompatible)

    assert graph.get_node("a").neighbors == before
    assert graph.get_node("b").neighbors == {"a": 0.4}


def test_raw_edge_map_remains_explicit_legacy_compatibility() -> None:
    graph = _graph()
    _add(graph, "a")
    _add(graph, "b")

    assert graph.hydrate_edges({"a": {"b": 0.6}}) == 1
    assert graph.get_node("a").neighbors == {"b": 0.6}
    assert graph.get_node("b").neighbors == {"a": 0.6}


def test_persistence_contract_is_exposed_in_stats() -> None:
    graph = _graph()
    stats = graph.get_stats()
    contract = graph.persistence_contract()
    assert stats["persistence_schema_version"] == contract["schema_version"]
    assert stats["edge_builder_version"] == contract["edge_builder_version"]
    assert stats["traversal_version"] == contract["traversal_version"]
