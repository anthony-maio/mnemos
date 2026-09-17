"""Standalone mutual top-k graph builder regressions."""

from __future__ import annotations

import math

import pytest

from mnemos.config import SpreadingConfig
from mnemos.modules.spreading import SpreadingActivation
from mnemos.utils.embeddings import SimpleEmbeddingProvider

_VECTORS = {
    "a": [1.0, 0.0],
    "b": [0.98, 0.2],
    "c": [0.8, 0.6],
    "d": [0.0, 1.0],
    "e": [-0.8, 0.6],
    "f": [-0.98, 0.2],
    "g": [-1.0, 0.0],
}


def _builder() -> SpreadingActivation:
    return SpreadingActivation(
        embedder=SimpleEmbeddingProvider(dim=2),
        config=SpreadingConfig(auto_connect_threshold=0.5, max_neighbors_per_node=1),
    )


def _add_all(builder: SpreadingActivation, order: list[str]) -> None:
    for identifier in order:
        builder.add_node(
            identifier,
            embedding=_VECTORS[identifier],
            node_id=identifier,
            metadata={"scope": "project", "scope_id": "project", "user_profile_id": "u"},
        )


def _oracle_preferences(threshold: float = 0.5) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for source_id, source in _VECTORS.items():
        source_norm = math.sqrt(sum(value * value for value in source))
        candidates = []
        for target_id, target in _VECTORS.items():
            if target_id == source_id:
                continue
            target_norm = math.sqrt(sum(value * value for value in target))
            similarity = sum(a * b for a, b in zip(source, target)) / (source_norm * target_norm)
            if similarity >= threshold:
                candidates.append((similarity, target_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        result[source_id] = {target_id: similarity for similarity, target_id in candidates[:1]}
    return result


def _edge_map(builder: SpreadingActivation) -> dict[str, dict[str, float]]:
    return {node.id: dict(node.neighbors) for node in builder.get_all_nodes() if node.neighbors}


def _assert_mutual_degree_one(builder: SpreadingActivation) -> None:
    edges = _edge_map(builder)
    assert all(len(node.neighbors) <= 1 for node in builder.get_all_nodes())
    for source_id, targets in edges.items():
        for target_id in targets:
            assert source_id in edges.get(target_id, {})


def _assert_preferences(builder: SpreadingActivation) -> None:
    actual = builder.get_preferences()
    expected = _oracle_preferences()
    assert actual.keys() == expected.keys()
    for source_id, targets in expected.items():
        assert actual[source_id].keys() == targets.keys()
        for target_id, similarity in targets.items():
            assert actual[source_id][target_id] == pytest.approx(similarity, abs=1e-6)


def test_connect_node_converges_to_same_top1_graph_in_forward_and_reverse_orders() -> None:
    results = []
    for order in (list(_VECTORS), list(reversed(_VECTORS))):
        builder = _builder()
        _add_all(builder, order)
        for identifier in order:
            builder.connect_node(identifier, threshold=0.5)
        _assert_preferences(builder)
        _assert_mutual_degree_one(builder)
        results.append(_edge_map(builder))

    assert results[0] == results[1]


def test_auto_connect_rebuild_is_order_independent_and_mutual() -> None:
    results = []
    for order in (list(_VECTORS), list(reversed(_VECTORS))):
        builder = _builder()
        _add_all(builder, order)
        assert builder.auto_connect(threshold=0.5) >= 0
        _assert_preferences(builder)
        _assert_mutual_degree_one(builder)
        results.append(_edge_map(builder))

    assert results[0] == results[1]


def test_sequential_arrival_refills_prior_sources_to_the_same_top1_oracle() -> None:
    results = []
    for order in (list(_VECTORS), list(reversed(_VECTORS))):
        builder = _builder()
        for identifier in order:
            _add_all(builder, [identifier])
            builder.connect_node(identifier, threshold=0.5)
        _assert_preferences(builder)
        _assert_mutual_degree_one(builder)
        results.append(_edge_map(builder))

    assert results[0] == results[1]


def test_degree_one_threshold_zero_fixture_is_order_independent() -> None:
    results = []
    for order in (list(_VECTORS), list(reversed(_VECTORS))):
        builder = SpreadingActivation(
            embedder=SimpleEmbeddingProvider(dim=2),
            config=SpreadingConfig(auto_connect_threshold=0.0, max_neighbors_per_node=1),
        )
        _add_all(builder, order)
        builder.auto_connect(threshold=0.0)
        actual = builder.get_preferences()
        expected = _oracle_preferences(0.0)
        assert actual.keys() == expected.keys()
        for source_id, targets in expected.items():
            assert actual[source_id].keys() == targets.keys()
            for target_id, similarity in targets.items():
                assert actual[source_id][target_id] == pytest.approx(similarity, abs=1e-6)
        _assert_mutual_degree_one(builder)
        results.append(_edge_map(builder))

    assert results[0] == results[1]


def test_custom_threshold_and_candidate_policy_survive_later_arrival() -> None:
    builder = SpreadingActivation(
        embedder=SimpleEmbeddingProvider(dim=2),
        config=SpreadingConfig(auto_connect_threshold=0.95, max_neighbors_per_node=1),
    )
    builder.add_node(
        "a",
        embedding=[1.0, 0.0],
        node_id="a",
        metadata={"scope": "global"},
    )
    builder.add_node(
        "b",
        embedding=[0.8, 0.6],
        node_id="b",
        metadata={"scope": "global"},
    )
    builder.connect_node("a", threshold=0.5, candidate_ids={"b"})
    assert set(builder.get_preferences()["a"]) == {"b"}

    builder.add_node(
        "c",
        embedding=[0.99, 0.1],
        node_id="c",
        metadata={"scope": "global"},
    )
    builder.connect_node("c", threshold=0.5)

    # The later, stronger c candidate is outside a's retained candidate pool.
    assert set(builder.get_preferences()["a"]) == {"b"}
    builder.connect_node("a")
    assert set(builder.get_preferences()["a"]) == {"b"}


def test_observed_degree_one_counterexample_matches_mutual_oracle():
    vectors = {
        "f0": [0.15152664441082642, -0.988453173414702],
        "f1": [-0.7445369768025836, 0.6675812236527247],
        "f2": [-0.024532014339829166, -0.9996990448492138],
        "f3": [0.8028070946997686, 0.5962388520548764],
        "f4": [-0.25585254710586003, 0.9667158187075681],
        "f5": [-0.4582623919245765, -0.8888169553668325],
        "f6": [-0.152049437648846, -0.98837288940494],
    }
    expected = {}
    for source, vector in vectors.items():
        candidates = []
        for target, other in vectors.items():
            if target != source:
                similarity = sum(a * b for a, b in zip(vector, other)) / math.sqrt(
                    sum(a * a for a in vector) * sum(b * b for b in other)
                )
                if similarity >= 0:
                    candidates.append((-similarity, target))
        expected[source] = min(candidates)[1]
    graphs = []
    for incremental in (True, False):
        for order in (list(vectors), list(reversed(vectors))):
            builder = SpreadingActivation(
                embedder=SimpleEmbeddingProvider(dim=2),
                config=SpreadingConfig(auto_connect_threshold=0.0, max_neighbors_per_node=1),
            )
            for identifier in order:
                builder.add_node(
                    identifier,
                    node_id=identifier,
                    embedding=vectors[identifier],
                    metadata={"scope": "global"},
                )
                if incremental:
                    builder.connect_node(identifier)
            if not incremental:
                builder.auto_connect()
            assert {
                source: next(iter(targets)) for source, targets in builder.get_preferences().items()
            } == expected
            edges = _edge_map(builder)
            assert {
                (source, target) for source, targets in edges.items() for target in targets
            } == {
                (source, target)
                for source, target in expected.items()
                if expected[target] == source
            }
            graphs.append(edges)
    assert all(graph == graphs[0] for graph in graphs)


def test_manual_add_edge_remains_compatible_until_derived_rebuild() -> None:
    builder = _builder()
    _add_all(builder, ["a", "b"])

    assert builder.add_edge("a", "b", 0.91, bidirectional=False)
    assert "b" in builder.get_node("a").neighbors
    assert "a" not in builder.get_node("b").neighbors

    builder.auto_connect(threshold=0.5)
    _assert_mutual_degree_one(builder)
    assert builder.persistence_contract()["edge_builder_version"] == "4"
