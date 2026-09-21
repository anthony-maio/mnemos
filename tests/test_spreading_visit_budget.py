"""Standalone graph calls must honor the configured visit budget."""

import pytest

from mnemos.config import SpreadingConfig
from mnemos.modules.spreading import SpreadingActivation
from mnemos.utils.embeddings import SimpleEmbeddingProvider


def tree(visit_budget):
    graph = SpreadingActivation(
        SimpleEmbeddingProvider(dim=3),
        SpreadingConfig(
            max_visited=visit_budget,
            max_hops=2,
            max_neighbors_per_node=2,
            activation_threshold=0.0,
            decay_rate=0.0,
        ),
    )
    for name in ("a", "b", "c", "d", "e", "f", "g"):
        graph.add_node(name, node_id=name, embedding=[1.0, 0.0, 0.0])
    for parent, child in (("a", "b"), ("a", "c"), ("b", "d"), ("b", "e"), ("c", "f"), ("c", "g")):
        assert graph.add_edge(parent, child, 1.0, bidirectional=False)
    return graph


@pytest.mark.parametrize("method", ["activate", "retrieve"])
def test_omitted_visit_argument_uses_configured_budget(method):
    graph = tree(1)
    if method == "activate":
        assert graph.activate("a") == {"a": 1.0}
    else:
        assert [node.id for node in graph.retrieve([1.0, 0.0, 0.0], top_k=10)] == ["a"]


def test_configured_budget_can_cover_a_branching_neighborhood():
    graph = tree(7)
    assert set(graph.activate("a")) == {"a", "b", "c", "d", "e", "f", "g"}


def test_explicit_visit_argument_overrides_configured_default():
    graph = tree(1)
    assert graph.activate("a", max_visited=3) == {"a": 1.0, "b": 1.0, "c": 1.0}
