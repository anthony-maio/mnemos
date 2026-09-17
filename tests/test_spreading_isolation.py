from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemos.config import SpreadingConfig
from mnemos.modules.spreading import SpreadingActivation
from mnemos.utils.embeddings import SimpleEmbeddingProvider


@pytest.fixture
def embedder() -> SimpleEmbeddingProvider:
    return SimpleEmbeddingProvider(dim=3)


def graph(config: SpreadingConfig | None = None) -> SpreadingActivation:
    return SpreadingActivation(SimpleEmbeddingProvider(dim=3), config=config)


def node(sa: SpreadingActivation, node_id: str, *, vector=None, **metadata):
    return sa.add_node(
        node_id,
        embedding=vector or [1.0, 0.0, 0.0],
        node_id=node_id,
        metadata=metadata,
    )


def test_chain_uses_max_path_energy_and_is_repeatable() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.2, activation_threshold=0.05, max_hops=5))
    for name in ("a", "b", "c", "d"):
        node(sa, name)
    sa.add_edge("a", "b", 1.0, bidirectional=False)
    sa.add_edge("b", "c", 1.0, bidirectional=False)
    sa.add_edge("c", "d", 1.0, bidirectional=False)

    first = sa.activate("a")
    second = sa.activate("a")

    assert first == second
    assert first["a"] == pytest.approx(1.0)
    assert first["b"] == pytest.approx(0.8)
    assert first["c"] == pytest.approx(0.64)
    assert first["d"] == pytest.approx(0.512)
    assert all(sa.get_node(name).energy == 0.0 for name in ("a", "b", "c", "d"))


def test_cycle_does_not_amplify_or_revisit_forever() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.0, activation_threshold=0.0, max_hops=100))
    for name in ("a", "b", "c"):
        node(sa, name)
    sa.add_edge("a", "b", 1.0)
    sa.add_edge("b", "c", 1.0, bidirectional=False)
    sa.add_edge("c", "a", 1.0, bidirectional=False)

    activated = sa.activate("a", max_visited=3)

    assert activated == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_profiled_seed_obeys_activation_threshold() -> None:
    sa = graph(SpreadingConfig(activation_threshold=0.5, max_hops=1))
    node(sa, "a", scope="project", scope_id="repo", user_profile_id="u")

    activated = sa.activate(
        "a",
        energy=0.1,
        current_scope="project",
        project_id="repo",
        user_profile_id="u",
        allowed_scopes=("project",),
    )

    assert activated == {}


def test_diamond_selects_strongest_path_instead_of_summing_paths() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.0, activation_threshold=0.0, max_hops=3))
    for name in ("a", "b", "c", "d"):
        node(sa, name)
    sa.add_edge("a", "b", 0.5, bidirectional=False)
    sa.add_edge("a", "c", 0.9, bidirectional=False)
    sa.add_edge("b", "d", 0.8, bidirectional=False)
    sa.add_edge("c", "d", 0.2, bidirectional=False)

    activated = sa.activate("a")

    assert activated["d"] == pytest.approx(0.4)


def test_scope_bridge_cannot_reach_foreign_project_or_profile() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.0, activation_threshold=0.0, max_hops=5))
    common = {"user_profile_id": "user-a"}
    node(sa, "alpha", vector=[1.0, 0.0, 0.0], scope="project", scope_id="alpha", **common)
    node(sa, "shared", vector=[1.0, 0.0, 0.0], scope="global", **common)
    node(sa, "beta", vector=[1.0, 0.0, 0.0], scope="project", scope_id="beta", **common)
    node(sa, "other-user", vector=[1.0, 0.0, 0.0], scope="global", user_profile_id="user-b")
    sa.add_edge("alpha", "shared", 1.0, bidirectional=False)
    sa.add_edge("shared", "beta", 1.0, bidirectional=False)
    sa.add_edge("shared", "other-user", 1.0, bidirectional=False)

    results = sa.retrieve(
        [1.0, 0.0, 0.0],
        top_k=10,
        current_scope="project",
        project_id="alpha",
        user_profile_id="user-a",
        allowed_scopes=("project", "global"),
    )

    assert [result.id for result in results] == ["alpha", "shared"]
    assert sa.retrieve([1.0, 0.0, 0.0], top_k=10) == []


def test_filter_and_seed_ids_are_applied_before_graph_traversal() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.0, activation_threshold=0.0, max_hops=2))
    for name in ("a", "b", "c"):
        node(sa, name, scope="project", scope_id="repo", user_profile_id="u")
    sa.add_edge("a", "b", 1.0)
    sa.add_edge("b", "c", 1.0, bidirectional=False)

    results = sa.retrieve(
        [1.0, 0.0, 0.0],
        top_k=10,
        current_scope="project",
        project_id="repo",
        user_profile_id="u",
        allowed_scopes=("project",),
        seed_ids=["b"],
        filter_fn=lambda item: item.id != "c",
    )

    assert [result.id for result in results] == ["a", "b"]


def test_explicit_filter_can_authorize_profiled_legacy_callers() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.0, activation_threshold=0.0, max_hops=1))
    node(sa, "a", scope="project", scope_id="repo-a", user_profile_id="u")
    node(sa, "b", scope="project", scope_id="repo-a", user_profile_id="u")
    sa.add_edge("a", "b", 1.0, bidirectional=False)

    results = sa.retrieve(
        [1.0, 0.0, 0.0],
        top_k=10,
        filter_fn=lambda item: item.metadata.get("scope_id") == "repo-a",
        seed_ids=["a"],
    )

    assert [result.id for result in results] == ["a", "b"]


def test_explicit_legacy_scope_ids_remain_compatible() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.0, activation_threshold=0.0, max_hops=1))
    node(sa, "repo", scope="project", scope_id="repo")
    node(sa, "other", scope="project", scope_id="other")

    results = sa.retrieve(
        [1.0, 0.0, 0.0],
        top_k=10,
        current_scope="project",
        scope_id="repo",
        allowed_scopes=("project",),
    )

    assert [result.id for result in results] == ["repo"]


def test_refresh_replaces_content_and_keeps_only_valid_neighbors() -> None:
    sa = graph()
    node(sa, "a", scope="project", scope_id="repo", user_profile_id="u")
    node(sa, "b", scope="project", scope_id="repo", user_profile_id="u")
    node(sa, "c", scope="project", scope_id="other", user_profile_id="u")
    sa.add_edge("a", "b", 0.8)
    sa.add_edge("a", "c", 0.7) is False

    refreshed = sa.refresh_node(
        "a",
        "corrected content",
        embedding=[0.0, 1.0, 0.0],
        metadata={"scope": "project", "scope_id": "repo", "user_profile_id": "u"},
    )

    assert refreshed.content == "corrected content"
    assert refreshed.neighbors == {"b": 0.8}
    assert "a" in sa.get_node("b").neighbors


def test_concurrent_queries_do_not_share_energy_state() -> None:
    sa = graph(SpreadingConfig(decay_rate=0.25, activation_threshold=0.01, max_hops=3))
    for name in ("a", "b", "c"):
        node(sa, name)
    sa.add_edge("a", "b", 0.8, bidirectional=False)
    sa.add_edge("b", "c", 0.8, bidirectional=False)

    def run() -> dict[str, float]:
        return sa.activate("a")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(32)))

    assert all(result == results[0] for result in results)
    assert all(sa.get_node(name).energy == 0.0 for name in ("a", "b", "c"))
