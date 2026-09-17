"""Adversarial coverage for partition-safe persisted graph derivation."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Iterable

import pytest

from mnemos.config import MnemosConfig, SpreadingConfig
from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore

PROFILE = "profile-p"
WORKSPACE = "workspace-shared"
PROJECT_A = "project-a"
PROJECT_B = "project-b"
THRESHOLD = 0.6
MAX_NEIGHBORS = 4


def _fact(
    identifier: str,
    vector: list[float],
    *,
    scope: str,
    scope_id: str | None,
    project_id: str | None = None,
    workspace_id: str | None = None,
) -> MemoryChunk:
    return MemoryChunk(
        id=identifier,
        content=f"fact {identifier}",
        embedding=vector,
        metadata={
            "user_profile_id": PROFILE,
            "scope": scope,
            "scope_id": scope_id,
            "project_id": project_id,
            "workspace_id": workspace_id,
        },
    )


def _facts() -> list[MemoryChunk]:
    # All vectors are above the similarity threshold. Scope compatibility is
    # therefore the only reason a pair is absent from the independent oracle.
    return [
        _fact("global", [1.0, 0.0, 0.0], scope="global", scope_id=None),
        _fact(
            "workspace",
            [0.99, 0.04, 0.0],
            scope="workspace",
            scope_id=WORKSPACE,
            workspace_id=WORKSPACE,
        ),
        _fact(
            "a1",
            [0.98, 0.08, 0.0],
            scope="project",
            scope_id=PROJECT_A,
            project_id=PROJECT_A,
            workspace_id=WORKSPACE,
        ),
        _fact(
            "a2",
            [0.97, 0.12, 0.0],
            scope="project",
            scope_id=PROJECT_A,
            project_id=PROJECT_A,
            workspace_id=WORKSPACE,
        ),
        _fact(
            "b1",
            [0.95, 0.20, 0.0],
            scope="project",
            scope_id=PROJECT_B,
            project_id=PROJECT_B,
            workspace_id=WORKSPACE,
        ),
        _fact(
            "b2",
            [0.94, 0.24, 0.0],
            scope="project",
            scope_id=PROJECT_B,
            project_id=PROJECT_B,
            workspace_id=WORKSPACE,
        ),
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _compatible(left: MemoryChunk, right: MemoryChunk) -> bool:
    left_metadata = left.metadata
    right_metadata = right.metadata
    if left_metadata.get("user_profile_id") != right_metadata.get("user_profile_id"):
        return False
    left_scope = left_metadata.get("scope")
    right_scope = right_metadata.get("scope")
    if left_scope == "global" or right_scope == "global":
        return True
    if left_scope == right_scope:
        return left_metadata.get("scope_id") == right_metadata.get("scope_id")
    if left_scope == "project" and right_scope == "workspace":
        return left_metadata.get("workspace_id") == right_metadata.get("scope_id")
    if left_scope == "workspace" and right_scope == "project":
        return right_metadata.get("workspace_id") == left_metadata.get("scope_id")
    return False


def _oracle(facts: Iterable[MemoryChunk]) -> dict[str, dict[str, float]]:
    records = list(facts)
    result: dict[str, dict[str, float]] = {}
    for source in records:
        candidates = [
            (float(_cosine(source.embedding or [], target.embedding or [])), target.id)
            for target in records
            if target.id != source.id
            and _compatible(source, target)
            and _cosine(source.embedding or [], target.embedding or []) >= THRESHOLD
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        result[source.id] = {target_id: weight for weight, target_id in candidates[:MAX_NEIGHBORS]}
    return result


def _projected_edges(preferences: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    edges: dict[str, dict[str, float]] = {}
    for source_id, targets in preferences.items():
        for target_id, source_weight in targets.items():
            target_weight = preferences.get(target_id, {}).get(source_id)
            if target_weight is None or source_id >= target_id:
                continue
            weight = min(source_weight, target_weight)
            edges.setdefault(source_id, {})[target_id] = weight
            edges.setdefault(target_id, {})[source_id] = weight
    return edges


def _assert_weight_maps(
    actual: dict[str, dict[str, float]], expected: dict[str, dict[str, float]]
) -> None:
    assert actual.keys() == expected.keys()
    for source_id, expected_targets in expected.items():
        actual_targets = actual[source_id]
        assert actual_targets.keys() == expected_targets.keys()
        for target_id, expected_weight in expected_targets.items():
            assert actual_targets[target_id] == pytest.approx(expected_weight, abs=2e-6)


def _config(*, hydration_limit: int = 5000) -> MnemosConfig:
    return MnemosConfig(
        spreading=SpreadingConfig(
            auto_connect_threshold=THRESHOLD,
            max_neighbors_per_node=MAX_NEIGHBORS,
            startup_hydration_limit=hydration_limit,
            startup_auto_connect=False,
        )
    )


def _build_order(
    path: Path, facts: list[MemoryChunk]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        for fact in facts:
            store.store(fact)
            engine.refresh_graph_chunk(fact)
        return store.get_graph_preferences(), store.get_graph_edges()
    finally:
        store.close()


def test_partitioned_sequential_derivation_matches_all_record_oracle(tmp_path: Path) -> None:
    facts = _facts()
    expected_preferences = _oracle(facts)
    expected_edges = _projected_edges(expected_preferences)

    forward_preferences, forward_edges = _build_order(tmp_path / "forward.sqlite", facts)
    reverse_preferences, reverse_edges = _build_order(
        tmp_path / "reverse.sqlite", list(reversed(facts))
    )

    _assert_weight_maps(forward_preferences, expected_preferences)
    _assert_weight_maps(reverse_preferences, expected_preferences)
    _assert_weight_maps(forward_preferences, reverse_preferences)
    _assert_weight_maps(forward_edges, expected_edges)
    _assert_weight_maps(reverse_edges, expected_edges)
    _assert_weight_maps(reverse_edges, forward_edges)
    assert all(len(targets) <= MAX_NEIGHBORS for targets in forward_edges.values())
    assert not forward_edges.get("a1", {}).get("b1")
    assert forward_edges["a1"]["global"] > THRESHOLD
    assert forward_edges["b1"]["global"] > THRESHOLD
    assert forward_edges["a1"]["workspace"] > THRESHOLD
    assert forward_edges["b1"]["workspace"] > THRESHOLD


def test_partitioned_preferences_and_projection_survive_small_restart_hydration(
    tmp_path: Path,
) -> None:
    facts = _facts()
    expected_preferences = _oracle(facts)
    expected_edges = _projected_edges(expected_preferences)
    path = tmp_path / "restart.sqlite"
    _build_order(path, facts)

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(hydration_limit=2),
            embedder=SimpleEmbeddingProvider(dim=3),
            store=store,
        )
        assert store.graph_derivation_status()["ready"] is True
        _assert_weight_maps(store.get_graph_preferences(), expected_preferences)
        _assert_weight_maps(store.get_graph_edges(), expected_edges)
        assert engine.spreading_activation.get_node("b3") is None
    finally:
        store.close()


def test_lost_same_contract_marker_reports_pending_and_bounded_rebuild_cursor(
    tmp_path: Path,
) -> None:
    facts = _facts()
    path = tmp_path / "lost-marker.sqlite"
    _build_order(path, facts)

    store = SQLiteStore(str(path))
    try:
        store._conn.execute("DELETE FROM mnemos_meta WHERE key='graph_preference_sources'")
        store._conn.commit()
    finally:
        store.close()

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        status = store.graph_derivation_status()
        assert status["ready"] is False
        assert status["pending_count"] == len(facts)
        # The same graph contract must repair a missing cursor as well as a
        # missing marker; otherwise recovery has no resumable starting point.
        assert store.graph_rebuild_cursor() == ""
        assert engine.rebuild_graph_batch(limit=2) == 2
        assert store.graph_rebuild_cursor() == "a2"
        # A bounded recovery batch may update at most its requested source
        # rows. Existing preference rows are not evidence that those sources
        # were recomputed in this batch.
        assert store.graph_derivation_status()["marked_sources"] <= 2
        assert store.graph_derivation_status()["pending_count"] <= len(facts)
    finally:
        store.close()


def test_stale_source_fingerprint_reopens_as_pending_and_drains_bounded_batches(
    tmp_path: Path,
) -> None:
    facts = _facts()
    path = tmp_path / "stale-marker.sqlite"
    _build_order(path, facts)

    store = SQLiteStore(str(path))
    try:
        raw = store._conn.execute(
            "SELECT value FROM mnemos_meta WHERE key='graph_preference_sources'"
        ).fetchone()[0]
        marker = json.loads(raw)
        marker["sources"]["a1"] = [999, "stale-source-fingerprint"]
        store._conn.execute(
            "UPDATE mnemos_meta SET value=? WHERE key='graph_preference_sources'",
            (json.dumps(marker, sort_keys=True),),
        )
        store._conn.commit()
    finally:
        store.close()

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        assert store.graph_derivation_status()["ready"] is False
        assert store.graph_derivation_status()["pending_count"] == 1
        assert store.graph_rebuild_cursor() == ""
        for _ in range(len(facts) + 1):
            processed = engine.rebuild_graph_batch(limit=1)
            if processed == 0:
                break
        assert store.graph_derivation_status()["ready"] is True
        assert store.graph_derivation_status()["pending_count"] == 0
    finally:
        store.close()


def test_correction_refills_new_top_k_preferences_against_all_partitions(
    tmp_path: Path,
) -> None:
    facts = [
        fact.model_copy(update={"embedding": [0.0, 1.0, 0.0]}) if fact.id == "a2" else fact
        for fact in _facts()
    ]
    store = SQLiteStore(str(tmp_path / "correction-refill.sqlite"))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        for fact in facts:
            store.store(fact)
            engine.refresh_graph_chunk(fact)
        before = store.get_graph_preferences()
        assert "a2" not in before["global"]
        assert "a2" not in before["workspace"]

        corrected = next(fact for fact in facts if fact.id == "a2").model_copy(
            update={"embedding": [1.0, 0.0, 0.0]}
        )
        store.store(corrected)
        engine.refresh_graph_chunk(corrected)
        expected = _oracle([corrected if fact.id == corrected.id else fact for fact in facts])
        _assert_weight_maps(store.get_graph_preferences(), expected)
        assert "a2" in store.get_graph_preferences()["global"]
        assert "a2" in store.get_graph_preferences()["workspace"]
        assert store.graph_derivation_status()["ready"] is True
    finally:
        store.close()


def test_empty_preference_source_is_refilled_after_restart_when_neighbor_arrives(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-source-restart.sqlite"
    isolated = _fact(
        "isolated",
        [1.0, 0.0, 0.0],
        scope="project",
        scope_id=PROJECT_A,
        project_id=PROJECT_A,
        workspace_id=WORKSPACE,
    )
    arriving = _fact(
        "arriving",
        [1.0, 0.0, 0.0],
        scope="project",
        scope_id=PROJECT_A,
        project_id=PROJECT_A,
        workspace_id=WORKSPACE,
    )
    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        store.store(isolated)
        engine.refresh_graph_chunk(isolated)
        assert store.get_graph_preferences().get("isolated", {}) == {}
        assert store.graph_derivation_status()["ready"] is True
    finally:
        store.close()

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        store.store(arriving)
        engine.refresh_graph_chunk(arriving)
        preferences = store.get_graph_preferences()
        assert "arriving" in preferences["isolated"]
        assert "isolated" in preferences["arriving"]
        edges = store.get_graph_edges()
        assert edges["isolated"]["arriving"] > THRESHOLD
        assert edges["arriving"]["isolated"] > THRESHOLD
    finally:
        store.close()


def test_delete_then_bounded_rebuild_refills_remaining_preferences(
    tmp_path: Path,
) -> None:
    facts = _facts()
    path = tmp_path / "delete-rebuild.sqlite"
    _build_order(path, facts)
    remaining = [fact for fact in facts if fact.id != "a1"]

    store = SQLiteStore(str(path))
    try:
        assert store.delete("a1")
        assert store.graph_derivation_status()["ready"] is False
    finally:
        store.close()

    store = SQLiteStore(str(path))
    try:
        engine = MnemosEngine(
            config=_config(), embedder=SimpleEmbeddingProvider(dim=3), store=store
        )
        for _ in range(len(remaining) + 2):
            if engine.rebuild_graph_batch(limit=2) == 0:
                break
        expected = _oracle(remaining)
        _assert_weight_maps(store.get_graph_preferences(), expected)
        _assert_weight_maps(store.get_graph_edges(), _projected_edges(expected))
        assert store.graph_derivation_status()["ready"] is True
        assert "a1" not in store.get_graph_edges()
    finally:
        store.close()
