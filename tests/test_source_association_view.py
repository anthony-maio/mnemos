from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from benchmarks.local_memory.source_association_view import relation_view
from mnemos.config import MnemosConfig, ModuleConfig, SpreadingConfig
from mnemos.engine import MnemosEngine
from mnemos.jobs import FactEvidence
from mnemos.retrieval import RetrievalConfig
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.storage import SQLiteStore


class FixedEmbedder(EmbeddingProvider):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    @property
    def dim(self) -> int:
        return len(next(iter(self.vectors.values())))

    def embed(self, text: str) -> list[float]:
        return list(self.vectors[text])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _event(event_id: str, *, profile: str = "profile-a") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role="user",
        content=f"source {event_id}",
        normalized_content=f"source {event_id}",
        scope="project",
        scope_id="project-a",
        source="manual",
        metadata={
            "user_profile_id": profile,
            "project_id": "project-a",
            "workspace_id": "workspace-a",
        },
    )


def _chunk(chunk_id: str, vector: list[float], *, profile: str = "profile-a") -> MemoryChunk:
    return MemoryChunk(
        id=chunk_id,
        content=chunk_id,
        embedding=vector,
        metadata={
            "scope": "project",
            "scope_id": "project-a",
            "user_profile_id": profile,
            "project_id": "project-a",
            "workspace_id": "workspace-a",
            "assertion_status": "asserted",
            "source": "explicit_remember",
        },
    )


def _add_evidence(store: SQLiteStore, fact_id: str, event_id: str) -> None:
    evidence = FactEvidence(
        evidence_id=f"evidence-{fact_id}-{event_id}",
        fact_id=fact_id,
        source_event_id=event_id,
        excerpt=fact_id,
    )
    store._conn.execute(
        "INSERT INTO fact_evidence "
        "(evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            evidence.evidence_id,
            evidence.fact_id,
            evidence.source_event_id,
            evidence.excerpt,
            evidence.source_ref,
            evidence.created_at.isoformat(),
        ),
    )


def _engine(tmp_path: Path) -> tuple[MnemosEngine, SQLiteStore]:
    path = tmp_path / "source-view.sqlite"
    store = SQLiteStore(str(path))
    embedder = FixedEmbedder(
        {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0], "query": [1.0, 0.0]}
    )
    config = MnemosConfig(
        modules=ModuleConfig(affective=False, spreading=True),
        retrieval=RetrievalConfig(
            profile="biomimetic",
            affective_enabled=False,
            spreading_enabled=True,
            relevance_floor=0.0,
            max_candidates=20,
        ),
        spreading=SpreadingConfig(
            auto_connect_threshold=0.9,
            max_neighbors_per_node=4,
        ),
    )
    engine = MnemosEngine(config=config, embedder=embedder, store=store)
    for chunk in (
        _chunk("a", [1.0, 0.0]),
        _chunk("b", [0.0, 1.0]),
        _chunk("c", [1.0, 0.0]),
    ):
        store.store(chunk)
        engine.refresh_graph_chunk(chunk)
    for event_id in ("source-ab", "source-c"):
        store.append_episodic_event(_event(event_id))
    _add_evidence(store, "a", "source-ab")
    _add_evidence(store, "b", "source-ab")
    _add_evidence(store, "c", "source-c")
    store._conn.commit()
    return engine, store


def _all(_: MemoryChunk) -> bool:
    return True


setattr(
    _all,
    "_mnemos_scope_filter",
    {
        "user_profile_id": "profile-a",
        "project_id": "project-a",
        "workspace_id": "workspace-a",
        "allowed_scopes": ("project",),
    },
)


def _provenance(view: Any, first_id: str, second_id: str) -> Mapping[str, Any]:
    value = view.provenance_for(first_id, second_id)
    assert value is not None
    return cast(Mapping[str, Any], value)


def test_source_only_excludes_semantic_edge_and_combined_keeps_provenance(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        store._conn.executemany(
            "INSERT OR REPLACE INTO memory_edges "
            "(source_id, target_id, weight, edge_type, updated_at) VALUES (?, ?, ?, ?, ?)",
            [
                ("a", "b", 0.4, "SEMANTIC_SIMILAR", "2026-01-01T00:00:00+00:00"),
                ("b", "a", 0.4, "SEMANTIC_SIMILAR", "2026-01-01T00:00:00+00:00"),
            ],
        )
        store._conn.commit()
        with relation_view(engine, arm="source_association") as source:
            assert [chunk.id for chunk, _ in source.get_graph_neighbors("a", filter_fn=_all)] == [
                "b"
            ]
            assert _provenance(source, "a", "b")["components"] == ["source_association"]
            relation = _provenance(source, "a", "b")["source_relations"][0]
            assert relation["source_event_id"] == "source-ab"
            assert relation["fact_versions"] == {"a": 1, "b": 1}
            assert relation["evidence_ids"] == ["evidence-a-source-ab", "evidence-b-source-ab"]
            assert relation["source_role"] == "user"
            assert source.diagnostic["summary"]["relation_count"] == 1
            assert source.provenance_for("a", "c") is None

        with relation_view(engine, arm="combined") as combined:
            assert [chunk.id for chunk, _ in combined.get_graph_neighbors("a", filter_fn=_all)] == [
                "b",
                "c",
            ]
            assert _provenance(combined, "a", "b")["components"] == [
                "semantic",
                "source_association",
            ]
            assert _provenance(combined, "a", "c")["components"] == ["semantic"]
            assert _provenance(combined, "a", "b")["effective_weight"] == pytest.approx(1.0)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_semantic_arm_preserves_engine_results_and_scores(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        baseline = await engine.retrieve(
            "query", top_k=3, reconsolidate=False, current_scope="project", scope_id="project-a"
        )
        with relation_view(engine, arm="semantic") as semantic:
            viewed = await engine.retrieve(
                "query", top_k=3, reconsolidate=False, current_scope="project", scope_id="project-a"
            )
            assert [chunk.id for chunk in viewed] == [chunk.id for chunk in baseline]
            neighbors = semantic.get_graph_neighbors("a", filter_fn=_all)
            assert [chunk.id for chunk, _ in neighbors] == ["c"]
            assert _provenance(semantic, "a", "c")["effective_weight"] == pytest.approx(1.0)
    finally:
        store.close()


def test_foreign_source_relation_is_rejected_by_structural_diagnostic(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        store.append_episodic_event(_event("source-foreign", profile="profile-foreign"))
        foreign = _chunk("foreign", [0.0, 1.0], profile="profile-foreign")
        store.store(foreign)
        _add_evidence(store, "a", "source-foreign")
        _add_evidence(store, "foreign", "source-foreign")
        store._conn.commit()
        with relation_view(engine, arm="source_association") as source:
            assert all(
                chunk.id != "foreign"
                for chunk, _ in source.get_graph_neighbors("a", filter_fn=_all)
            )
            assert source.provenance_for("a", "foreign") is None
    finally:
        store.close()


def test_context_restores_activation_neighbors_even_on_exception(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        before = {
            node.id: dict(node.neighbors) for node in engine.spreading_activation.get_all_nodes()
        }
        with pytest.raises(RuntimeError, match="stop"):
            with relation_view(engine, arm="source_association"):
                node = engine.spreading_activation.get_node("a")
                assert node is not None
                assert node.neighbors == {}
                raise RuntimeError("stop")
        assert {
            node.id: dict(node.neighbors) for node in engine.spreading_activation.get_all_nodes()
        } == before
        assert "get_graph_neighbors" not in store.__dict__
    finally:
        store.close()


def test_context_removes_nodes_added_during_relation_view_use(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        store.append_episodic_event(_event("source-ad"))
        new_chunk = _chunk("d", [0.0, 1.0])
        store.store(new_chunk)
        _add_evidence(store, "a", "source-ad")
        _add_evidence(store, "d", "source-ad")
        store._conn.commit()
        engine.refresh_graph_chunk(new_chunk)
        spreading = engine.spreading_activation
        getattr(spreading, "_nodes").pop("d", None)
        getattr(spreading, "_authoritative").pop("d", None)
        assert spreading.get_node("d") is None

        with relation_view(engine, arm="source_association"):
            spreading.add_node_from_chunk(new_chunk)
            assert spreading.get_node("d") is not None
        assert spreading.get_node("d") is None
    finally:
        store.close()


def test_new_view_reads_correction_and_forget_changes(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        with relation_view(engine, arm="source_association") as before:
            assert [chunk.id for chunk, _ in before.get_graph_neighbors("a", filter_fn=_all)] == [
                "b"
            ]

        corrected = _chunk("a", [1.0, 0.0]).model_copy(
            update={"content": "a-corrected", "version": 2}
        )
        store.upsert_canonical(corrected)
        store._conn.execute(
            "UPDATE fact_evidence SET excerpt=? WHERE fact_id=? AND source_event_id=?",
            ("a-corrected", "a", "source-ab"),
        )
        store._conn.commit()
        with relation_view(engine, arm="source_association") as corrected_view:
            assert [
                chunk.id for chunk, _ in corrected_view.get_graph_neighbors("a", filter_fn=_all)
            ] == ["b"]

        assert store.forget_fact("b", expected_version=1)
        with relation_view(engine, arm="source_association") as after_forget:
            assert after_forget.get_graph_neighbors("a", filter_fn=_all) == []
    finally:
        store.close()


def test_cut_removes_only_active_incident_edges_and_preserves_persisted_graph(
    tmp_path: Path,
) -> None:
    engine, store = _engine(tmp_path)
    try:
        before_edges = store.get_graph_edges()
        direct_before = [chunk.id for chunk in store.retrieve([1.0, 0.0], top_k=10)]
        with relation_view(engine, arm="combined", cut_fact_id="c") as cut:
            neighbors = [chunk.id for chunk, _ in cut.get_graph_neighbors("a", filter_fn=_all)]
            assert neighbors == ["b"]
            assert cut.provenance_for("a", "c") is None
            assert ("a", "c") in cut.excluded_pairs
            assert [chunk.id for chunk in store.retrieve([1.0, 0.0], top_k=10)] == direct_before
        assert store.get_graph_edges() == before_edges
    finally:
        store.close()


def test_invalid_arm_weight_and_fanout_fail_closed(tmp_path: Path) -> None:
    engine, store = _engine(tmp_path)
    try:
        with pytest.raises(ValueError):
            with relation_view(engine, arm="other"):  # type: ignore[arg-type]
                pass
        with pytest.raises(ValueError):
            with relation_view(engine, arm="source_association", source_edge_weight=0.0):
                pass
        with pytest.raises(ValueError):
            with relation_view(engine, arm="source_association", source_edge_weight=1.01):
                pass
        with pytest.raises(ValueError):
            with relation_view(engine, arm="source_association", fanout_limit=1):
                pass
    finally:
        store.close()
