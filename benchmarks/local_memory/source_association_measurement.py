"""Production retrieval observations for the frozen association experiment.

This module does not contact a provider. Fixture extraction and relation views
are development controls, and observations are not quality acceptance claims.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from mnemos.config import MnemosConfig, ModuleConfig, SpreadingConfig
from mnemos.engine import MnemosEngine
from mnemos.retrieval import RetrievalConfig, context_token_upper_bound, score_candidate
from mnemos.scopes import ProjectIdentity, ScopeContext, ScopeRegistry, WorkspaceIdentity
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import cosine_similarity
from mnemos.utils.storage import SQLiteStore

from .graph_measurement import _independent_activation, _paths, _scope_predicate
from .module_mechanism import _FrozenEmbedder
from .source_association_view import RelationArm, relation_view


def open_measurement_engine(
    database: Path,
    case: Mapping[str, Any],
    vectors: Mapping[str, Sequence[float]],
    options: Mapping[str, Any],
) -> tuple[MnemosEngine, SQLiteStore]:
    """Open a disposable populated fixture and finish normal graph derivation."""
    if not database.is_file():
        raise FileNotFoundError(database)
    context = ScopeContext.model_validate(case["context"])
    if context.project_id is None or context.workspace_id is None:
        raise ValueError("association fixtures require project and workspace context")
    registry = ScopeRegistry(
        profiles=[context.user_profile_id],
        projects={
            context.project_id: ProjectIdentity(
                id=context.project_id,
                root=str(database.parent),
                user_profile_id=context.user_profile_id,
                enabled=True,
                workspace_ids=[context.workspace_id],
            )
        },
        workspaces={
            context.workspace_id: WorkspaceIdentity(
                id=context.workspace_id,
                name=context.workspace_id,
                user_profile_id=context.user_profile_id,
            )
        },
    )
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False, mutable_rag=False, affective=False, sleep=False, spreading=True
        ),
        retrieval=RetrievalConfig(
            profile="biomimetic",
            affective_enabled=False,
            spreading_enabled=True,
            relevance_floor=options["relevance_floor"],
            graph_weight=options["graph_weight"],
            max_candidates=options["max_candidates"],
        ),
        spreading=SpreadingConfig(
            auto_connect_threshold=options["semantic_edge_threshold"],
            max_neighbors_per_node=options["neighbor_budget"],
            activation_threshold=options["activation_threshold"],
            decay_rate=options["decay"],
            max_hops=options["max_hops"],
            max_visited=options["max_visited"],
        ),
    )
    embedder = _FrozenEmbedder(vectors, query_vectors=vectors, document_vectors=vectors)
    store = SQLiteStore(str(database))
    try:
        engine = MnemosEngine(config, embedder=embedder, store=store, scope_registry=registry)
        # Use the normal pending-source rebuild lifecycle. Never force readiness.
        for _ in range(100):
            if store.graph_derivation_status().get("ready") is True:
                break
            engine.rebuild_graph_batch(limit=100)
        else:
            raise RuntimeError("fixture semantic graph did not become ready")
        return engine, store
    except BaseException:
        store.close()
        raise


async def observe_retrieval(
    engine: MnemosEngine,
    store: SQLiteStore,
    case: Mapping[str, Any],
    vectors: Mapping[str, Sequence[float]],
    options: Mapping[str, Any],
    *,
    arm: RelationArm | None,
    cut_fact_id: str | None = None,
) -> dict[str, Any]:
    """Observe one call and retain every candidate's score and exclusion reason.

    ``arm=None`` uses the untouched production semantic path for parity checks.
    The caller owns separate stores for restart/order/lifecycle controls.
    """
    context = ScopeContext.model_validate(case["context"])
    predicate = _scope_predicate(context)
    query = str(case["query"])
    query_vector = list(vectors[query])
    pool_limit = min(max(int(options["top_k"]) * 6, 20), int(options["max_candidates"]))
    baseline = store.retrieve(query_vector, top_k=pool_limit, filter_fn=predicate)
    lexical = store.retrieve_lexical(query, top_k=pool_limit, filter_fn=predicate)
    vector_ids = [chunk.id for chunk in baseline]
    lexical_ids = [chunk.id for chunk in lexical]
    direct_ids = set(vector_ids) | set(lexical_ids)
    chunks = {chunk.id: chunk for chunk in store.get_all()}
    allowed_ids = {key for key, chunk in chunks.items() if predicate(chunk)}
    seed = (
        min(
            vector_ids,
            key=lambda key: (-cosine_similarity(query_vector, chunks[key].embedding or []), key),
        )
        if vector_ids
        else None
    )
    before_edges = store.get_graph_edges()
    selected_view = (
        relation_view(
            engine,
            arm=arm,
            source_edge_weight=options["source_edge_weight"],
            fanout_limit=options["fanout_limit"],
            cut_fact_id=cut_fact_id,
        )
        if arm is not None
        else nullcontext(None)
    )
    with selected_view as view:
        results = await engine.retrieve(
            query,
            top_k=int(options["top_k"]),
            max_context_tokens=int(options["context_budget_tokens"]),
            context=context,
            current_scope="project",
            allowed_scopes=("project", "workspace", "global"),
            profile="biomimetic",
        )
        # This repeats only deterministic traversal to expose all candidate
        # energies. It does not issue another retrieval or touch the provider.
        activated = engine.spreading_activation.retrieve(
            query_vector,
            top_k=pool_limit,
            seed_ids=vector_ids,
            filter_fn=lambda node: predicate(
                MemoryChunk(id=node.id, content=node.content, metadata=node.metadata)
            ),
            max_visited=options["max_visited"],
        )
        energy = {node.id: node.energy for node in activated}
        effective: dict[str, dict[str, float]] = {}
        provenance: list[dict[str, Any]] = []
        if view is None:
            effective = before_edges
        else:
            provenance = [dict(value) for _, value in sorted(view.provenance.items())]
            for item in provenance:
                left, right = item["pair"]
                weight = float(item["effective_weight"])
                effective.setdefault(left, {})[right] = weight
                effective.setdefault(right, {})[left] = weight
        actual_all_energy = (
            engine.spreading_activation.activate(
                seed, filter_fn=lambda node: node.id in allowed_ids
            )
            if seed is not None
            else {}
        )
        expected_energy = (
            _independent_activation(effective, seed, allowed_ids, engine.config.spreading)
            if seed is not None
            else {}
        )
        candidate_ids = direct_ids | set(energy)
        scores = {
            key: score_candidate(
                chunks[key],
                query_vector,
                current_state=None,
                affective=False,
                graph_energy=energy.get(key, 0.0),
                graph_weight=options["graph_weight"],
            )
            for key in candidate_ids
        }
        ranked = sorted(
            (key for key in candidate_ids if scores[key].semantic >= options["relevance_floor"]),
            key=lambda key: (
                -scores[key].total,
                chunks[key].metadata.get("scope") != "project",
                key,
            ),
        )
        budget = int(options["context_budget_tokens"])
        reconstructed: list[str] = []
        for key in ranked:
            cost = context_token_upper_bound(chunks[key])
            if cost <= budget:
                reconstructed.append(key)
                budget -= cost
                if len(reconstructed) == options["top_k"]:
                    break
        target = str(case["target_id"])
        structural = (
            _paths(effective, seed, target, allowed_ids, engine.config.spreading, min_energy=0)
            if seed is not None
            else []
        )
        activation_paths = (
            _paths(effective, seed, target, allowed_ids, engine.config.spreading)
            if seed is not None
            else []
        )
        returned_ids = [chunk.id for chunk in results]
        observations = {
            "case_id": case["id"],
            "arm": arm or "untouched_semantic_control",
            "query": query,
            "candidate_pool": {
                "vector_ids": vector_ids,
                "lexical_ids": lexical_ids,
                "union_ids": sorted(direct_ids),
                "actual_selected_seed_id": seed,
            },
            "edges": effective,
            "typed_provenance": provenance,
            "source_diagnostic": dict(view.diagnostic) if view is not None else None,
            "candidate_scores": [
                {
                    "id": key,
                    "version": chunks[key].version,
                    **scores[key].model_dump(),
                    "rank": ranked.index(key) + 1 if key in ranked else None,
                    "selected": key in returned_ids,
                    "status": "below_relevance_floor" if key not in ranked else "scored",
                    "direct_candidate": key in direct_ids,
                }
                for key in sorted(candidate_ids)
            ],
            "returned_ids": returned_ids,
            "returned_traces": {chunk.id: chunk.metadata["retrieval_trace"] for chunk in results},
            "activation": actual_all_energy,
            "independent_activation": expected_energy,
            "target": {
                "id": target,
                "direct_candidate": target in direct_ids,
                "structural_path": bool(structural),
                "activation_path": bool(activation_paths),
                "structural_paths": structural,
                "activation_paths": activation_paths,
                "energy": actual_all_energy.get(target, 0.0),
                "rank": ranked.index(target) + 1 if target in ranked else None,
                "in_top_k": target in returned_ids,
            },
            "controls": {
                "scope_pass": candidate_ids <= allowed_ids
                and set(actual_all_energy) <= allowed_ids,
                "score_budget_parity": reconstructed == returned_ids,
                "activation_oracle_pass": actual_all_energy == expected_energy,
            },
        }
    observations["controls"]["persisted_edges_unchanged"] = store.get_graph_edges() == before_edges
    return observations
