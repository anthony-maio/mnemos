"""
mnemos/inspectability.py — Shared memory inspection helpers.
"""

from __future__ import annotations

from typing import Any

from .utils import cosine_similarity
from .types import MemoryChunk

_VALID_SCOPES = {"project", "workspace", "global"}


def _chunk_scope(chunk: MemoryChunk) -> tuple[str, str | None]:
    scope_raw = chunk.metadata.get("scope")
    scope = str(scope_raw).strip().lower() if isinstance(scope_raw, str) else "global"
    if scope not in _VALID_SCOPES:
        scope = "global"
    if scope == "global":
        return scope, None

    scope_id_raw = chunk.metadata.get("scope_id")
    scope_id = str(scope_id_raw).strip() if isinstance(scope_id_raw, str) else "default"
    if not scope_id:
        scope_id = "default"
    return scope, scope_id


def _serialize_cognitive_state(chunk: MemoryChunk) -> dict[str, float] | None:
    if chunk.cognitive_state is None:
        return None
    return {
        "valence": round(chunk.cognitive_state.valence, 3),
        "arousal": round(chunk.cognitive_state.arousal, 3),
        "complexity": round(chunk.cognitive_state.complexity, 3),
    }


def _normalize_scope(scope: str | None, *, default: str = "project") -> str:
    normalized = (scope or default).strip().lower()
    if normalized not in _VALID_SCOPES:
        return default
    return normalized


def _normalize_scope_id(scope: str, scope_id: str | None) -> str | None:
    if scope == "global":
        return None
    if scope_id is None:
        return "default"
    trimmed = scope_id.strip()
    return trimmed if trimmed else "default"


def _normalize_allowed_scopes(
    allowed_scopes: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not allowed_scopes:
        return ("project", "workspace", "global")
    deduped: list[str] = []
    for scope in allowed_scopes:
        normalized = _normalize_scope(scope, default="global")
        if normalized not in deduped:
            deduped.append(normalized)
    return tuple(deduped)


def _build_retrieval_explanation(
    engine: Any,
    chunk: MemoryChunk,
    *,
    query: str,
    current_scope: str,
    scope_id: str | None,
    allowed_scopes: tuple[str, ...],
) -> dict[str, Any]:
    query_embedding = engine.embedder.embed(query)
    semantic_candidates = engine.store.retrieve(query_embedding, top_k=10)
    semantic_ids = [candidate.id for candidate in semantic_candidates]

    scope, chunk_scope_id = _chunk_scope(chunk)
    scope_match = scope in allowed_scopes and (
        scope == "global" or (scope_id is not None and chunk_scope_id == scope_id)
    )

    semantic_similarity = (
        cosine_similarity(query_embedding, chunk.embedding) if chunk.embedding is not None else None
    )

    activated_nodes = engine.spreading_activation.retrieve(query_embedding, top_k=10)
    activation_by_id = {node.id: round(node.energy, 4) for node in activated_nodes}

    explanation: list[str] = []
    if scope_match:
        explanation.append(f"Matched the current {current_scope} retrieval scope.")
    else:
        explanation.append("Did not match the current retrieval scope cleanly.")
    if chunk.id in semantic_ids:
        explanation.append("Appeared in semantic candidates for this query.")
    if chunk.id in activation_by_id:
        explanation.append("Received spreading activation from an associative neighbor.")
    if semantic_similarity is not None and semantic_similarity >= 0.7:
        explanation.append("Query embedding is strongly similar to this memory.")

    return {
        "query": query,
        "scope_match": scope_match,
        "in_semantic_candidates": chunk.id in semantic_ids,
        "semantic_rank": (semantic_ids.index(chunk.id) + 1 if chunk.id in semantic_ids else None),
        "semantic_similarity": (
            None if semantic_similarity is None else round(float(semantic_similarity), 4)
        ),
        "graph_activated": chunk.id in activation_by_id,
        "graph_energy": activation_by_id.get(chunk.id),
        "allowed_scopes": list(allowed_scopes),
        "current_scope": current_scope,
        "scope_id": scope_id,
        "explanation": explanation,
    }


def build_chunk_inspection(
    engine: Any,
    chunk_id: str,
    *,
    query: str | None = None,
    current_scope: str = "project",
    scope_id: str | None = None,
    allowed_scopes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    chunk = engine.store.get(chunk_id)
    if chunk is None:
        return None

    scope, chunk_scope_id = _chunk_scope(chunk)
    node = engine.spreading_activation.get_node(chunk.id)
    history = chunk.metadata.get("revision_history", [])
    if not isinstance(history, list):
        history = []

    neighbors: list[dict[str, Any]] = []
    if node is not None:
        neighbor_pairs = sorted(node.neighbors.items(), key=lambda item: item[1], reverse=True)[:5]
        for neighbor_id, weight in neighbor_pairs:
            neighbor_chunk = engine.store.get(neighbor_id)
            neighbors.append(
                {
                    "id": neighbor_id,
                    "weight": round(weight, 4),
                    "content_preview": (
                        None if neighbor_chunk is None else neighbor_chunk.content[:120]
                    ),
                }
            )

    retrieval = None
    if query is not None and query.strip():
        normalized_current_scope = _normalize_scope(current_scope)
        normalized_scope_id = _normalize_scope_id(normalized_current_scope, scope_id)
        retrieval = _build_retrieval_explanation(
            engine,
            chunk,
            query=query.strip(),
            current_scope=normalized_current_scope,
            scope_id=normalized_scope_id,
            allowed_scopes=_normalize_allowed_scopes(allowed_scopes),
        )

    return {
        "id": chunk.id,
        "content": chunk.content,
        "scope": scope,
        "scope_id": chunk_scope_id,
        "salience": round(chunk.salience, 4),
        "version": chunk.version,
        "access_count": chunk.access_count,
        "created_at": chunk.created_at.isoformat(),
        "updated_at": chunk.updated_at.isoformat(),
        "cognitive_state": _serialize_cognitive_state(chunk),
        "provenance": {
            "stored_by": chunk.metadata.get("source", "unknown"),
            "ingest_channel": chunk.metadata.get("ingest_channel"),
            "encoding_reason": chunk.metadata.get("encoding_reason"),
            "original_timestamp": chunk.metadata.get("original_timestamp"),
            "reconsolidated_at": chunk.metadata.get("reconsolidated_at"),
        },
        "history": history,
        "graph": {
            "present": node is not None,
            "neighbor_count": 0 if node is None else len(node.neighbors),
            "neighbors": neighbors,
        },
        "retrieval": retrieval,
        "metadata": chunk.metadata,
    }
