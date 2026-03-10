"""
mnemos/inspectability.py — Shared memory inspection helpers.
"""

from __future__ import annotations

from typing import Any

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


def build_chunk_inspection(engine: Any, chunk_id: str) -> dict[str, Any] | None:
    chunk = engine.store.get(chunk_id)
    if chunk is None:
        return None

    scope, scope_id = _chunk_scope(chunk)
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

    return {
        "id": chunk.id,
        "content": chunk.content,
        "scope": scope,
        "scope_id": scope_id,
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
        "metadata": chunk.metadata,
    }
