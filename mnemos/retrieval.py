"""One scoring and context-budget contract for memory retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .types import CognitiveState, MemoryChunk
from .utils.embeddings import cosine_similarity


class RetrievalConfig(BaseModel):
    profile: Literal["baseline", "biomimetic"] = "baseline"
    affective_enabled: bool = True
    spreading_enabled: bool = True
    relevance_floor: float = Field(default=0.2, ge=0, le=1)
    graph_weight: float = Field(default=0.1, ge=0, le=1)
    max_candidates: int = Field(default=100, ge=1, le=1000)


class ScoreBreakdown(BaseModel):
    semantic: float
    state_match: float | None = None
    graph_energy: float = 0
    graph_contribution: float = 0
    total: float
    state_status: str


def score_candidate(
    chunk: MemoryChunk,
    query_embedding: list[float],
    *,
    current_state: CognitiveState | None,
    affective: bool,
    similarity_weight: float = 0.7,
    state_weight: float = 0.3,
    graph_energy: float = 0,
    graph_weight: float = 0,
) -> ScoreBreakdown:
    similarity = 0.0
    if chunk.embedding and len(chunk.embedding) == len(query_embedding):
        similarity = max(0.0, cosine_similarity(query_embedding, chunk.embedding))
    match = None
    status = "disabled" if not affective else "unavailable"
    score = similarity
    if affective and current_state is not None and chunk.cognitive_state is not None:
        match = 1.0 - chunk.cognitive_state.distance(current_state)
        score = similarity * similarity_weight + match * state_weight
        status = "measured"
    graph_contribution = graph_weight * graph_energy
    return ScoreBreakdown(
        semantic=similarity,
        state_match=match,
        graph_energy=graph_energy,
        graph_contribution=graph_contribution,
        total=score + graph_contribution,
        state_status=status,
    )


def assertion_status(chunk: MemoryChunk) -> Literal["asserted", "conflict", "unknown"]:
    status = chunk.metadata.get("assertion_status")
    if status == "asserted":
        return "asserted"
    if status == "conflict":
        return "conflict"
    return "unknown"


def render_context_chunk(chunk: MemoryChunk) -> str:
    qualifier = "; unresolved conflict" if assertion_status(chunk) == "conflict" else ""
    return f"[{chunk.id} version {chunk.version}{qualifier}] {chunk.content}\n"


def context_token_upper_bound(chunk: MemoryChunk) -> int:
    """A conservative UTF-8 byte bound until the host provides its tokenizer."""
    return len(render_context_chunk(chunk).encode("utf-8"))
