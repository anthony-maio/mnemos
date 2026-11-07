"""Retrieval settings shared by clients without importing retrieval models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    profile: Literal["baseline", "biomimetic"] = "baseline"
    affective_enabled: bool = True
    spreading_enabled: bool = True
    relevance_floor: float = Field(default=0.2, ge=0, le=1)
    graph_weight: float = Field(default=0.1, ge=0, le=1)
    max_candidates: int = Field(default=100, ge=1, le=1000)
    # Where the biomimetic profile gets the query's cognitive state.
    # classify: ask the affective classifier per query (today's behaviour).
    # recent_state: the partition's last encoded state, no model call; the
    # runtime selects this when the judge is hosted, so recall never waits
    # on a round trip. none: no state match.
    query_state_source: Literal["classify", "recent_state", "none"] = "classify"
