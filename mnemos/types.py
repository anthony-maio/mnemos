"""
mnemos/types.py — Shared domain types for the Mnemos memory system.

Neuroscience inspiration:
- MemoryChunk maps to a memory engram: a discrete, physically stored trace.
- CognitiveState mirrors the amygdala's tagging system: valence (positive/negative),
  arousal (calm/urgent), and complexity (simple/complex load).
- Interactions are the raw episodic inputs processed by the hippocampus before
  consolidation to semantic long-term memory.
- The 'version' field on MemoryChunk tracks reconsolidation events, just as
  each recall-and-rewrite cycle in human memory can alter the stored trace.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class CognitiveState(BaseModel):
    """
    A three-dimensional representation of emotional/cognitive context.

    Inspired by the dimensional theory of emotion (Russell, 1980) and the
    amygdala's role in tagging memories with affective significance:
      - valence: how positive or negative the interaction feels (-1.0 to 1.0)
      - arousal: urgency/intensity of the interaction (0.0 = calm, 1.0 = urgent)
      - complexity: cognitive load required (0.0 = simple, 1.0 = complex)

    State-dependent memory retrieval (Bower, 1981) shows that memories encoded
    in a given state are more readily recalled in a matching state — this is
    the neuroscientific basis for AffectiveRouter's scoring formula.
    """

    valence: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Emotional valence: -1.0 (very negative) to 1.0 (very positive).",
    )
    arousal: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Arousal/urgency level: 0.0 (calm) to 1.0 (highly urgent).",
    )
    complexity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cognitive complexity: 0.0 (simple) to 1.0 (highly complex).",
    )

    def distance(self, other: "CognitiveState") -> float:
        """
        Compute Euclidean distance between two cognitive states, normalized to [0, 1].

        The raw 3D Euclidean distance has a maximum of sqrt((2^2 + 1^2 + 1^2)) = sqrt(6)
        when valence spans [-1, 1] and arousal/complexity span [0, 1].
        We normalize by this maximum to get a value in [0, 1].

        Args:
            other: Another CognitiveState to compare against.

        Returns:
            A float in [0, 1] where 0.0 = identical states, 1.0 = maximally different.
        """
        max_distance = math.sqrt(4.0 + 1.0 + 1.0)  # sqrt(6) ≈ 2.449
        raw = math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.complexity - other.complexity) ** 2
        )
        return min(raw / max_distance, 1.0)

    model_config = {"frozen": False}


class MemoryChunk(BaseModel):
    """
    A discrete unit of stored memory — analogous to a memory engram.

    In neuroscience, an engram is the physical substrate of a memory: a set of
    neurons whose synaptic weights have been modified by a learning experience.
    MemoryChunk represents the AI equivalent: a piece of content with associated
    metadata about *how* and *when* it was learned, and its *emotional context*.

    The 'version' field tracks reconsolidation — each time a chunk is recalled
    and rewritten with new context (Mutable RAG), the version increments, mirroring
    how human memory is physically altered by each act of recall.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this memory chunk.",
    )
    content: str = Field(description="The text content of this memory.")
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of content for similarity search.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (source, tags, etc.).",
    )
    salience: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Salience weight from surprisal gate (0=mundane, 1=highly surprising).",
    )
    cognitive_state: CognitiveState | None = Field(
        default=None,
        description="The emotional/cognitive state at encoding time (amygdala tag).",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of initial creation.",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last modification.",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this chunk has been retrieved.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Reconsolidation version counter; increments on each rewrite.",
    )

    model_config = {"frozen": False}

    def touch(self) -> None:
        """Increment access count and update timestamp (simulates memory reactivation)."""
        self.access_count += 1
        self.updated_at = _utcnow()

    def reconsolidate(self, new_content: str) -> "MemoryChunk":
        """
        Create a new version of this chunk with updated content.

        Mirrors biological memory reconsolidation: the act of recalling a memory
        destabilizes it, allowing new information to be integrated before the
        trace is re-stabilized (restabilization/re-consolidation).

        Args:
            new_content: The updated content after reconsolidation.

        Returns:
            A new MemoryChunk with incremented version and updated content.
        """
        return MemoryChunk(
            id=self.id,  # Same ID — same memory, new version
            content=new_content,
            embedding=None,  # Will need re-embedding
            metadata={**self.metadata, "previous_content": self.content},
            salience=self.salience,
            cognitive_state=self.cognitive_state,
            created_at=self.created_at,
            updated_at=_utcnow(),
            access_count=self.access_count + 1,
            version=self.version + 1,
        )


class Interaction(BaseModel):
    """
    A single turn in the conversation — the raw episodic input.

    Analogous to a hippocampal episodic memory: a time-stamped record of
    what was experienced, before consolidation into semantic long-term memory.
    """

    role: str = Field(
        description="Speaker role: 'user', 'assistant', or 'system'.",
    )
    content: str = Field(description="The text content of this interaction.")
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the interaction.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (session_id, user_id, etc.).",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is a non-empty string."""
        if not v.strip():
            raise ValueError("Role must not be empty.")
        return v.lower().strip()


class ProcessResult(BaseModel):
    """
    The outcome of processing an interaction through the memory pipeline.

    Captures whether the interaction was encoded (stored), the resulting chunk
    if stored, the salience score that drove the decision, and a human-readable
    reason explaining the memory gate's decision.
    """

    stored: bool = Field(
        description="Whether this interaction was committed to long-term memory.",
    )
    chunk: MemoryChunk | None = Field(
        default=None,
        description="The created/updated MemoryChunk if stored is True.",
    )
    salience: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Computed salience/surprisal score for this interaction.",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the storage decision.",
    )


class ActivationNode(BaseModel):
    """
    A node in the spreading activation graph.

    Inspired by Collins & Loftus (1975) spreading activation theory: concepts
    in semantic memory are nodes in a network, connected by typed edges. Activation
    energy spreads along edges, decaying with distance, creating a 'spotlight' of
    primed context around any retrieved concept.

    The 'energy' field represents current activation level — nodes above a threshold
    are considered 'primed' and included in the LLM's context.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique node identifier.",
    )
    content: str = Field(description="The concept/memory content at this node.")
    energy: float = Field(
        default=0.0,
        ge=0.0,
        description="Current activation energy (≥ 0). Decays over time.",
    )
    neighbors: dict[str, float] = Field(
        default_factory=dict,
        description="Maps neighbor node_id → edge weight (0.0 to 1.0).",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding for initial similarity-based lookup.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Scoped metadata used for graph-safe connectivity decisions.",
    )

    model_config = {"frozen": False}


class ConsolidationResult(BaseModel):
    """
    The output of a sleep consolidation cycle.

    Mirrors the hippocampal-neocortical transfer that occurs during sleep:
    - Raw episodic memories (conversations) are replayed and distilled
    - Permanent semantic facts are extracted and stored in long-term memory
    - The raw episodes are pruned (forgotten) to save capacity
    - Repeated procedural patterns may be crystallized into executable tools
    """

    facts_extracted: list[str] = Field(
        default_factory=list,
        description="Permanent facts/preferences extracted from episodic buffer.",
    )
    chunks_pruned: int = Field(
        default=0,
        ge=0,
        description="Number of raw episodic chunks deleted after consolidation.",
    )
    tools_generated: list[str] = Field(
        default_factory=list,
        description=(
            "Python tool code strings generated via proceduralization. "
            "SECURITY WARNING: These are UNTRUSTED LLM-generated code strings. "
            "They MUST NOT be executed without thorough human review."
        ),
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Wall-clock time the consolidation cycle took.",
    )
