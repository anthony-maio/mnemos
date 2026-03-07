"""
mnemos/config.py — Hierarchical configuration for the Mnemos memory system.

All configuration is expressed as Pydantic BaseModels so it can be:
- Validated at startup
- Serialized to/from JSON/YAML for persistence
- Overridden at any level (global → module-specific)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SurprisalConfig(BaseModel):
    """
    Configuration for the SurprisalGate module.

    The surprisal threshold mirrors the brain's predictive coding mechanism:
    only prediction errors above a certain magnitude trigger memory encoding.
    Lower thresholds = more sensitive gate (stores more).
    Higher thresholds = stricter gate (only stores truly surprising content).
    """

    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine distance threshold for encoding. Interactions with surprisal "
            "score above this are stored. Default 0.3 corresponds to ~72° separation "
            "in embedding space — a meaningful semantic shift."
        ),
    )
    prediction_model: str = Field(
        default="llama3",
        description="LLM model name used for next-intent prediction.",
    )
    embedding_dim: int = Field(
        default=384,
        gt=0,
        description="Dimensionality of embeddings used for surprisal calculation.",
    )
    history_window: int = Field(
        default=10,
        gt=0,
        description="Number of recent turns to include in prediction context.",
    )
    min_content_length: int = Field(
        default=10,
        ge=0,
        description="Minimum character length of content to consider for encoding.",
    )


class MutableRAGConfig(BaseModel):
    """
    Configuration for the MutableRAG (Memory Reconsolidation) module.

    Reconsolidation is triggered asynchronously after retrieval. The LLM
    is asked whether retrieved facts have changed in light of new context,
    mirroring the destabilization-restabilization cycle of biological memory.
    """

    enabled: bool = Field(
        default=True,
        description="Whether to run reconsolidation after retrieval.",
    )
    staleness_check_prompt: str = Field(
        default=(
            "You are a memory maintenance agent. A user's stored memory says:\n"
            '"{content}"\n\n'
            "New context from the conversation:\n"
            '"{context}"\n\n'
            "Has this stored memory become outdated, contradicted, or significantly "
            "changed based on the new context? Reply with exactly one of:\n"
            "- UNCHANGED: if the memory is still accurate\n"
            "- CHANGED: <updated version of the memory>\n\n"
            "Be conservative: only mark CHANGED if there is a clear contradiction "
            "or significant update."
        ),
        description="Prompt template for reconsolidation LLM check.",
    )
    max_labile_chunks: int = Field(
        default=20,
        gt=0,
        description="Maximum number of chunks to reconsolidate in a single pass.",
    )
    reconsolidation_cooldown_seconds: int = Field(
        default=60,
        ge=0,
        description="Minimum seconds between reconsolidation runs for the same chunk.",
    )


class AffectiveConfig(BaseModel):
    """
    Configuration for the AffectiveRouter module.

    The retrieval scoring formula weights semantic similarity against affective
    state match. The defaults (0.7/0.3) prioritize content relevance while
    giving meaningful influence to emotional context — matching empirical findings
    from mood-congruent memory research (Bower, 1981).
    """

    weight_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight given to cosine similarity in retrieval scoring.",
    )
    weight_state: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight given to affective state match in retrieval scoring.",
    )
    classifier_model: str = Field(
        default="llama3",
        description="LLM model name used for emotional state classification.",
    )
    classifier_prompt: str = Field(
        default=(
            "Analyze the emotional/cognitive state expressed in this text and score it "
            "on three dimensions. Reply ONLY with three floats separated by commas:\n"
            "valence (-1.0 to 1.0, negative to positive), "
            "arousal (0.0 to 1.0, calm to urgent), "
            "complexity (0.0 to 1.0, simple to complex)\n\n"
            'Text: "{content}"\n\n'
            "Example reply: 0.3, 0.7, 0.5"
        ),
        description="Prompt template for cognitive state classification.",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "AffectiveConfig":
        """Ensure weights sum to approximately 1.0."""
        total = self.weight_similarity + self.weight_state
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"AffectiveConfig weights must sum to 1.0, got {total:.3f}. "
                "Adjust weight_similarity and weight_state."
            )
        return self


class SleepConfig(BaseModel):
    """
    Configuration for the SleepDaemon module.

    Mirrors the parameters of biological sleep consolidation:
    - Consolidation occurs during 'idle' periods (low activity windows)
    - A minimum number of episodes must accumulate before consolidation (like
      the hippocampus waiting for enough material to replay)
    - Proceduralization is optional (analogous to implicit/procedural memory
      formation from declarative experiences)
    """

    consolidation_interval_seconds: int = Field(
        default=3600,
        ge=0,
        description="Minimum seconds between consolidation runs (default: 1 hour). Use 0 to always allow consolidation (useful for testing).",
    )
    min_episodes_before_consolidation: int = Field(
        default=10,
        gt=0,
        description=(
            "Minimum episodes in buffer before consolidation triggers. "
            "Prevents premature consolidation with insufficient data."
        ),
    )
    enable_proceduralization: bool = Field(
        default=False,
        description=(
            "Whether to attempt generating Python tool code from repeated patterns. "
            "Disabled by default as it requires careful LLM prompting. "
            "SECURITY NOTE: Generated code is UNTRUSTED LLM output and must "
            "never be executed without human review."
        ),
    )
    consolidation_prompt: str = Field(
        default=(
            "You are a memory consolidation agent. Below is a sequence of conversation "
            "episodes from a user session:\n\n{episodes}\n\n"
            "Extract permanent facts, user preferences, behavioral patterns, and "
            "important information that should be retained long-term. "
            "Format your response as a numbered list of distinct facts, one per line. "
            "Be concise and specific. Omit transient information."
        ),
        description="Prompt template for fact extraction during consolidation.",
    )
    proceduralization_prompt: str = Field(
        default=(
            "You are analyzing conversation logs to identify repeated reasoning patterns "
            "that could be automated. Review these episodes:\n\n{episodes}\n\n"
            "If you identify a clearly repeated task pattern (appearing 3+ times), "
            "write a complete, executable Python function that automates it. "
            "If no clear pattern exists, reply with 'NO_PATTERN'. "
            "Return ONLY valid Python code or 'NO_PATTERN'."
        ),
        description="Prompt template for tool generation via proceduralization.",
    )
    max_episodes_per_consolidation: int = Field(
        default=100,
        gt=0,
        description="Maximum episodes to include in a single consolidation prompt.",
    )


class SpreadingConfig(BaseModel):
    """
    Configuration for the SpreadingActivation module.

    Parameters mirror the biophysics of neural spreading activation:
    - initial_energy: the magnitude of activation injected at the seed node
    - decay_rate: fraction of energy lost per hop (analogous to synaptic loss)
    - activation_threshold: minimum energy for a node to be considered 'primed'
    - max_hops: limits propagation depth (biological networks have finite reach)
    """

    initial_energy: float = Field(
        default=1.0,
        gt=0.0,
        description="Initial activation energy injected at the seed node.",
    )
    decay_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of energy lost per hop during propagation. "
            "0.2 = 20% loss per hop, matching the original design spec."
        ),
    )
    activation_threshold: float = Field(
        default=0.3,
        ge=0.0,
        description="Minimum energy for a node to be included in retrieval results.",
    )
    max_hops: int = Field(
        default=3,
        gt=0,
        description="Maximum graph traversal depth for energy propagation.",
    )
    auto_connect_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for auto-connecting nodes during graph construction."
        ),
    )
    natural_decay_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Rate at which all node energies decay naturally over time (forgetting).",
    )
    hydrate_on_startup: bool = Field(
        default=True,
        description=("Rebuild spreading activation nodes from persisted store at engine startup."),
    )
    startup_hydration_limit: int = Field(
        default=5000,
        gt=0,
        description=("Maximum number of stored chunks to load into spreading graph on startup."),
    )
    startup_auto_connect: bool = Field(
        default=True,
        description=("Whether to run graph auto-connect after startup hydration."),
    )


class MemorySafetyConfig(BaseModel):
    """
    Configuration for memory write safety filtering.

    This policy is applied to all long-term memory writes (direct encoding,
    reconsolidation updates, and sleep consolidation facts).
    """

    enabled: bool = Field(
        default=True,
        description="Whether memory safety filtering is enforced on writes.",
    )
    secret_action: Literal["allow", "redact", "block"] = Field(
        default="block",
        description="Action when secret-like patterns are detected in memory content.",
    )
    pii_action: Literal["allow", "redact", "block"] = Field(
        default="redact",
        description="Action when PII-like patterns are detected in memory content.",
    )


class MemoryGovernanceConfig(BaseModel):
    """
    Configuration for memory governance controls.

    - capture_mode controls which ingestion channels are allowed to persist.
    - retention_ttl_days prunes memories older than the TTL.
    - max_chunks_per_scope caps memory growth per (scope, scope_id).
    """

    capture_mode: Literal["all", "manual_only", "hooks_only"] = Field(
        default="all",
        description=(
            "Allowed ingestion channels: all, manual_only, hooks_only. "
            "Manual means non-hook process() calls; hooks means claude hook ingestion."
        ),
    )
    retention_ttl_days: int = Field(
        default=0,
        ge=0,
        description="If > 0, prune memories older than this many days on write operations.",
    )
    max_chunks_per_scope: int = Field(
        default=0,
        ge=0,
        description="If > 0, keep at most this many chunks per scope partition.",
    )


class MnemosConfig(BaseModel):
    """
    Top-level configuration for the MnemosEngine.

    Composes all module-specific configs into a single validated object.
    Pass this to MnemosEngine.__init__ to configure the entire system.

    Example usage:
        config = MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.25),
            sleep=SleepConfig(consolidation_interval_seconds=1800),
        )
        engine = MnemosEngine(config=config, ...)
    """

    surprisal: SurprisalConfig = Field(
        default_factory=SurprisalConfig,
        description="Configuration for the SurprisalGate module.",
    )
    mutable_rag: MutableRAGConfig = Field(
        default_factory=MutableRAGConfig,
        description="Configuration for the MutableRAG module.",
    )
    affective: AffectiveConfig = Field(
        default_factory=AffectiveConfig,
        description="Configuration for the AffectiveRouter module.",
    )
    sleep: SleepConfig = Field(
        default_factory=SleepConfig,
        description="Configuration for the SleepDaemon module.",
    )
    spreading: SpreadingConfig = Field(
        default_factory=SpreadingConfig,
        description="Configuration for the SpreadingActivation module.",
    )
    safety: MemorySafetyConfig = Field(
        default_factory=MemorySafetyConfig,
        description="Memory write safety policy shared across all write paths.",
    )
    governance: MemoryGovernanceConfig = Field(
        default_factory=MemoryGovernanceConfig,
        description="Memory governance controls for capture, retention, and growth limits.",
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose debug logging across all modules.",
    )
