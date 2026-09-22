"""
mnemos/modules/affective.py — State-Dependent Affective Routing.

Neuroscience Basis:
    Human memory is strongly state-dependent: memories encoded during a
    particular emotional state are preferentially retrieved when that state
    is re-experienced (Bower, 1981; Eich & Metcalfe, 1989). The amygdala
    tags memories with emotional context (valence and arousal), and this
    tag influences which memories become accessible during retrieval.

    The prefrontal-amygdala circuit modulates memory access based on current
    emotional state — when you're panicking, you tend to recall other times
    you panicked and how you resolved them, not semantically similar but
    emotionally distant memories.

    Three dimensions from dimensional emotion theory (Russell, 1980):
    - Valence: positive/negative emotional tone
    - Arousal: calm/urgent intensity level
    - Complexity: simple/complex cognitive load

Mnemos Implementation:
    Each interaction is scored on these three axes via an LLM classifier.
    The CognitiveState is attached as metadata to stored MemoryChunks.

    During retrieval, the scoring formula is:
        final_score = (cosine_similarity * 0.7) + (state_match * 0.3)

    Where state_match = 1.0 - cognitive_state.distance(current_state)

    This ensures emotionally urgent queries (bugs, crises) retrieve memories
    of past crises — not just semantically similar but emotionally cold content.
"""

from __future__ import annotations

import re
import math
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

from ..config import AffectiveConfig
from ..judgment import Judge, Score, ScoreAnswer, UnavailableJudge
from ..scopes import interaction_partition
from ..types import CognitiveState, Interaction, MemoryChunk
from ..utils.embeddings import EmbeddingProvider, cosine_similarity, embed_text_async
from ..utils.llm import LLMProvider
from ..utils.reliability import MnemosConfigurationError
from ..utils.storage import MemoryStore

AFFECTIVE_SCORES: dict[str, Score] = {
    "valence": Score(
        "How does the speaker feel in text, from negative to positive?",
        (
            "very negative: frustration, anger, distress",
            "somewhat negative",
            "neutral or purely technical",
            "somewhat positive",
            "very positive: delight, gratitude, relief",
        ),
    ),
    "arousal": Score(
        "How urgent or intense is text?",
        (
            "calm, routine",
            "mild",
            "moderate",
            "pressing",
            "urgent: an outage, a deadline, a blocking failure",
        ),
    ),
    "complexity": Score(
        "How much reasoning does text involve?",
        (
            "trivial",
            "simple",
            "moderate",
            "involved",
            "highly complex, multi-part reasoning",
        ),
    ),
}


class AffectiveRouter:
    """
    State-dependent memory routing with emotional/cognitive tagging.

    Classifies interactions on three affective axes (valence, arousal, complexity)
    and uses the current cognitive state to modulate retrieval scores — giving
    priority to memories encoded in emotionally similar states.

    Args:
        llm: LLM provider used for affective state classification.
        embedder: Embedding provider for semantic similarity calculation.
        config: AffectiveConfig controlling weights and model settings.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        config: AffectiveConfig | None = None,
        judge: Judge | None = None,
    ) -> None:
        self._llm = llm
        self._judge: Judge = judge if judge is not None else UnavailableJudge()
        self._embedder = embedder
        self._config = config or AffectiveConfig()
        self._classifier_options()
        # Running window of recent states (last N interactions)
        self._state_history: deque[CognitiveState] = deque(maxlen=10)
        self._states: OrderedDict[tuple[str, str, str, str], deque[CognitiveState]] = OrderedDict()
        # Stats
        self._total_classified: int = 0
        self._total_retrieved: int = 0

    def _classifier_options(self) -> dict[str, Any]:
        model = self._config.classifier_model
        if model is None:
            return {}
        if not getattr(self._llm, "supports_model_override", False):
            raise MnemosConfigurationError(
                "The classifier provider does not support a model override"
            )
        return {"model": model}

    def _build_classifier_prompt(self, content: str) -> str:
        """
        Build the LLM prompt for affective state classification.

        Args:
            content: The interaction text to classify.

        Returns:
            Formatted classification prompt.
        """
        return self._config.classifier_prompt.format(content=content)

    def _parse_state_response(self, response: str) -> CognitiveState | None:
        """
        Parse the LLM's three-number response into a CognitiveState.

        Expected format: "valence, arousal, complexity" e.g. "-0.3, 0.8, 0.6"

        Returns unavailable on parse failure so callers do not attach an
        unverified state to memory.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed CognitiveState, or neutral state if parsing fails.
        """
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        match = re.fullmatch(
            rf"\s*({number})\s*,\s*({number})\s*,\s*({number})\s*",
            response,
        )
        if match is None:
            return None

        try:
            valence, arousal, complexity = (float(value) for value in match.groups())
        except ValueError:
            return None
        if not all(math.isfinite(value) for value in (valence, arousal, complexity)):
            return None

        # Clamp to valid ranges
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        complexity = max(0.0, min(1.0, complexity))
        return CognitiveState(
            valence=valence,
            arousal=arousal,
            complexity=complexity,
        )

    async def classify_state(self, interaction: Interaction) -> CognitiveState | None:
        """
        Classify the emotional/cognitive state of an interaction.

        Uses the configured LLM classifier to score the interaction on
        valence, arousal, and complexity. Updates the internal state history
        to track the user's current cognitive context.

        Args:
            interaction: The interaction to classify.

        Returns:
            A CognitiveState representing the emotional/cognitive context.
        """
        self._total_classified += 1
        if self._judge.provider != "unavailable":
            state = await self._judge_state(interaction)
        else:
            prompt = self._build_classifier_prompt(interaction.content)
            try:
                response = await self._llm.predict(prompt, **self._classifier_options())
                state = self._parse_state_response(response)
            except Exception:
                state = None

        if state is not None:
            self._remember_state(interaction, state)
        return state

    async def _judge_state(self, interaction: Interaction) -> CognitiveState | None:
        """Three Score questions; the answer is already a weighted position."""
        try:
            judgment = await self._judge.ask(
                {"text": interaction.content, "role": interaction.role},
                AFFECTIVE_SCORES,
                purpose="classify_state",
                model=self._config.classifier_model,
            )
        except Exception:
            return None
        valence = judgment.answers["valence"]
        arousal = judgment.answers["arousal"]
        complexity = judgment.answers["complexity"]
        if not all(isinstance(a, ScoreAnswer) for a in (valence, arousal, complexity)):
            return None
        assert isinstance(valence, ScoreAnswer)
        assert isinstance(arousal, ScoreAnswer)
        assert isinstance(complexity, ScoreAnswer)
        return CognitiveState(
            valence=valence.scaled(-1.0, 1.0, 5),
            arousal=arousal.scaled(0.0, 1.0, 5),
            complexity=complexity.scaled(0.0, 1.0, 5),
        )

    def record_state(self, interaction: Interaction, state: CognitiveState) -> None:
        """Adopt a state judged elsewhere (for example alongside classification)."""
        self._total_classified += 1
        self._remember_state(interaction, state)

    def _remember_state(self, interaction: Interaction, state: CognitiveState) -> None:
        # Update state history (deque handles maxlen automatically)
        self._state_history.append(state)
        partition = interaction_partition(interaction.metadata)
        self._states.setdefault(partition, deque(maxlen=10)).append(state)
        self._states.move_to_end(partition)
        while len(self._states) > 256:
            self._states.popitem(last=False)

    def tag_chunk(self, chunk: MemoryChunk, state: CognitiveState | None) -> MemoryChunk:
        """
        Attach a CognitiveState to a MemoryChunk (amygdala tagging).

        Adds the state both as the chunk's cognitive_state field and in
        metadata for easy inspection and filtering.

        Args:
            chunk: The chunk to tag.
            state: The cognitive state at encoding time.

        Returns:
            The modified chunk (mutated in-place and returned).
        """
        if state is None:
            chunk.cognitive_state = None
            chunk.cognitive_state_status = "provider_unavailable"
            chunk.metadata["cognitive_state_status"] = "provider_unavailable"
            chunk.metadata.pop("cognitive_state", None)
            return chunk

        chunk.cognitive_state = state
        chunk.cognitive_state_status = "measured"
        chunk.metadata["cognitive_state_status"] = "measured"
        chunk.metadata["cognitive_state"] = {
            "valence": state.valence,
            "arousal": state.arousal,
            "complexity": state.complexity,
        }
        return chunk

    def score_chunk(
        self,
        chunk: MemoryChunk,
        query_embedding: list[float],
        current_state: CognitiveState,
    ) -> float:
        """
        Compute the affective retrieval score for a single chunk.

        Formula (from the neuroscience-inspired design):
            final_score = (cosine_similarity × w_sim) + (state_match × w_state)

        Where:
            cosine_similarity = semantic similarity between query and chunk
            state_match = 1.0 - cognitive_state.distance(current_state)
            w_sim = config.weight_similarity (default 0.7)
            w_state = config.weight_state (default 0.3)

        If a chunk has no cognitive state tag, state_match defaults to 0.5
        (neutral — neither boosted nor penalized).

        Args:
            chunk: The chunk to score.
            query_embedding: Embedding of the current query.
            current_state: The user's current cognitive state.

        Returns:
            Final blended score in approximately [0, 1].
        """
        # Semantic similarity component
        if chunk.embedding and len(chunk.embedding) == len(query_embedding):
            sim = cosine_similarity(query_embedding, chunk.embedding)
            # Clamp to [0, 1] for scoring (cosine can technically be negative)
            sim = max(0.0, sim)
        else:
            sim = 0.0

        # Affective state match component
        if chunk.cognitive_state is not None:
            state_distance = chunk.cognitive_state.distance(current_state)
            state_match = 1.0 - state_distance
        else:
            state_match = 0.5  # Neutral for untagged chunks

        # Blended score
        return sim * self._config.weight_similarity + state_match * self._config.weight_state

    async def retrieve(
        self,
        query: str,
        current_state: CognitiveState,
        store: MemoryStore,
        top_k: int = 5,
        candidate_k: int | None = None,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
        query_embedding: list[float] | None = None,
        candidates: list[MemoryChunk] | None = None,
    ) -> list[MemoryChunk]:
        """
        Retrieve and re-rank chunks using affective state scoring.

        Retrieval pipeline:
        1. Get a larger candidate set by raw embedding similarity (candidate_k)
        2. Re-score each candidate using the affective blended formula
        3. Return the top_k by final score

        This two-stage approach ensures:
        - We don't miss semantically relevant chunks (large candidate pool)
        - Affective state can promote/demote candidates in the final ranking

        Args:
            query: The text query to search for.
            current_state: The user's current cognitive state.
            store: Memory store to search.
            top_k: Number of results to return.
            candidate_k: Size of candidate pool before re-ranking.
                         Defaults to max(top_k * 3, 20) for good coverage.
            filter_fn: Optional predicate to filter candidate chunks by metadata.
            query_embedding: Optional precomputed query embedding to reuse.
            candidates: Optional pre-fetched candidate pool to rerank in memory.

        Returns:
            Top-k MemoryChunks sorted by affective blended score.
        """
        self._total_retrieved += 1

        # Embed the query
        if query_embedding is None:
            query_embedding = await embed_text_async(self._embedder, query, role="query")

        if candidates is None:
            # Fetch a larger candidate pool for re-ranking
            ck = candidate_k or max(top_k * 3, 20)
            candidates = store.retrieve(query_embedding, top_k=ck, filter_fn=filter_fn)
        elif filter_fn is not None:
            candidates = [chunk for chunk in candidates if filter_fn(chunk)]
            if candidate_k is not None:
                candidates = candidates[:candidate_k]

        if not candidates:
            return []

        # Re-score with affective blending
        scored: list[tuple[float, MemoryChunk]] = []
        for chunk in candidates:
            score = self.score_chunk(chunk, query_embedding, current_state)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def get_current_state(self, metadata: dict[str, Any] | None = None) -> CognitiveState:
        """
        Return the most recent cognitive state from history.

        If no interactions have been classified yet, returns a neutral baseline.

        Returns:
            The last classified CognitiveState, or neutral defaults.
        """
        history = self._states.get(interaction_partition(metadata or {}), ())
        if history:
            return history[-1]
        return CognitiveState(valence=0.0, arousal=0.5, complexity=0.5)

    def get_average_state(self, metadata: dict[str, Any] | None = None) -> CognitiveState:
        """
        Compute the average cognitive state across the history window.

        Provides a smoothed representation of the user's recent emotional
        context, less sensitive to single-turn fluctuations.

        Returns:
            Averaged CognitiveState across history.
        """
        history = self._states.get(interaction_partition(metadata or {}), ())
        if not history:
            return CognitiveState(valence=0.0, arousal=0.5, complexity=0.5)

        n = len(history)
        return CognitiveState(
            valence=sum(s.valence for s in history) / n,
            arousal=sum(s.arousal for s in history) / n,
            complexity=sum(s.complexity for s in history) / n,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the AffectiveRouter module."""
        current = self.get_current_state()
        return {
            "module": "AffectiveRouter",
            "total_classified": self._total_classified,
            "total_retrieved": self._total_retrieved,
            "history_length": len(self._state_history),
            "classifier": {
                "model": (
                    self._config.classifier_model
                    if self._config.classifier_model is not None
                    else getattr(self._llm, "model", None)
                ),
                "selection": (
                    "override" if self._config.classifier_model is not None else "provider_default"
                ),
            },
            "current_state": {
                "valence": round(current.valence, 3),
                "arousal": round(current.arousal, 3),
                "complexity": round(current.complexity, 3),
            },
            "weights": {
                "similarity": self._config.weight_similarity,
                "state": self._config.weight_state,
            },
        }
