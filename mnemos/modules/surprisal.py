"""
mnemos/modules/surprisal.py — Surprisal-Triggered Encoding (Predictive Processing).

Neuroscience Basis:
    The brain operates on Predictive Coding (Friston, 2005; Clark, 2013).
    We constantly generate predictions about our sensory environment and
    only commit experiences to long-term memory when there is a *prediction
    error* — a significant mismatch between what was expected and what occurred.
    This is why mundane, expected events are forgotten while surprising,
    unexpected events are vividly remembered.

    The hippocampus acts as a 'novelty detector', firing strongly in response
    to prediction errors, triggering memory consolidation for surprising events.

Mnemos Implementation:
    A background LLM predicts the user's next intent given conversation history.
    When the user's actual input arrives, we compute the semantic distance
    (cosine distance in embedding space) between the prediction and reality.
    If this *surprisal score* exceeds the configured threshold, the interaction
    is committed to long-term memory with a high salience weight.

    Low surprisal (expected input) → discard (no memory encoding needed)
    High surprisal (unexpected input) → encode with high salience weight
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from ..config import SurprisalConfig
from ..types import Interaction, MemoryChunk, ProcessResult
from ..utils.embeddings import EmbeddingProvider, cosine_distance
from ..utils.llm import LLMProvider
from ..utils.storage import MemoryStore


class SurprisalGate:
    """
    Surprisal-triggered memory encoding gate.

    Acts as the information-theoretic filter at the ingestion layer.
    Only interactions with semantic surprisal above the configured threshold
    are passed to long-term storage — mirroring the brain's efficiency in
    encoding only prediction errors, not mundane expected inputs.

    Args:
        llm: LLM provider used for next-intent prediction.
        embedder: Embedding provider for semantic distance calculation.
        store: Memory store where high-surprisal chunks are written.
        config: SurprisalConfig controlling thresholds and model settings.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        store: MemoryStore,
        config: SurprisalConfig | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._config = config or SurprisalConfig()
        # Conversation history buffer — acts as hippocampal working memory
        self._history: deque[Interaction] = deque(maxlen=self._config.history_window)
        # Stats tracking
        self._total_processed: int = 0
        self._total_stored: int = 0

    def _build_prediction_prompt(self, current_interaction: Interaction) -> str:
        """
        Build the LLM prompt for predicting the user's next intent.

        Includes recent conversation history to provide context for prediction,
        just as the brain uses contextual priming to generate predictions.

        Args:
            current_interaction: The interaction we're about to score.

        Returns:
            A formatted prompt string.
        """
        history_lines = []
        for turn in self._history:
            role_label = turn.role.capitalize()
            history_lines.append(f"{role_label}: {turn.content}")

        history_str = "\n".join(history_lines) if history_lines else "(No prior history)"

        return (
            "You are a predictive intent engine. Given the conversation history below, "
            "predict what the user will say or ask next. Be specific and concise. "
            "Output ONLY your prediction, nothing else.\n\n"
            f"Conversation history:\n{history_str}\n\n"
            "Predicted next user input:"
        )

    async def _compute_surprisal(self, interaction: Interaction) -> float:
        """
        Compute the surprisal score for an interaction.

        Steps:
        1. Ask the LLM to predict the next user input from history
        2. Embed both the prediction and the actual input
        3. Return cosine distance between the two embeddings

        Args:
            interaction: The interaction to score.

        Returns:
            Surprisal score in [0, 2], typically [0, 1] for text embeddings.
            Higher = more surprising = higher salience.
        """
        if not self._history:
            # No history to predict from — treat as moderately surprising
            return 0.5

        # Step 1: Get prediction from LLM
        prediction_prompt = self._build_prediction_prompt(interaction)
        try:
            prediction = await self._llm.predict(prediction_prompt)
        except Exception:
            # If LLM fails, fall back to moderate surprisal
            return 0.5

        if not prediction.strip():
            return 0.5

        # Step 2: Embed both prediction and actual input
        prediction_embedding = self._embedder.embed(prediction)
        actual_embedding = self._embedder.embed(interaction.content)

        # Step 3: Cosine distance = semantic divergence = surprisal score
        return cosine_distance(prediction_embedding, actual_embedding)

    def _salience_from_surprisal(self, surprisal: float) -> float:
        """
        Convert a raw surprisal score to a [0, 1] salience weight.

        The salience function is sigmoid-like: scores near the threshold
        produce moderate salience; scores well above produce high salience.
        We normalize so that surprisal = 1.0 maps to salience ≈ 1.0.

        Args:
            surprisal: Raw cosine distance surprisal score.

        Returns:
            Salience weight in [0, 1].
        """
        # Simple linear normalization: clamp to [0, 1]
        return min(max(surprisal, 0.0), 1.0)

    async def process(self, interaction: Interaction) -> ProcessResult:
        """
        Process an interaction through the surprisal gate.

        If the interaction's surprisal score exceeds the threshold, it is
        committed to long-term memory with a salience-weighted MemoryChunk.
        If below threshold, it is discarded (not stored).

        The interaction is always added to the history buffer regardless
        of whether it was stored — history is used for prediction context.

        Args:
            interaction: The interaction to evaluate and potentially store.

        Returns:
            ProcessResult indicating whether the interaction was stored,
            the resulting chunk (if stored), the surprisal score, and a reason.
        """
        self._total_processed += 1

        # Skip very short content (likely not meaningful)
        if len(interaction.content.strip()) < self._config.min_content_length:
            self._history.append(interaction)
            return ProcessResult(
                stored=False,
                salience=0.0,
                reason=f"Content too short (< {self._config.min_content_length} chars).",
            )

        # Compute surprisal score
        surprisal = await self._compute_surprisal(interaction)

        # Always update history buffer (even if not stored)
        self._history.append(interaction)

        if surprisal <= self._config.threshold:
            return ProcessResult(
                stored=False,
                salience=surprisal,
                reason=(
                    f"Low surprisal ({surprisal:.3f} ≤ threshold {self._config.threshold:.3f}). "
                    "Interaction matched prediction — not encoded."
                ),
            )

        # High surprisal: encode as a memory chunk
        salience = self._salience_from_surprisal(surprisal)
        embedding = self._embedder.embed(interaction.content)

        chunk = MemoryChunk(
            content=interaction.content,
            embedding=embedding,
            metadata={
                "role": interaction.role,
                "surprisal_score": surprisal,
                "source": "surprisal_gate",
                "original_timestamp": interaction.timestamp.isoformat(),
                **interaction.metadata,
            },
            salience=salience,
        )

        self._store.store(chunk)
        self._total_stored += 1

        return ProcessResult(
            stored=True,
            chunk=chunk,
            salience=salience,
            reason=(
                f"High surprisal ({surprisal:.3f} > threshold {self._config.threshold:.3f}). "
                f"Encoded with salience {salience:.3f}."
            ),
        )

    def add_to_history(self, interaction: Interaction) -> None:
        """
        Manually add an interaction to the history buffer without processing.

        Useful for seeding history with prior context (e.g., loading a saved session).

        Args:
            interaction: The interaction to add to history.
        """
        self._history.append(interaction)

    def clear_history(self) -> None:
        """Clear the conversation history buffer."""
        self._history.clear()

    def get_history(self) -> list[Interaction]:
        """Return the current conversation history buffer as a list."""
        return list(self._history)

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the surprisal gate's processing."""
        store_rate = (
            self._total_stored / self._total_processed if self._total_processed > 0 else 0.0
        )
        return {
            "module": "SurprisalGate",
            "total_processed": self._total_processed,
            "total_stored": self._total_stored,
            "store_rate": round(store_rate, 4),
            "history_length": len(self._history),
            "threshold": self._config.threshold,
        }
