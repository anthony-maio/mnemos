"""
mnemos/engine.py — MnemosEngine: the orchestrating conductor of all memory modules.

The MnemosEngine wires all five biomimetic modules into a coherent pipeline
that mirrors the full human memory cycle:

    Encode path (process):
        Input → SurprisalGate (filter) → AffectiveRouter (tag) → Store
                                        ↓
                              SleepDaemon episodic buffer

    Retrieve path (retrieve):
        Query → SpreadingActivation (associative context)
               + AffectiveRouter (state-dependent re-ranking)
               + MutableRAG (reconsolidation of retrieved chunks)
             → Ranked results

    Maintenance path (consolidate):
        Idle trigger → SleepDaemon (hippocampal-neocortical transfer)

This is the main entry point for library consumers. Instantiate MnemosEngine
and call process()/retrieve()/consolidate() to use the full pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import MnemosConfig
from .modules.affective import AffectiveRouter
from .modules.mutable_rag import MutableRAG
from .modules.sleep import SleepDaemon
from .modules.spreading import SpreadingActivation
from .modules.surprisal import SurprisalGate
from .types import ConsolidationResult, Interaction, MemoryChunk, ProcessResult
from .utils.embeddings import EmbeddingProvider, cosine_similarity
from .utils.llm import LLMProvider
from .utils.storage import MemoryStore

logger = logging.getLogger(__name__)


def _on_reconsolidation_done(t: asyncio.Task[Any]) -> None:
    """Log errors from background reconsolidation tasks instead of silently swallowing them."""
    if not t.cancelled() and t.exception():
        logger.error("Background reconsolidation failed: %s", t.exception())


class MnemosEngine:
    """
    Orchestrator for the full biomimetic memory pipeline.

    Composes all five memory modules into a unified interface:
    - process(): encode interactions through the full pipeline
    - retrieve(): contextually retrieve memories with affective re-ranking
    - consolidate(): manually trigger sleep consolidation
    - get_stats(): inspect the state of all modules

    Args:
        config: MnemosConfig with settings for all modules.
        llm: LLM provider used across all modules.
        embedder: Embedding provider for all vector operations.
        store: Primary long-term memory store.
        episodic_store: Optional separate store for episodic buffer.
                        Defaults to the same as store if not provided.

    Example:
        from mnemos import MnemosEngine, MnemosConfig
        from mnemos.utils import MockLLMProvider, SimpleEmbeddingProvider, InMemoryStore

        engine = MnemosEngine(
            config=MnemosConfig(),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(),
            store=InMemoryStore(),
        )

        result = await engine.process(Interaction(role="user", content="Hello!"))
        memories = await engine.retrieve("user preferences", top_k=5)
    """

    def __init__(
        self,
        config: MnemosConfig | None = None,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        from .utils.llm import MockLLMProvider
        from .utils.embeddings import SimpleEmbeddingProvider
        from .utils.storage import InMemoryStore

        self.config = config or MnemosConfig()

        # Wire up providers with defaults for zero-config startup
        self._llm = llm or MockLLMProvider()
        self._embedder = embedder or SimpleEmbeddingProvider(
            dim=self.config.surprisal.embedding_dim
        )
        self._store = store or InMemoryStore()

        # Initialize all five modules
        self.surprisal_gate = SurprisalGate(
            llm=self._llm,
            embedder=self._embedder,
            store=self._store,
            config=self.config.surprisal,
        )

        self.mutable_rag = MutableRAG(
            llm=self._llm,
            embedder=self._embedder,
            store=self._store,
            config=self.config.mutable_rag,
        )

        self.affective_router = AffectiveRouter(
            llm=self._llm,
            embedder=self._embedder,
            config=self.config.affective,
        )

        self.sleep_daemon = SleepDaemon(
            store=self._store,
            config=self.config.sleep,
        )

        self.spreading_activation = SpreadingActivation(
            embedder=self._embedder,
            config=self.config.spreading,
        )

    async def process(self, interaction: Interaction) -> ProcessResult:
        """
        Process an interaction through the full encoding pipeline.

        Pipeline:
        1. SurprisalGate: compute prediction error; gate low-surprisal inputs
        2. If passes gate: AffectiveRouter: classify emotional state and tag chunk
        3. Update SpreadingActivation graph with new chunk
        4. Add to SleepDaemon episodic buffer (always, regardless of gate)

        Args:
            interaction: The user/assistant interaction to process.

        Returns:
            ProcessResult with storage decision, chunk (if stored), and reason.
        """
        # Always add to episodic buffer for eventual consolidation
        # (The hippocampus stores everything briefly, even mundane inputs)
        self.sleep_daemon.add_episode(interaction)

        # Step 1: Surprisal gate — filter by prediction error
        result = await self.surprisal_gate.process(interaction)

        if not result.stored or result.chunk is None:
            return result

        # Step 2: Affective tagging — classify and attach cognitive state
        state = await self.affective_router.classify_state(interaction)
        self.affective_router.tag_chunk(result.chunk, state)

        # Update the stored chunk with the affective tag
        self._store.update(result.chunk.id, result.chunk)

        # Step 3: Add to spreading activation graph
        self.spreading_activation.add_node_from_chunk(result.chunk)

        # Auto-connect new node to its semantic neighbors in the graph
        # (Only connect the new node — avoid O(N²) on every process call)
        new_node = self.spreading_activation.get_node(result.chunk.id)
        if new_node and new_node.embedding:
            for existing_node in self.spreading_activation.get_all_nodes():
                if existing_node.id == new_node.id or existing_node.embedding is None:
                    continue
                sim = cosine_similarity(new_node.embedding, existing_node.embedding)
                if sim >= self.config.spreading.auto_connect_threshold:
                    self.spreading_activation.add_edge(new_node.id, existing_node.id, sim)

        if self.config.debug:
            logger.debug(
                f"[MnemosEngine] Processed: stored={result.stored}, "
                f"salience={result.salience:.3f}, reason={result.reason}"
            )

        return result

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        reconsolidate: bool = True,
    ) -> list[MemoryChunk]:
        """
        Retrieve memories through the full contextual retrieval pipeline.

        Pipeline:
        1. Classify current affective state from query text
        2. SpreadingActivation: find associatively-connected context
        3. AffectiveRouter: re-rank all candidates with emotional weighting
        4. MutableRAG: flag retrieved chunks as labile for reconsolidation

        Args:
            query: The query string to search for.
            top_k: Number of results to return.
            reconsolidate: Whether to run reconsolidation on labile chunks
                           after retrieval (default True, runs in background).

        Returns:
            Top-k MemoryChunks ranked by affective blended score.
        """
        # Step 1: Classify affective state of the query
        query_interaction = Interaction(role="user", content=query)
        current_state = await self.affective_router.classify_state(query_interaction)

        # Step 2: SpreadingActivation — find associatively connected neighbors
        query_embedding = self._embedder.embed(query)
        activated_nodes = self.spreading_activation.retrieve(
            query_embedding,
            top_k=top_k * 3,  # Get larger pool for re-ranking
        )

        # Gather IDs of activated nodes to boost them in final scoring
        activated_ids: set[str] = {node.id for node in activated_nodes}

        # Step 3: AffectiveRouter — retrieve and re-rank from store
        affective_chunks = await self.affective_router.retrieve(
            query=query,
            current_state=current_state,
            store=self._store,
            top_k=top_k * 2,  # Slightly larger pool
        )

        # Merge activated nodes: boost chunks that appear in activation graph
        chunk_scores: dict[str, tuple[float, MemoryChunk]] = {}
        for i, chunk in enumerate(affective_chunks):
            # Base score: inverse rank from affective retrieval
            base_score = 1.0 - (i / max(len(affective_chunks), 1))
            # Activation boost: chunks in the spreading graph get a 20% boost
            if chunk.id in activated_ids:
                base_score *= 1.2
            chunk_scores[chunk.id] = (base_score, chunk)

        # Also include any activated nodes not in the affective results
        missing_ids = activated_ids - set(chunk_scores.keys())
        if missing_ids:
            store_chunks_by_id = {c.id: c for c in self._store.get_all()}
            for node in activated_nodes:
                if node.id in missing_ids and node.id in store_chunks_by_id:
                    chunk = store_chunks_by_id[node.id]
                    # Score based on activation energy
                    score = node.energy * 0.5  # Activation-only score (lower weight)
                    chunk_scores[node.id] = (score, chunk)

        # Sort by score and take top_k
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )
        final_chunks = [chunk for _, chunk in sorted_chunks[:top_k]]

        # Step 4: MutableRAG — flag as labile and optionally reconsolidate
        for chunk in final_chunks:
            self.mutable_rag.mark_labile(chunk, query)

        if reconsolidate and final_chunks:
            # Run reconsolidation as a background task (non-blocking)
            task = asyncio.create_task(self.mutable_rag.process_labile_chunks())
            task.add_done_callback(_on_reconsolidation_done)

        if self.config.debug:
            logger.debug(
                f"[MnemosEngine] Retrieved {len(final_chunks)} chunks for query: {query!r}"
            )

        return final_chunks

    async def consolidate(self) -> ConsolidationResult:
        """
        Manually trigger a sleep consolidation cycle.

        Extracts permanent facts from the episodic buffer, stores them as
        semantic MemoryChunks, clears the buffer, and optionally generates
        procedural tool code.

        Returns:
            ConsolidationResult with facts extracted, chunks pruned, tools generated.
        """
        result = await self.sleep_daemon.consolidate(
            llm_provider=self._llm,
            embedder=self._embedder,
        )

        # Add consolidated chunks to the spreading activation graph
        new_chunks = self._store.get_all()
        for chunk in new_chunks:
            if (
                chunk.metadata.get("source") == "sleep_consolidation"
                and self.spreading_activation.get_node(chunk.id) is None
            ):
                self.spreading_activation.add_node_from_chunk(chunk)

        # Rebuild connections for newly added nodes
        if new_chunks:
            self.spreading_activation.auto_connect()

        if self.config.debug:
            logger.debug(
                f"[MnemosEngine] Consolidated: {len(result.facts_extracted)} facts, "
                f"{result.chunks_pruned} episodes pruned"
            )

        return result

    async def process_batch(self, interactions: list[Interaction]) -> list[ProcessResult]:
        """
        Process a list of interactions sequentially.

        Useful for loading prior conversation history into memory at startup.

        Args:
            interactions: List of interactions to process in order.

        Returns:
            List of ProcessResults, one per interaction.
        """
        results = []
        for interaction in interactions:
            result = await self.process(interaction)
            results.append(result)
        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Return statistics from all modules and the engine itself.

        Returns:
            Dict with stats from each module plus overall engine stats.
        """
        store_stats = self._store.get_stats()
        return {
            "engine": {
                "store_backend": store_stats.get("backend", "unknown"),
                "total_chunks": store_stats.get("total_chunks", 0),
            },
            "surprisal_gate": self.surprisal_gate.get_stats(),
            "mutable_rag": self.mutable_rag.get_stats(),
            "affective_router": self.affective_router.get_stats(),
            "sleep_daemon": self.sleep_daemon.get_stats(),
            "spreading_activation": self.spreading_activation.get_stats(),
            "store": store_stats,
        }

    @property
    def store(self) -> MemoryStore:
        """Direct access to the underlying memory store."""
        return self._store

    @property
    def llm(self) -> LLMProvider:
        """Direct access to the LLM provider."""
        return self._llm

    @property
    def embedder(self) -> EmbeddingProvider:
        """Direct access to the embedding provider."""
        return self._embedder
