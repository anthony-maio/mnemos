"""
mnemos/modules/mutable_rag.py — Mutable RAG (Memory Reconsolidation).

Neuroscience Basis:
    Human memory is not read-only. Every time a memory is retrieved, it enters
    a 'labile' (unstable) state — it becomes open to modification before being
    re-stabilized (reconsolidated) into long-term storage (Nader, Schafe &
    LeDoux, 2000; Misanin et al., 1968).

    This lability is not a bug — it's a feature. It allows memories to be
    updated with current information, preventing the accumulation of outdated
    facts. The memory trace is physically altered each time it is recalled.

    Key insight: In humans, the very act of remembering modifies what is remembered.

Mnemos Implementation:
    When chunks are retrieved, they are flagged as 'labile'. An async background
    process then asks the LLM: "Has this fact changed given the new context?"
    If yes, the chunk is overwritten in-place with an updated version (version++).
    If no, the access count is incremented and the chunk is left unchanged.

    This creates a *self-healing* database that naturally adapts to new information
    without accumulating contradictory entries — solving the classic RAG problem
    of stale facts competing with fresh ones in the context window.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..config import MutableRAGConfig
from ..memory_safety import MemoryWriteFirewall
from ..types import MemoryChunk, ProcessResult
from ..utils.embeddings import EmbeddingProvider, embed_text_async
from ..utils.llm import LLMProvider
from ..utils.storage import MemoryStore


class MutableRAG:
    """
    Memory reconsolidation module — makes the vector database self-healing.

    On retrieval, chunks are marked as 'labile' (unstable). A reconsolidation
    pass then evaluates each labile chunk against the current conversation
    context and rewrites stale facts in-place.

    Args:
        llm: LLM provider used for staleness evaluation.
        embedder: Embedding provider for re-embedding updated chunks.
        store: Memory store containing the chunks to maintain.
        config: MutableRAGConfig controlling reconsolidation behavior.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        store: MemoryStore,
        config: MutableRAGConfig | None = None,
        write_firewall: MemoryWriteFirewall | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._config = config or MutableRAGConfig()
        self._write_firewall = write_firewall
        # Labile chunk queue: chunk_id → (chunk, context_when_retrieved)
        self._labile_chunks: dict[str, tuple[MemoryChunk, str]] = {}
        # Last reconsolidation timestamp per chunk (seconds since epoch)
        self._last_reconsolidation_at: dict[str, float] = {}
        # Stats
        self._total_retrieved: int = 0
        self._total_reconsolidated: int = 0
        self._total_unchanged: int = 0

    def _in_cooldown(self, chunk_id: str) -> bool:
        cooldown = self._config.reconsolidation_cooldown_seconds
        if cooldown <= 0:
            return False
        last = self._last_reconsolidation_at.get(chunk_id)
        if last is None:
            return False
        return (time.time() - last) < cooldown

    def _mark_reconsolidated_now(self, chunk_id: str) -> None:
        self._last_reconsolidation_at[chunk_id] = time.time()

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        current_context: str = "",
    ) -> list[MemoryChunk]:
        """
        Retrieve chunks by similarity and flag them as labile.

        Each retrieved chunk is queued for reconsolidation — the next time
        process_labile_chunks() runs, they will be evaluated for staleness.
        This mirrors how biological memory destabilizes upon retrieval.

        Args:
            query_embedding: Embedding of the current query.
            top_k: Maximum number of chunks to return.
            current_context: The current conversation context (used for
                             reconsolidation evaluation).

        Returns:
            List of top-k MemoryChunks by cosine similarity.
        """
        chunks = self._store.retrieve(query_embedding, top_k=top_k)
        self._total_retrieved += len(chunks)

        if self._config.enabled:
            for chunk in chunks:
                if self._in_cooldown(chunk.id):
                    continue
                # Flag as labile with the current context
                self._labile_chunks[chunk.id] = (chunk, current_context)

        # Increment access counts
        for chunk in chunks:
            touched_at = datetime.now(timezone.utc)
            next_access_count = chunk.access_count + 1
            if self._store.touch(chunk.id, updated_at=touched_at):
                if chunk.access_count < next_access_count:
                    chunk.access_count = next_access_count
                chunk.updated_at = touched_at

        return chunks

    def mark_labile(self, chunk: MemoryChunk, context: str) -> None:
        """
        Flag a chunk as labile (unstable) for future reconsolidation.

        Called by MnemosEngine after retrieval to queue chunks for
        staleness evaluation without accessing private internals.

        Args:
            chunk: The retrieved chunk to mark as labile.
            context: The query/context that triggered retrieval.
        """
        if self._config.enabled and not self._in_cooldown(chunk.id):
            self._labile_chunks[chunk.id] = (chunk, context)

    def _build_reconsolidation_prompt(self, chunk: MemoryChunk, context: str) -> str:
        """
        Build the LLM prompt for evaluating whether a chunk needs updating.

        Args:
            chunk: The memory chunk to evaluate.
            context: Current conversation context.

        Returns:
            Formatted prompt string.
        """
        return self._config.staleness_check_prompt.format(
            content=chunk.content,
            context=context if context else "(No new context available)",
        )

    async def reconsolidate(self, chunk: MemoryChunk, new_context: str) -> tuple[MemoryChunk, bool]:
        """
        Evaluate a chunk for staleness and update if necessary.

        Asks the LLM: "Has this memory changed given new context?"
        - If UNCHANGED: increment access_count, return original chunk
        - If CHANGED: create new chunk (version+1), re-embed, overwrite in store

        Args:
            chunk: The labile chunk to evaluate.
            new_context: The current conversation context.

        Returns:
            Tuple of (resulting_chunk, was_changed: bool).
        """
        prompt = self._build_reconsolidation_prompt(chunk, new_context)

        try:
            response = await self._llm.predict(prompt)
        except Exception:
            # LLM failure: leave chunk unchanged
            return chunk, False

        response = response.strip()

        if response.upper().startswith("UNCHANGED"):
            # Memory is still valid — just touch it
            touched_at = datetime.now(timezone.utc)
            next_access_count = chunk.access_count + 1
            if self._store.touch(chunk.id, updated_at=touched_at):
                if chunk.access_count < next_access_count:
                    chunk.access_count = next_access_count
                chunk.updated_at = touched_at
            self._total_unchanged += 1
            self._mark_reconsolidated_now(chunk.id)
            return chunk, False

        elif response.upper().startswith("CHANGED"):
            # Extract the updated content (after "CHANGED: ")
            if ":" in response:
                new_content = response.split(":", 1)[1].strip()
            else:
                # Response is just "CHANGED" without content — don't modify
                touched_at = datetime.now(timezone.utc)
                next_access_count = chunk.access_count + 1
                if self._store.touch(chunk.id, updated_at=touched_at):
                    if chunk.access_count < next_access_count:
                        chunk.access_count = next_access_count
                    chunk.updated_at = touched_at
                self._total_unchanged += 1
                self._mark_reconsolidated_now(chunk.id)
                return chunk, False

            if not new_content:
                touched_at = datetime.now(timezone.utc)
                next_access_count = chunk.access_count + 1
                if self._store.touch(chunk.id, updated_at=touched_at):
                    if chunk.access_count < next_access_count:
                        chunk.access_count = next_access_count
                    chunk.updated_at = touched_at
                self._total_unchanged += 1
                self._mark_reconsolidated_now(chunk.id)
                return chunk, False

            safe_content = new_content
            redactions: list[str] = []
            if self._write_firewall is not None:
                safety = self._write_firewall.apply(new_content)
                if not safety.allowed:
                    touched_at = datetime.now(timezone.utc)
                    next_access_count = chunk.access_count + 1
                    if self._store.touch(chunk.id, updated_at=touched_at):
                        if chunk.access_count < next_access_count:
                            chunk.access_count = next_access_count
                        chunk.updated_at = touched_at
                    self._total_unchanged += 1
                    self._mark_reconsolidated_now(chunk.id)
                    return chunk, False
                safe_content = safety.content
                redactions = [match.label for match in safety.matches]

            # Create reconsolidated version
            updated_chunk = chunk.reconsolidate(safe_content)
            # Re-embed the new content
            updated_chunk.embedding = await embed_text_async(self._embedder, safe_content)
            updated_chunk.metadata["reconsolidated_at"] = datetime.now(timezone.utc).isoformat()
            updated_chunk.metadata["reconsolidation_context"] = new_context[:200]  # Truncate
            updated_chunk.metadata["safety_redactions"] = redactions

            # Overwrite in store (same ID, new version)
            self._store.update(chunk.id, updated_chunk)
            self._total_reconsolidated += 1
            self._mark_reconsolidated_now(chunk.id)
            return updated_chunk, True

        else:
            # Unexpected response format — conservative: leave unchanged
            touched_at = datetime.now(timezone.utc)
            next_access_count = chunk.access_count + 1
            if self._store.touch(chunk.id, updated_at=touched_at):
                if chunk.access_count < next_access_count:
                    chunk.access_count = next_access_count
                chunk.updated_at = touched_at
            self._total_unchanged += 1
            self._mark_reconsolidated_now(chunk.id)
            return chunk, False

    async def process_labile_chunks(
        self, max_chunks: int | None = None
    ) -> list[tuple[MemoryChunk, bool]]:
        """
        Batch reconsolidate all queued labile chunks.

        This is designed to run asynchronously after a conversation turn,
        not blocking the main response pipeline — mirroring how biological
        reconsolidation occurs in the background after retrieval.

        Args:
            max_chunks: Maximum number of labile chunks to process in this pass.
                        Defaults to config.max_labile_chunks.

        Returns:
            List of (chunk, was_changed) tuples for each processed chunk.
        """
        if not self._labile_chunks:
            return []

        limit = max_chunks or self._config.max_labile_chunks
        to_process = list(self._labile_chunks.items())[:limit]

        results = []
        for chunk_id, (chunk, context) in to_process:
            result = await self.reconsolidate(chunk, context)
            results.append(result)
            # Remove from labile queue after processing
            self._labile_chunks.pop(chunk_id, None)

        return results

    def clear_labile_queue(self) -> int:
        """
        Clear the labile chunk queue without reconsolidating.

        Returns:
            Number of chunks that were queued and discarded.
        """
        count = len(self._labile_chunks)
        self._labile_chunks.clear()
        return count

    def get_labile_count(self) -> int:
        """Return the number of chunks currently in the labile queue."""
        return len(self._labile_chunks)

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the MutableRAG module."""
        return {
            "module": "MutableRAG",
            "enabled": self._config.enabled,
            "total_retrieved": self._total_retrieved,
            "total_reconsolidated": self._total_reconsolidated,
            "total_unchanged": self._total_unchanged,
            "labile_queue_size": len(self._labile_chunks),
        }
