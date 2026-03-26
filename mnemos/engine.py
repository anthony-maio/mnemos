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
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import MnemosConfig
from .curation import durable_memory_skip_reason
from .memory_safety import MemoryWriteFirewall
from .modules.affective import AffectiveRouter
from .modules.mutable_rag import MutableRAG
from .modules.sleep import SleepDaemon
from .modules.spreading import SpreadingActivation
from .modules.surprisal import SurprisalGate
from .types import ConsolidationResult, Interaction, MemoryChunk, ProcessResult
from .observability import log_event
from .utils.embeddings import EmbeddingProvider, embed_text_async
from .utils.llm import LLMProvider
from .utils.storage import MemoryStore

logger = logging.getLogger(__name__)


def _on_reconsolidation_done(t: asyncio.Task[Any]) -> None:
    """Log errors from background reconsolidation tasks instead of silently swallowing them."""
    if not t.cancelled() and t.exception():
        logger.error("Background reconsolidation failed: %s", t.exception())


_VALID_SCOPES = ("project", "workspace", "global")
_DEFAULT_SCOPE = "project"
_DEFAULT_SCOPE_ID = "default"
_DEFAULT_ALLOWED_SCOPES = ("project", "workspace", "global")


def _normalize_scope(scope: str | None, *, default: str = _DEFAULT_SCOPE) -> str:
    normalized = (scope or default).strip().lower()
    if normalized not in _VALID_SCOPES:
        raise ValueError(f"Invalid scope {scope!r}; expected one of {', '.join(_VALID_SCOPES)}.")
    return normalized


def _normalize_scope_id(scope: str, scope_id: str | None) -> str | None:
    if scope == "global":
        return None
    if scope_id is None:
        return _DEFAULT_SCOPE_ID
    trimmed = scope_id.strip()
    return trimmed if trimmed else _DEFAULT_SCOPE_ID


def _normalize_allowed_scopes(
    allowed_scopes: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if allowed_scopes is None:
        return _DEFAULT_ALLOWED_SCOPES
    normalized: list[str] = []
    for scope in allowed_scopes:
        norm = _normalize_scope(scope)
        if norm not in normalized:
            normalized.append(norm)
    return tuple(normalized)


def _chunk_scope_data(chunk: MemoryChunk) -> tuple[str, str | None]:
    scope_raw = chunk.metadata.get("scope")
    if isinstance(scope_raw, str):
        try:
            scope = _normalize_scope(scope_raw)
        except ValueError:
            scope = "global"
    else:
        # Backward-compatibility for chunks that predate scoped metadata.
        scope = "global"

    if scope == "global":
        return scope, None

    scope_id_raw = chunk.metadata.get("scope_id")
    if isinstance(scope_id_raw, str) and scope_id_raw.strip():
        return scope, scope_id_raw.strip()
    return scope, _DEFAULT_SCOPE_ID


def _build_scope_filter(
    *,
    current_scope: str,
    scope_id: str | None,
    allowed_scopes: tuple[str, ...],
) -> Callable[[MemoryChunk], bool]:
    def _is_allowed(chunk: MemoryChunk) -> bool:
        chunk_scope, chunk_scope_id = _chunk_scope_data(chunk)
        if chunk_scope not in allowed_scopes:
            return False
        if chunk_scope == "global":
            return True
        if scope_id is None:
            return False
        return chunk_scope_id == scope_id

    _ = current_scope  # reserved for future policy variants
    setattr(
        _is_allowed,
        "_mnemos_scope_filter",
        {
            "current_scope": current_scope,
            "scope_id": scope_id,
            "allowed_scopes": allowed_scopes,
        },
    )
    return _is_allowed


def _scope_match_boost(chunk_scope: str, current_scope: str) -> float:
    if chunk_scope == current_scope:
        return 0.25
    if chunk_scope == "workspace" and current_scope == "project":
        return 0.12
    if chunk_scope == "global":
        return 0.05
    return 0.02


def _infer_ingest_channel(interaction: Interaction) -> str:
    source = str(interaction.metadata.get("source", "")).strip().lower()
    if source == "claude_hook" or "hook_event" in interaction.metadata:
        return "hook"
    ingest_channel = str(interaction.metadata.get("ingest_channel", "")).strip().lower()
    if ingest_channel in {"hook", "manual"}:
        return ingest_channel
    return "manual"


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
        self._write_firewall = MemoryWriteFirewall(self.config.safety)

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
            write_firewall=self._write_firewall,
        )

        self.affective_router = AffectiveRouter(
            llm=self._llm,
            embedder=self._embedder,
            config=self.config.affective,
        )

        self.sleep_daemon = SleepDaemon(
            store=self._store,
            config=self.config.sleep,
            write_firewall=self._write_firewall,
        )

        self.spreading_activation = SpreadingActivation(
            embedder=self._embedder,
            config=self.config.spreading,
        )

        # Rebuild associative graph from persisted storage so retrieval
        # keeps working across process restarts.
        self._hydrate_spreading_graph_from_store()

    def _capture_channel_allowed(self, ingest_channel: str) -> bool:
        mode = self.config.governance.capture_mode
        if mode == "all":
            return True
        if mode == "manual_only":
            return ingest_channel != "hook"
        if mode == "hooks_only":
            return ingest_channel == "hook"
        return True

    def _delete_chunk_everywhere(self, chunk_id: str) -> None:
        deleted = self._store.delete(chunk_id)
        if deleted and self.spreading_activation.get_node(chunk_id) is not None:
            self.spreading_activation.remove_node(chunk_id)

    def _sync_persisted_graph_edges(self) -> None:
        for node in self.spreading_activation.get_all_nodes():
            self._store.replace_graph_neighbors(node.id, dict(node.neighbors))

    def _apply_governance(
        self,
        *,
        target_scope: str | None = None,
        target_scope_id: str | None = None,
    ) -> None:
        governance = self.config.governance
        retention_days = governance.retention_ttl_days
        max_per_scope = governance.max_chunks_per_scope
        if retention_days <= 0 and max_per_scope <= 0:
            return

        chunks = self._store.get_all()

        if retention_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            for chunk in chunks:
                if chunk.updated_at < cutoff and chunk.created_at < cutoff:
                    self._delete_chunk_everywhere(chunk.id)
            chunks = self._store.get_all()

        if max_per_scope > 0:
            grouped: dict[tuple[str, str | None], list[MemoryChunk]] = {}
            for chunk in chunks:
                scope, scope_id = _chunk_scope_data(chunk)
                if target_scope is not None and (
                    scope != target_scope or scope_id != target_scope_id
                ):
                    continue
                grouped.setdefault((scope, scope_id), []).append(chunk)

            for _, group in grouped.items():
                if len(group) <= max_per_scope:
                    continue
                ranked = sorted(
                    group,
                    key=lambda chunk: (chunk.updated_at, chunk.created_at),
                    reverse=True,
                )
                for stale in ranked[max_per_scope:]:
                    self._delete_chunk_everywhere(stale.id)

    def _hydrate_spreading_graph_from_store(self) -> None:
        """Load persisted chunks into spreading activation graph at startup."""
        spreading_cfg = self.config.spreading
        if not spreading_cfg.hydrate_on_startup:
            return

        stored_chunks = self._store.get_all()
        if not stored_chunks:
            return

        limit = spreading_cfg.startup_hydration_limit
        chunks_to_load = stored_chunks[:limit]

        for chunk in chunks_to_load:
            if self.spreading_activation.get_node(chunk.id) is None:
                self.spreading_activation.add_node_from_chunk(chunk)

        persisted_edges = self._store.get_graph_edges([chunk.id for chunk in chunks_to_load])
        if persisted_edges:
            self.spreading_activation.hydrate_edges(persisted_edges)
            self._sync_persisted_graph_edges()
        elif spreading_cfg.startup_auto_connect and len(chunks_to_load) > 1:
            self.spreading_activation.auto_connect(exclude_existing=False)
            self._sync_persisted_graph_edges()

        if self.config.debug and len(stored_chunks) > limit:
            logger.debug(
                "[MnemosEngine] Startup hydration truncated at %d chunks (store has %d).",
                limit,
                len(stored_chunks),
            )

    async def process(
        self,
        interaction: Interaction,
        *,
        scope: str = _DEFAULT_SCOPE,
        scope_id: str | None = None,
    ) -> ProcessResult:
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
        normalized_scope = _normalize_scope(scope)
        normalized_scope_id = _normalize_scope_id(normalized_scope, scope_id)

        scoped_metadata = dict(interaction.metadata)
        scoped_metadata["scope"] = normalized_scope
        if normalized_scope_id is None:
            scoped_metadata.pop("scope_id", None)
        else:
            scoped_metadata["scope_id"] = normalized_scope_id
        scoped_interaction = interaction.model_copy(update={"metadata": scoped_metadata})
        ingest_channel = _infer_ingest_channel(scoped_interaction)
        if not self._capture_channel_allowed(ingest_channel):
            return ProcessResult(
                stored=False,
                chunk=None,
                salience=0.0,
                reason=(
                    "Blocked by capture mode policy: "
                    f"{self.config.governance.capture_mode} disallows {ingest_channel} ingestion."
                ),
            )

        # Always add to episodic buffer for eventual consolidation
        # (The hippocampus stores everything briefly, even mundane inputs)
        self.sleep_daemon.add_episode(scoped_interaction)

        # Step 1: Surprisal gate — filter by prediction error
        result = await self.surprisal_gate.process(scoped_interaction)

        if not result.stored or result.chunk is None:
            return result

        safety = self._write_firewall.apply(result.chunk.content)
        if not safety.allowed:
            self._store.delete(result.chunk.id)
            return ProcessResult(
                stored=False,
                chunk=None,
                salience=result.salience,
                reason=f"Blocked by safety policy: {safety.reason}",
            )

        skip_reason = durable_memory_skip_reason(safety.content)
        if skip_reason is not None:
            self._store.delete(result.chunk.id)
            return ProcessResult(
                stored=False,
                chunk=None,
                salience=result.salience,
                reason=f"Skipped transient/noisy memory: {skip_reason}.",
            )

        if safety.content != result.chunk.content:
            result.chunk.content = safety.content
            result.chunk.embedding = await embed_text_async(self._embedder, safety.content)
            result.chunk.metadata["safety_redactions"] = [match.label for match in safety.matches]
            self._store.update(result.chunk.id, result.chunk)

        # Ensure scope tags are always present on persisted chunks.
        result.chunk.metadata["scope"] = normalized_scope
        if normalized_scope_id is None:
            result.chunk.metadata.pop("scope_id", None)
        else:
            result.chunk.metadata["scope_id"] = normalized_scope_id
        result.chunk.metadata["ingest_channel"] = ingest_channel
        result.chunk.metadata["encoding_reason"] = result.reason

        # Step 2: Affective tagging — classify and attach cognitive state
        state = await self.affective_router.classify_state(scoped_interaction)
        self.affective_router.tag_chunk(result.chunk, state)

        # Update the stored chunk with the affective tag
        self._store.update(result.chunk.id, result.chunk)

        # Step 3: Add to spreading activation graph
        self.spreading_activation.add_node_from_chunk(result.chunk)

        # Connect the new node to its strongest eligible scoped neighbors and
        # persist the resulting sparse graph when the backend supports it.
        self.spreading_activation.connect_node(result.chunk.id)
        self._sync_persisted_graph_edges()

        if self.config.debug:
            logger.debug(
                f"[MnemosEngine] Processed: stored={result.stored}, "
                f"salience={result.salience:.3f}, reason={result.reason}"
            )

        self._apply_governance(
            target_scope=normalized_scope,
            target_scope_id=normalized_scope_id,
        )

        return result

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        reconsolidate: bool = True,
        *,
        current_scope: str = _DEFAULT_SCOPE,
        scope_id: str | None = None,
        allowed_scopes: tuple[str, ...] | list[str] | None = None,
    ) -> list[MemoryChunk]:
        """
        Retrieve memories through the contextual retrieval pipeline.

        Pipeline:
        1. Embed the query and gather scoped semantic candidates
        2. Optionally expand with SpreadingActivation when returning multiple results
        3. Optionally re-rank with AffectiveRouter when returning multiple results
        4. Flag retrieved chunks as labile and persist access metadata

        Args:
            query: The query string to search for.
            top_k: Number of results to return.
            reconsolidate: Whether to run reconsolidation on labile chunks
                           after retrieval (default True, runs in background).

        Returns:
            Top-k MemoryChunks ranked by the active retrieval pipeline.
        """
        start = time.perf_counter()
        normalized_current_scope = _normalize_scope(current_scope)
        normalized_scope_id = _normalize_scope_id(normalized_current_scope, scope_id)
        normalized_allowed_scopes = _normalize_allowed_scopes(allowed_scopes)
        scope_filter = _build_scope_filter(
            current_scope=normalized_current_scope,
            scope_id=normalized_scope_id,
            allowed_scopes=normalized_allowed_scopes,
        )

        # Step 1: Embed query once for every downstream retrieval stage.
        query_embedding = await embed_text_async(self._embedder, query)

        # Step 2: SpreadingActivation — find associatively connected neighbors
        # Single-result lookups are better served by the direct semantic path.
        # Associative expansion adds fixed overhead without improving top-1
        # quality on our scoped coding-agent workloads.
        use_spreading = top_k > 1 and self.spreading_activation.get_node_count() > 1
        activated_nodes = (
            self.spreading_activation.retrieve(
                query_embedding,
                top_k=top_k * 3,  # Get larger pool for re-ranking
            )
            if use_spreading
            else []
        )
        candidate_pool_k = max(top_k * 6, 20)
        baseline_chunks = self._store.retrieve(
            query_embedding,
            top_k=candidate_pool_k,
            filter_fn=scope_filter,
        )

        # Gather IDs of activated nodes to boost them in final scoring
        activated_ids: set[str] = {node.id for node in activated_nodes}
        query_lower = query.lower()
        recency_hint = any(
            token in query_lower for token in ("current", "now", "latest", "today", "recent")
        )

        # Step 3: AffectiveRouter — retrieve and re-rank from store
        # State-dependent reranking is most useful when returning multiple
        # options. For top-1, the direct semantic/scoped path is both faster
        # and more deterministic.
        use_affective = top_k > 1
        affective_chunks: list[MemoryChunk] = []
        if use_affective:
            query_interaction = Interaction(role="user", content=query)
            current_state = await self.affective_router.classify_state(query_interaction)
            affective_chunks = await self.affective_router.retrieve(
                query=query,
                current_state=current_state,
                store=self._store,
                top_k=top_k * 2,  # Slightly larger pool
                query_embedding=query_embedding,
                candidates=baseline_chunks,
            )

        # Merge activated nodes: boost chunks that appear in activation graph
        chunk_scores: dict[str, tuple[float, MemoryChunk]] = {}
        now = datetime.now(timezone.utc)
        for i, chunk in enumerate(baseline_chunks):
            semantic_rank_score = 1.0 - (i / max(len(baseline_chunks), 1))
            base_score = 0.9 * semantic_rank_score
            chunk_scope, _ = _chunk_scope_data(chunk)
            base_score += _scope_match_boost(chunk_scope, normalized_current_scope)
            if recency_hint:
                age_seconds = max((now - chunk.updated_at).total_seconds(), 0.0)
                recency_weight = 1.0 / (1.0 + (age_seconds / 86400.0))
                base_score = (0.35 * semantic_rank_score) + (0.85 * recency_weight)
                base_score += _scope_match_boost(chunk_scope, normalized_current_scope)
            chunk_scores[chunk.id] = (base_score, chunk)

        for i, chunk in enumerate(affective_chunks):
            # Base score: inverse rank from affective retrieval
            base_score = 0.4 * (1.0 - (i / max(len(affective_chunks), 1)))
            chunk_scope, _ = _chunk_scope_data(chunk)
            base_score += _scope_match_boost(chunk_scope, normalized_current_scope)
            # Activation boost: chunks in the spreading graph get a 20% boost
            if chunk.id in activated_ids:
                base_score *= 1.2
            if recency_hint:
                age_seconds = max((now - chunk.updated_at).total_seconds(), 0.0)
                recency_weight = 1.0 / (1.0 + (age_seconds / 86400.0))
                base_score += 0.35 * recency_weight
            existing = chunk_scores.get(chunk.id)
            if existing is None:
                chunk_scores[chunk.id] = (base_score, chunk)
            else:
                chunk_scores[chunk.id] = (existing[0] + base_score, chunk)

        # Also include any activated nodes not in the affective results
        missing_ids = activated_ids - set(chunk_scores.keys())
        if missing_ids:
            for node in activated_nodes:
                if node.id not in missing_ids:
                    continue
                candidate = self._store.get(node.id)
                if candidate is None or not scope_filter(candidate):
                    continue
                chunk = candidate
                # Score based on activation energy
                score = node.energy * 0.5  # Activation-only score (lower weight)
                chunk_scope, _ = _chunk_scope_data(chunk)
                score += _scope_match_boost(chunk_scope, normalized_current_scope)
                chunk_scores[node.id] = (score, chunk)

        # Sort by score and take top_k
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )
        if recency_hint and len(sorted_chunks) > 1:
            head_count = min(len(sorted_chunks), max(top_k * 2, 2))
            recency_head = sorted(
                sorted_chunks[:head_count],
                key=lambda x: x[1].updated_at,
                reverse=True,
            )
            sorted_chunks = recency_head + sorted_chunks[head_count:]
        final_chunks = [chunk for _, chunk in sorted_chunks[:top_k]]

        # Step 4: MutableRAG — flag as labile and optionally reconsolidate
        for chunk in final_chunks:
            touched_at = datetime.now(timezone.utc)
            next_access_count = chunk.access_count + 1
            if self._store.touch(chunk.id, updated_at=touched_at):
                if chunk.access_count < next_access_count:
                    chunk.access_count = next_access_count
                chunk.updated_at = touched_at
            self.mutable_rag.mark_labile(chunk, query)

        if reconsolidate and final_chunks:
            # Run reconsolidation as a background task (non-blocking)
            task = asyncio.create_task(self.mutable_rag.process_labile_chunks())
            task.add_done_callback(_on_reconsolidation_done)

        if self.config.debug:
            logger.debug(
                f"[MnemosEngine] Retrieved {len(final_chunks)} chunks for query: {query!r}"
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        log_event(
            "mnemos.retrieval_latency",
            latency_ms=round(latency_ms, 3),
            result_count=len(final_chunks),
            top_k=top_k,
            scope=normalized_current_scope,
            scope_id=normalized_scope_id,
            allowed_scopes=",".join(normalized_allowed_scopes),
            store_backend=type(self._store).__name__,
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
            self.spreading_activation.auto_connect(exclude_existing=False)
            self._sync_persisted_graph_edges()

        self._apply_governance()

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
