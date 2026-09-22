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
        Idle trigger → SleepDaemon (hippocampal-neocortical transfer,
        optionally recall-gated before semantic writes)

This is the main entry point for library consumers. Instantiate MnemosEngine
and call process()/retrieve()/consolidate() to use the full pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from .config import MnemosConfig
from .curation import durable_memory_skip_reason
from .judgment import Judge
from .memory_safety import MemoryWriteFirewall
from .scopes import ScopeContext, ScopeRegistry, context_metadata, scope_allows
from .modules.affective import AffectiveRouter
from .modules.mutable_rag import MutableRAG
from .modules.sleep import SleepDaemon
from .modules.spreading import SpreadingActivation
from .modules.surprisal import SurprisalGate
from .types import ActivationNode, ConsolidationResult, Interaction, MemoryChunk, ProcessResult
from .observability import log_event
from .retrieval import RetrievedChunks, score_candidate, context_token_upper_bound
from .utils.embeddings import EmbeddingProvider, embed_text_async
from .utils.llm import LLMProvider
from .utils.reliability import MnemosTransientError
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
            scope = "unassigned"
    else:
        # Backward-compatibility for chunks that predate scoped metadata.
        scope = "unassigned"

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
        if chunk.metadata.get("user_profile_id") is not None:
            return False
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
        scope_registry: ScopeRegistry | None = None,
        require_scope_context: bool = False,
        *,
        judge: Judge | None = None,
        owns_providers: bool = False,
    ) -> None:
        from .judgment import UnavailableJudge
        from .utils.llm import UnavailableLLMProvider
        from .utils.embeddings import SimpleEmbeddingProvider
        from .utils.storage import InMemoryStore

        self.config = config or MnemosConfig()
        self._scope_registry = scope_registry
        self._require_scope_context = require_scope_context
        self._retrieval_traces: dict[tuple[Any, ...], dict[str, Any]] = {}

        # Wire up providers with defaults for zero-config startup
        self._llm = llm if llm is not None else UnavailableLLMProvider()
        self._embedder = (
            embedder
            if embedder is not None
            else SimpleEmbeddingProvider(dim=self.config.surprisal.embedding_dim)
        )
        self._store = store or InMemoryStore()
        # Typed judgments are optional. Without a configured judge every
        # judged path is off and the generative paths are the only ones.
        self._judge: Judge = judge if judge is not None else UnavailableJudge()
        self._owns_llm = owns_providers or llm is None
        self._owns_embedder = owns_providers or embedder is None
        self._owns_judge = owns_providers or judge is None
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed_resources: set[str] = set()
        self._write_firewall = MemoryWriteFirewall(self.config.safety)

        # Initialize all five modules
        self.surprisal_gate = SurprisalGate(
            llm=self._llm,
            embedder=self._embedder,
            store=self._store,
            config=self.config.surprisal,
            write_firewall=self._write_firewall,
            judge=self._judge,
        )

        self.mutable_rag = MutableRAG(
            llm=self._llm,
            embedder=self._embedder,
            store=self._store,
            config=self.config.mutable_rag,
            write_firewall=self._write_firewall,
            judge=self._judge,
            judgment_config=self.config.judgment,
        )

        self.affective_router = AffectiveRouter(
            llm=self._llm,
            embedder=self._embedder,
            config=self.config.affective,
            judge=self._judge,
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
        bind_graph = getattr(self._store, "bind_graph_contract", None)
        if callable(bind_graph):
            bind_graph(
                {
                    **self.spreading_activation.persistence_contract(),
                    "edge_threshold": self.config.spreading.auto_connect_threshold,
                    "max_neighbors": self.config.spreading.max_neighbors_per_node,
                }
            )
        self._hydrate_spreading_graph_from_store()

    async def aclose(self) -> None:
        """Release owned providers and storage after engine operations have stopped.

        Configured runtimes own their providers. Library-injected providers
        remain caller-owned unless ``owns_providers=True`` was supplied.
        The local service cancels and drains its operations before calling this.
        """
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close_resources())
        task = self._close_task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Cancelling an await cannot stop a synchronous provider close
                # in a worker thread. Drain the shared cleanup before returning
                # cancellation or letting another caller retry failed resources.
                cancelled = True
            except Exception:
                break
        if cancelled:
            if not task.cancelled() and task.exception() is not None:
                logger.error("Provider cleanup failed while its caller was cancelled")
            raise asyncio.CancelledError
        task.result()

    async def _close_resources(self) -> None:
        async with self._close_lock:
            error: Exception | None = None
            for name, provider, owned in (
                ("embedding", self._embedder, self._owns_embedder),
                ("llm", self._llm, self._owns_llm),
                ("judge", self._judge, self._owns_judge),
            ):
                if not owned or name in self._closed_resources:
                    continue
                try:
                    aclose = getattr(provider, "aclose", None)
                    close = getattr(provider, "close", None)
                    if callable(aclose):
                        await aclose()
                    elif callable(close):
                        # Embedding requests may still be finishing in worker
                        # threads after their awaiting task was cancelled.
                        await asyncio.to_thread(close)
                    self._closed_resources.add(name)
                except Exception as exc:
                    if error is None:
                        error = exc
            if "store" not in self._closed_resources:
                try:
                    close_store = getattr(self._store, "close", None)
                    if callable(close_store):
                        close_store()
                    self._closed_resources.add("store")
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error

    def rebuild_graph_batch(self, limit: int = 10) -> int:
        cursor_reader = getattr(self._store, "graph_rebuild_cursor", None)
        if not self.config.modules.spreading or not callable(cursor_reader):
            return 0
        cursor = cursor_reader()
        if cursor is None:
            return 0
        chunks = self._store.list_chunks(limit=limit, after_id=cursor)
        for chunk in chunks:
            self.refresh_graph_chunk(chunk, _rebuild_source_ids={chunk.id})
        status_reader = getattr(self._store, "graph_derivation_status", None)
        status = status_reader() if callable(status_reader) else {"pending_count": 0}
        if int(status.get("pending_count", 0)) == 0:
            self._store.set_graph_rebuild_cursor(None)
        elif chunks:
            self._store.set_graph_rebuild_cursor(chunks[-1].id)
        else:
            # An invalidated source can sort before the previous cursor.
            # Keep recovery runnable instead of leaving pending work stranded.
            self._store.set_graph_rebuild_cursor("")
        return len(chunks)

    def _validated_context_metadata(
        self, context: ScopeContext | None, scope: str, scope_id: str | None
    ) -> dict[str, Any] | None:
        if context is None:
            if self._require_scope_context:
                raise ValueError("A resolved scope context is required")
            return None
        if self._scope_registry is None:
            raise ValueError("A scope registry is required to validate membership")
        self._scope_registry.validate_context(context)
        metadata = context_metadata(context, scope)
        if scope_id is not None and scope_id != metadata["scope_id"]:
            raise ValueError("Explicit scope_id conflicts with the resolved scope context")
        return metadata

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
            all_chunks = self._store.get_all()
            records = [
                ActivationNode(
                    id=item.id,
                    content=item.content,
                    embedding=item.embedding,
                    metadata=dict(item.metadata),
                )
                for item in all_chunks
                if item.embedding is not None
            ]
            self.spreading_activation.set_authoritative_snapshot(records)
            affected = self.spreading_activation.derive_authoritative_node(
                chunk_id, deleted_id=chunk_id
            )
            batch = getattr(self._store, "replace_graph_preferences_batch", None)
            if callable(batch):
                batch(
                    {
                        source_id: self.spreading_activation.get_preferences().get(source_id, {})
                        for source_id in affected
                        if source_id != chunk_id
                    },
                    builder_version=SpreadingActivation.EDGE_BUILDER_VERSION,
                )
            self.spreading_activation.remove_node(chunk_id)
            self._sync_persisted_graph_edges()

    def _sync_persisted_graph_edges(self) -> None:
        preference_reader = getattr(self._store, "get_graph_preferences", None)
        projector = getattr(self._store, "project_graph_edges", None)
        if callable(preference_reader) and callable(projector):
            status_reader = getattr(self._store, "graph_derivation_status", None)
            status = status_reader() if callable(status_reader) else None
            # SQLite graph state is managed by directed preferences even when
            # every completed source has an empty map. Projecting an empty
            # set clears stale semantic edges without entering legacy mode.
            if status is not None and status.get("managed_preferences", False):
                projector()
                self.spreading_activation.refresh_projected_edges()
                return
            preferences = preference_reader()
            if preferences:
                projector()
                self.spreading_activation.refresh_projected_edges()
                return
        for node in self.spreading_activation.get_all_nodes():
            self._store.replace_graph_neighbors(node.id, dict(node.neighbors))

    def refresh_graph_chunk(
        self, chunk: MemoryChunk, *, _rebuild_source_ids: set[str] | None = None
    ) -> None:
        """Refresh one fact from bounded scoped neighbors, including unhydrated rows."""
        for absorbed_id in chunk.metadata.get("absorbed_captured_ids", []):
            if self.store.get(absorbed_id) is None:
                self.spreading_activation.remove_node(absorbed_id)
        if not self.config.modules.spreading or not chunk.embedding:
            return
        scope, scope_id = _chunk_scope_data(chunk)
        profile_id = chunk.metadata.get("user_profile_id")
        permitted: Callable[[MemoryChunk], bool]
        if scope == "global":

            def global_permitted(candidate: MemoryChunk) -> bool:
                return candidate.metadata.get(
                    "user_profile_id"
                ) == profile_id and candidate.metadata.get("scope") in {
                    "project",
                    "workspace",
                    "global",
                }

            permitted = global_permitted
            spec = {
                "allowed_scopes": ("project", "workspace", "global"),
                "all_profile_scopes": True,
            }
            if profile_id is not None:
                spec["user_profile_id"] = profile_id
            setattr(permitted, "_mnemos_scope_filter", spec)
        elif profile_id is None:
            permitted = _build_scope_filter(
                current_scope=scope, scope_id=scope_id, allowed_scopes=(scope, "global")
            )
        else:
            member_workspace_id = (
                scope_id if scope == "workspace" else chunk.metadata.get("workspace_id")
            )

            def partition_permitted(candidate: MemoryChunk) -> bool:
                return candidate.metadata.get("user_profile_id") == profile_id and (
                    candidate.metadata.get("scope") == "global"
                    or (
                        candidate.metadata.get("scope") == scope
                        and candidate.metadata.get("scope_id") == scope_id
                    )
                    or (
                        member_workspace_id is not None
                        and (
                            (
                                candidate.metadata.get("scope") == "workspace"
                                and candidate.metadata.get("scope_id") == member_workspace_id
                            )
                            or (
                                scope == "workspace"
                                and candidate.metadata.get("scope") == "project"
                                and candidate.metadata.get("workspace_id") == member_workspace_id
                            )
                        )
                    )
                )

            permitted = partition_permitted
            setattr(
                permitted,
                "_mnemos_scope_filter",
                {
                    "user_profile_id": profile_id,
                    "project_id": scope_id if scope == "project" else None,
                    "workspace_id": member_workspace_id,
                    "member_workspace_id": member_workspace_id if scope == "workspace" else None,
                    "allowed_scopes": ("project", "workspace", "global"),
                },
            )
        nearby = self._store.retrieve(
            chunk.embedding,
            top_k=self.config.spreading.max_neighbors_per_node * 4 + 1,
            filter_fn=permitted,
        )
        for item in nearby:
            self.spreading_activation.add_node_from_chunk(item)
        self.spreading_activation.add_node_from_chunk(chunk)
        # Derivation sees the complete authoritative snapshot. Scope
        # compatibility is checked for each source-target pair in the graph
        # module; restricting this snapshot to the changed partition would
        # silently change global and workspace preferences.
        authoritative = self._store.get_all()
        self.spreading_activation.set_authoritative_snapshot(
            [
                ActivationNode(
                    id=item.id,
                    content=item.content,
                    embedding=item.embedding,
                    metadata=dict(item.metadata),
                )
                for item in authoritative
                if item.embedding is not None
            ]
        )
        graph_status_reader = getattr(self._store, "graph_derivation_status", None)
        graph_status = graph_status_reader() if callable(graph_status_reader) else {}
        managed_preferences = bool(graph_status.get("managed_preferences", False))
        if managed_preferences:
            # The durable completed sources are authoritative across partial
            # rebuilds, restarts, corrections, and direct store invalidations.
            # Never promote a stale source's remaining preference rows merely
            # because a new target happens to fit its old cutoff.
            self.spreading_activation.hydrate_preferences(
                self._store.get_graph_preferences(completed_only=True)
            )
        preferences = self.spreading_activation.get_preferences()
        if _rebuild_source_ids is not None:
            affected = self.spreading_activation.derive_authoritative_nodes(_rebuild_source_ids)
        elif managed_preferences:
            # New facts require one source scan plus one cutoff comparison per
            # completed source. Dirty sources left by correction/restart are
            # recomputed in bounded groups; maintenance drains any remainder.
            pending_ids = sorted(
                item.id for item in authoritative if item.embedding and item.id not in preferences
            )
            refresh_limit = self.config.spreading.max_neighbors_per_node * 4 + 1
            affected = self.spreading_activation.derive_authoritative_nodes(
                {chunk.id, *pending_ids[:refresh_limit]}
            )
            affected |= self.spreading_activation.derive_authoritative_node(chunk.id)
        else:
            affected = self.spreading_activation.derive_authoritative_node(chunk.id)
        batch = getattr(self._store, "replace_graph_preferences_batch", None)
        if managed_preferences and callable(batch):
            preferences = self.spreading_activation.get_preferences()
            batch(
                {source_id: preferences.get(source_id, {}) for source_id in affected},
                builder_version=SpreadingActivation.EDGE_BUILDER_VERSION,
            )
        self._sync_persisted_graph_edges()

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

        stored_chunks = self._store.list_chunks(limit=spreading_cfg.startup_hydration_limit)
        if not stored_chunks:
            return

        authoritative = self._store.get_all()
        self.spreading_activation.set_authoritative_snapshot(
            [
                ActivationNode(
                    id=item.id,
                    content=item.content,
                    embedding=item.embedding,
                    metadata=dict(item.metadata),
                )
                for item in authoritative
                if item.embedding is not None
            ]
        )

        limit = spreading_cfg.startup_hydration_limit
        chunks_to_load = stored_chunks[:limit]

        for chunk in chunks_to_load:
            if self.spreading_activation.get_node(chunk.id) is None:
                self.spreading_activation.add_node_from_chunk(chunk)

        cursor_reader = getattr(self._store, "graph_rebuild_cursor", None)
        cursor = cursor_reader() if callable(cursor_reader) else None
        preference_reader = getattr(self._store, "get_graph_preferences", None)
        preferences = preference_reader() if callable(preference_reader) else {}
        status_reader = getattr(self._store, "graph_derivation_status", None)
        graph_status = status_reader() if callable(status_reader) else {}
        managed_graph_pending = (
            graph_status.get("managed_preferences", False) and graph_status.get("ready") is not True
        )
        if cursor == "" or managed_graph_pending:
            self.spreading_activation.set_graph_pending()
        elif (
            graph_status.get("managed_preferences", False)
            and graph_status.get("ready") is True
            and not graph_status.get("legacy_graph", False)
        ):
            self.spreading_activation.hydrate_preferences(preferences)
            persisted_edges = self._store.get_graph_edges([chunk.id for chunk in chunks_to_load])
            if persisted_edges:
                self.spreading_activation.hydrate_edges(persisted_edges)
        else:
            persisted_edges = self._store.get_graph_edges([chunk.id for chunk in chunks_to_load])
            if persisted_edges:
                self.spreading_activation.hydrate_edges(persisted_edges)
            elif (
                spreading_cfg.startup_auto_connect
                and len(chunks_to_load) > 1
                and (not callable(status_reader) or graph_status.get("ready") is True)
            ):
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
        context: ScopeContext | None = None,
        explicit_remember: bool = False,
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
        identity_metadata = self._validated_context_metadata(context, normalized_scope, scope_id)
        if identity_metadata is not None:
            normalized_scope_id = identity_metadata["scope_id"]
        elif interaction.metadata.get("user_profile_id") is not None:
            raise ValueError("Profile-scoped writes require a validated scope context")

        scoped_metadata = dict(interaction.metadata)
        if identity_metadata is not None:
            scoped_metadata.update(identity_metadata)
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

        safety = self._write_firewall.apply(scoped_interaction.content)
        if not safety.allowed:
            return ProcessResult(
                stored=False,
                chunk=None,
                salience=0.0,
                reason=f"Blocked by safety policy: {safety.reason}",
            )

        safe_metadata = dict(scoped_interaction.metadata)
        if safety.content != scoped_interaction.content:
            safe_metadata["safety_redactions"] = [match.label for match in safety.matches]
        safe_interaction = scoped_interaction.model_copy(
            update={"content": safety.content, "metadata": safe_metadata}
        )

        skip_reason = durable_memory_skip_reason(safe_interaction.content)
        if skip_reason is not None and not explicit_remember:
            return ProcessResult(
                stored=False,
                chunk=None,
                salience=0.0,
                reason=f"Skipped transient/noisy memory: {skip_reason}.",
            )

        # Always add to episodic buffer for eventual consolidation
        # (The hippocampus stores everything briefly, even mundane inputs)
        self.sleep_daemon.add_episode(safe_interaction)

        # Step 1: Surprisal gate — filter by prediction error
        result = await self.surprisal_gate.process(safe_interaction, force_encode=explicit_remember)

        if not result.stored or result.chunk is None:
            return result

        # Ensure scope tags are always present on persisted chunks.
        result.chunk.metadata["scope"] = normalized_scope
        if normalized_scope_id is None:
            result.chunk.metadata.pop("scope_id", None)
        else:
            result.chunk.metadata["scope_id"] = normalized_scope_id
        result.chunk.metadata["ingest_channel"] = ingest_channel
        result.chunk.metadata["encoding_reason"] = result.reason

        # Step 2: Affective tagging — classify and attach cognitive state
        state = await self.affective_router.classify_state(safe_interaction)
        self.affective_router.tag_chunk(result.chunk, state)

        # Update the stored chunk with the affective tag
        result.chunk = self._store.upsert_canonical(result.chunk)

        # Step 3: Add to spreading activation graph
        self.spreading_activation.add_node_from_chunk(result.chunk)

        # Connect the new node to its strongest eligible scoped neighbors and
        # persist the resulting sparse graph when the backend supports it.
        self.refresh_graph_chunk(result.chunk)

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

    async def remember(
        self,
        content: str,
        *,
        context: ScopeContext | None = None,
        scope: str = "project",
        scope_id: str | None = None,
    ) -> ProcessResult:
        """Honor an explicit memory write while preserving safety and scope checks."""
        return await self.process(
            Interaction(role="user", content=content),
            context=context,
            scope=scope,
            scope_id=scope_id,
            explicit_remember=True,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        reconsolidate: bool = True,
        *,
        current_scope: str = _DEFAULT_SCOPE,
        scope_id: str | None = None,
        allowed_scopes: tuple[str, ...] | list[str] | None = None,
        context: ScopeContext | None = None,
        profile: str | None = None,
        max_context_tokens: int = 1200,
        view: Literal["memory", "captured_evidence"] = "memory",
    ) -> list[MemoryChunk]:
        """
        Retrieve memories through the contextual retrieval pipeline.

        Pipeline:
        1. Embed the query and classify state when eligible candidates exist
        2. Gather current scoped candidates after asynchronous provider work
        3. Optionally expand the graph and apply affective ranking, including top-1
        4. Enforce the context budget and persist access metadata

        Args:
            query: The query string to search for.
            top_k: Number of results to return.
            reconsolidate: Retained for compatibility. Review requires new evidence
                           or an explicit correction, never a retrieval query.

        Returns:
            Top-k MemoryChunks ranked by the active retrieval pipeline.
        """
        if view != "memory" and view != "captured_evidence":
            raise ValueError("Unknown retrieval view")
        start = time.perf_counter()
        normalized_current_scope = _normalize_scope(current_scope)
        normalized_scope_id = _normalize_scope_id(normalized_current_scope, scope_id)
        normalized_allowed_scopes = _normalize_allowed_scopes(allowed_scopes)
        identity_metadata = self._validated_context_metadata(
            context, normalized_current_scope, scope_id
        )
        if identity_metadata is not None:
            normalized_scope_id = identity_metadata["scope_id"]
            if allowed_scopes is None:
                normalized_allowed_scopes = ("project", "global")
        scope_filter = _build_scope_filter(
            current_scope=normalized_current_scope,
            scope_id=normalized_scope_id,
            allowed_scopes=normalized_allowed_scopes,
        )
        if context is not None:

            def scope_filter(chunk: MemoryChunk) -> bool:
                return scope_allows(chunk.metadata, context, normalized_allowed_scopes)

            setattr(
                scope_filter,
                "_mnemos_scope_filter",
                {
                    "user_profile_id": context.user_profile_id,
                    "project_id": context.project_id,
                    "workspace_id": context.workspace_id,
                    "allowed_scopes": normalized_allowed_scopes,
                },
            )

        base_scope_filter = scope_filter

        def _view_filter(chunk: MemoryChunk) -> bool:
            is_captured = chunk.metadata.get("source") == "captured_passthrough"
            in_view = is_captured if view == "captured_evidence" else not is_captured
            return base_scope_filter(chunk) and in_view

        setattr(
            _view_filter,
            "_mnemos_scope_filter",
            {
                **getattr(base_scope_filter, "_mnemos_scope_filter"),
                "source_view": view,
            },
        )
        scope_filter = _view_filter

        if top_k <= 0 or max_context_tokens <= 0 or not query.strip():
            return []
        selected_profile = profile or self.config.retrieval.profile
        if selected_profile not in {"baseline", "biomimetic"}:
            raise ValueError("Unknown retrieval profile")
        candidate_pool_k = min(max(top_k * 6, 20), self.config.retrieval.max_candidates)
        lexical_search = getattr(self._store, "retrieve_lexical", None)
        diagnostics = {}
        query_embedding: list[float] | None
        try:
            query_embedding = await embed_text_async(self._embedder, query, role="query")
        except MnemosTransientError as exc:
            if not callable(lexical_search):
                raise
            query_embedding = None
            diagnostics = {
                "mode": "lexical_only",
                "reason": "embedding_unavailable",
                "error_type": type(exc).__name__,
                "semantic_status": "not_measured",
            }
            log_event(
                "mnemos.retrieval_degraded",
                mode="lexical_only",
                reason="embedding_unavailable",
                error_type=type(exc).__name__,
            )

        def collect_candidates() -> tuple[list[MemoryChunk], str, set[str], dict[str, MemoryChunk]]:
            baseline = (
                self._store.retrieve(
                    query_embedding, top_k=candidate_pool_k, filter_fn=scope_filter
                )
                if query_embedding is not None
                else []
            )
            backend = (
                getattr(self._store, "last_retrieval_backend", "embedded_cosine")
                if query_embedding is not None
                else "lexical_only"
            )
            lexical = (
                lexical_search(query, top_k=candidate_pool_k, filter_fn=scope_filter)
                if callable(lexical_search)
                else []
            )
            merged = {chunk.id: chunk for chunk in baseline}
            merged.update((chunk.id, chunk) for chunk in lexical)
            return baseline, backend, {chunk.id for chunk in lexical}, merged

        baseline_chunks, retrieval_backend, lexical_ids, candidates = collect_candidates()
        use_affective = (
            selected_profile == "biomimetic"
            and query_embedding is not None
            and self.config.retrieval.affective_enabled
            and self.config.modules.affective
        )
        current_state = None
        state_source = self.config.retrieval.query_state_source
        if use_affective and candidates and state_source == "recent_state":
            query_metadata = identity_metadata or {
                "scope": normalized_current_scope,
                "scope_id": normalized_scope_id,
            }
            # No model call on the recall path: the partition's last encoded
            # state stands in for the query's.
            current_state = self.affective_router.get_current_state(query_metadata)
        elif use_affective and candidates and state_source == "classify":
            query_metadata = identity_metadata or {
                "scope": normalized_current_scope,
                "scope_id": normalized_scope_id,
            }
            current_state = await self.affective_router.classify_state(
                Interaction(role="user", content=query, metadata=query_metadata)
            )
            # Corrections, forgets, and new facts may commit while the classifier
            # awaits its provider. Gather a fresh candidate set before graph,
            # scoring, and budgeting; the owner-loop path has no further awaits.
            baseline_chunks, retrieval_backend, lexical_ids, candidates = collect_candidates()
        graph_requested = (
            selected_profile == "biomimetic"
            and self.config.retrieval.spreading_enabled
            and self.config.modules.spreading
        )
        graph_status_reader = getattr(self._store, "graph_derivation_status", None)
        graph_status = (
            graph_status_reader()
            if graph_requested and callable(graph_status_reader)
            else {"ready": True}
        )
        graph_ready = bool(graph_status.get("ready", True))
        use_spreading = graph_requested and graph_ready and query_embedding is not None
        activation = {}
        if use_spreading and baseline_chunks and query_embedding is not None:
            # Refresh semantic seeds from authoritative rows, including corrections.
            for chunk in baseline_chunks:
                node = self.spreading_activation.get_node(chunk.id)
                if (
                    node is None
                    or node.content != chunk.content
                    or node.embedding != chunk.embedding
                    or node.metadata != chunk.metadata
                ):
                    self.spreading_activation.add_node_from_chunk(chunk)
            neighborhood = getattr(self._store, "get_graph_neighbors", None)
            if callable(neighborhood):
                frontier = [chunk.id for chunk in baseline_chunks]
                visited = set(frontier)
                for _hop in range(self.config.spreading.max_hops):
                    next_frontier = []
                    for source_id in frontier:
                        remaining = self.config.spreading.max_visited - len(visited)
                        if remaining <= 0:
                            break
                        for neighbor, weight in neighborhood(
                            source_id, filter_fn=scope_filter, limit=min(remaining, 32)
                        ):
                            existing = self.spreading_activation.get_node(neighbor.id)
                            if (
                                existing is None
                                or existing.content != neighbor.content
                                or existing.embedding != neighbor.embedding
                            ):
                                self.spreading_activation.add_node_from_chunk(neighbor)
                            self.spreading_activation.add_edge(source_id, neighbor.id, weight)
                            if neighbor.id not in visited:
                                visited.add(neighbor.id)
                                next_frontier.append(neighbor.id)
                    frontier = next_frontier
                    if not frontier:
                        break
            nodes = self.spreading_activation.retrieve(
                query_embedding,
                top_k=candidate_pool_k,
                filter_fn=lambda node: scope_filter(
                    MemoryChunk(id=node.id, content=node.content, metadata=node.metadata)
                ),
                seed_ids=[chunk.id for chunk in baseline_chunks],
                max_visited=self.config.spreading.max_visited,
            )
            for node in nodes:
                candidate = self._store.get(node.id)
                if candidate is None or not scope_filter(candidate):
                    continue
                candidates[candidate.id] = candidate
                activation[candidate.id] = node.energy

        scored: list[tuple[dict[str, Any], MemoryChunk]] = []
        for rank, chunk in enumerate(candidates.values(), start=1):
            if query_embedding is None:
                scored.append(
                    (
                        {
                            "semantic": None,
                            "state_match": None,
                            "graph_energy": 0,
                            "graph_contribution": 0,
                            "total": None,
                            "state_status": "not_measured",
                            "lexical_rank": rank,
                        },
                        chunk,
                    )
                )
                continue
            score = score_candidate(
                chunk,
                query_embedding,
                current_state=current_state,
                affective=use_affective,
                similarity_weight=self.config.affective.weight_similarity,
                state_weight=self.config.affective.weight_state,
                graph_energy=activation.get(chunk.id, 0),
                graph_weight=self.config.retrieval.graph_weight if use_spreading else 0,
            )
            # State or graph energy cannot turn an irrelevant fact into an answer.
            if score.semantic < self.config.retrieval.relevance_floor:
                continue
            scored.append((score.model_dump(), chunk))
        if query_embedding is not None:
            scored.sort(
                key=lambda item: (
                    -item[0]["total"],
                    _chunk_scope_data(item[1])[0] != normalized_current_scope,
                    item[1].id,
                )
            )
        final_chunks = []
        remaining_budget = max_context_tokens
        trace_key = (
            query.strip(),
            normalized_current_scope,
            normalized_scope_id,
            normalized_allowed_scopes,
            context.user_profile_id if context else None,
            context.session_id if context else None,
            view,
        )
        trace_rows = {}
        for score_details, chunk in scored:
            trace_rows[chunk.id] = {
                **score_details,
                "version": chunk.version,
                "profile": selected_profile,
                "scope_match": True,
                "semantic_rank": next(
                    (i + 1 for i, item in enumerate(baseline_chunks) if item.id == chunk.id), None
                ),
                "selected": False,
                "status": "degraded" if diagnostics else "observed",
            }
            returned = chunk.model_copy(deep=True)
            returned.metadata["retrieval_trace"] = {
                "retrieval_mode": "lexical_only" if diagnostics else "semantic_and_lexical"
            }
            cost = context_token_upper_bound(returned)
            if cost > remaining_budget:
                continue
            returned.metadata["retrieval_trace"] = {
                **score_details,
                **returned.metadata["retrieval_trace"],
                "profile": selected_profile,
                "scope_match": True,
                "context_token_upper_bound": cost,
                "token_count_method": "utf8_bytes_upper_bound",
                "graph_status": (
                    "not_used"
                    if diagnostics
                    else (
                        "enabled"
                        if use_spreading
                        else ("pending" if graph_requested and not graph_ready else "disabled")
                    )
                ),
                "candidate_backend": retrieval_backend,
                "state_source": state_source if use_affective else "not_used",
                "lexical_candidate": chunk.id in lexical_ids,
                "lexical_status": (
                    "fallback"
                    if diagnostics
                    else "candidate_union" if callable(lexical_search) else "unavailable"
                ),
            }
            final_chunks.append(returned)
            trace_rows[chunk.id].update(returned.metadata["retrieval_trace"])
            trace_rows[chunk.id]["selected"] = True
            remaining_budget -= cost
            if len(final_chunks) == top_k:
                break
        # Persist this retrieval as one durable metadata update. A top-five
        # result should not require five independent FULL-synchronous commits.
        self._store.mark_retrieved_many(
            [chunk.id for chunk in final_chunks], retrieved_at=datetime.now(timezone.utc)
        )
        refreshed_chunks = []
        for returned in final_chunks:
            refreshed = self._store.get(returned.id)
            if refreshed is not None:
                refreshed = refreshed.model_copy(deep=True)
                refreshed.metadata["retrieval_trace"] = returned.metadata["retrieval_trace"]
                refreshed_chunks.append(refreshed)
        final_chunks = refreshed_chunks
        # Retained for call compatibility; review requires new evidence or a correction.
        _ = reconsolidate
        self._retrieval_traces[trace_key] = trace_rows
        if view == "memory":
            # Keep the pre-view inspection key valid for existing consumers.
            legacy_trace_key = trace_key[:-1]
            self._retrieval_traces[legacy_trace_key] = trace_rows
        while len(self._retrieval_traces) > 32:
            del self._retrieval_traces[next(iter(self._retrieval_traces))]

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

        return (
            RetrievedChunks(final_chunks, diagnostics=diagnostics) if diagnostics else final_chunks
        )

    async def consolidate(self) -> ConsolidationResult:
        """
        Manually trigger a sleep consolidation cycle.

        Extracts permanent facts from the episodic buffer, stores them as
        semantic MemoryChunks, clears the buffer, and optionally generates
        procedural tool code.

        Returns:
            ConsolidationResult with facts extracted, chunks pruned, tools generated.
        """
        reconsolidation_results = await self.mutable_rag.process_labile_chunks()

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
        changed_ids = {chunk.id for chunk, changed in reconsolidation_results if changed}
        for chunk in new_chunks:
            if chunk.id in changed_ids or chunk.metadata.get("source") == "sleep_consolidation":
                self.refresh_graph_chunk(chunk)

        self._apply_governance()

        if self.config.debug:
            logger.debug(
                f"[MnemosEngine] Consolidated: {len(result.facts_extracted)} facts, "
                f"{result.chunks_pruned} episodes pruned, "
                f"{len(reconsolidation_results)} reconsolidation jobs processed"
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
        spreading_stats = self.spreading_activation.get_stats()
        graph_status = self._store.graph_derivation_status()
        if graph_status.get("managed_preferences", False):
            spreading_stats["graph_derivation_status"] = (
                "ready" if graph_status.get("ready") else "pending"
            )
            spreading_stats["pending_sources"] = graph_status.get("pending_count", 0)
        return {
            "engine": {
                "store_backend": store_stats.get("backend", "unknown"),
                "total_chunks": store_stats.get("total_chunks", 0),
            },
            "surprisal_gate": self.surprisal_gate.get_stats(),
            "mutable_rag": self.mutable_rag.get_stats(),
            "affective_router": self.affective_router.get_stats(),
            "sleep_daemon": self.sleep_daemon.get_stats(),
            "spreading_activation": spreading_stats,
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
    def judge(self) -> Judge:
        """The typed judgment backend, or an UnavailableJudge when none is configured."""
        return self._judge

    @property
    def embedder(self) -> EmbeddingProvider:
        """Direct access to the embedding provider."""
        return self._embedder
