# mypy: disable-error-code=untyped-decorator
"""
mnemos/mcp_server.py — Model Context Protocol (MCP) server for Mnemos.

Exposes the full biomimetic memory system as MCP tools that any
MCP-compatible agent (Claude Code, Cursor, Windsurf, etc.) can
discover and call natively.

MCP Tools provided:
  - mnemos_store:       Process and store a memory through the full pipeline
  - mnemos_retrieve:    Retrieve memories with affective + spreading activation
  - mnemos_consolidate: Trigger sleep consolidation (episodic → semantic)
  - mnemos_forget:      Delete a specific memory by ID
  - mnemos_stats:       Get system statistics across all modules
  - mnemos_health:      Profile readiness and dependency diagnostics
  - mnemos_inspect:     Inspect a specific memory chunk by ID
  - mnemos_list:        List all stored memories (with optional limit)

MCP Resources provided:
  - mnemos://stats         — Live system statistics
  - mnemos://architecture  — Runtime architecture and module overview

Usage:
  # stdio transport (for Claude Code, Cursor, etc.)
  python -m mnemos.mcp_server

  # Or via the CLI entry point
  mnemos-mcp

Configuration via environment variables:
  MNEMOS_CONFIG_PATH    — Canonical global config path (optional; host configs should prefer this)
  MNEMOS_LLM_PROVIDER   — "mock" (default), "ollama", "openai", "openclaw", or "openrouter"
  MNEMOS_LLM_MODEL      — Model name for LLM provider (default: "llama3")
  MNEMOS_EMBEDDING_PROVIDER — "simple" (default), "ollama", "openai", "openclaw", or "openrouter"
  MNEMOS_EMBEDDING_MODEL — Embedding model name (provider-specific default if unset)
  MNEMOS_EMBEDDING_DIM  — Embedding dimension for simple provider (default: 384)
  MNEMOS_OLLAMA_URL     — Ollama API base URL (default: "http://localhost:11434")
  MNEMOS_OPENAI_API_KEY — OpenAI API key (required if provider is "openai")
  MNEMOS_OPENAI_URL     — OpenAI-compatible base URL
  MNEMOS_OPENCLAW_API_KEY — OpenClaw API key (or fallback to MNEMOS_OPENAI_API_KEY)
  MNEMOS_OPENCLAW_URL   — OpenClaw API base URL (or fallback to MNEMOS_OPENAI_URL)
  MNEMOS_OPENROUTER_API_KEY — OpenRouter API key (required if provider is "openrouter")
  MNEMOS_OPENROUTER_URL — OpenRouter API base URL (default: "https://openrouter.ai/api/v1")
  MNEMOS_STORE_TYPE     — "memory" (default), "sqlite", or "qdrant"
  MNEMOS_SQLITE_PATH    — Path for SQLite store (default: "mnemos_memory.db")
  MNEMOS_QDRANT_URL     — Qdrant server URL (default: "http://localhost:6333")
  MNEMOS_QDRANT_API_KEY — Qdrant API key (optional)
  MNEMOS_QDRANT_PATH    — Local embedded Qdrant path (optional, overrides URL)
  MNEMOS_QDRANT_COLLECTION — Qdrant collection name (default: "mnemos_memory")
  MNEMOS_QDRANT_VECTOR_SIZE — Optional fixed vector size for pre-creating collection
  MNEMOS_STORAGE        — Alias for MNEMOS_STORE_TYPE
  MNEMOS_DB_PATH        — Alias for MNEMOS_SQLITE_PATH
  MNEMOS_SURPRISAL_THRESHOLD — Surprisal gate threshold (default: 0.3)
  MNEMOS_MEMORY_SAFETY_ENABLED — Enable shared memory write safety firewall (default: true)
  MNEMOS_MEMORY_SECRET_ACTION — Secret handling: allow|redact|block (default: block)
  MNEMOS_MEMORY_PII_ACTION — PII handling: allow|redact|block (default: redact)
  MNEMOS_MEMORY_CAPTURE_MODE — Ingestion mode: all|manual_only|hooks_only (default: all)
  MNEMOS_MEMORY_RETENTION_TTL_DAYS — Prune memories older than this many days (0=disabled)
  MNEMOS_MEMORY_MAX_CHUNKS_PER_SCOPE — Max chunks per (scope,scope_id) partition (0=disabled)
  MNEMOS_DEBUG          — "true" to enable debug logging
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from .config import MemoryGovernanceConfig, MemorySafetyConfig, MnemosConfig, SurprisalConfig
from .engine import MnemosEngine
from .health import run_health_checks
from .inspectability import build_chunk_inspection
from .observability import configure_logging, log_event
from .runtime import (
    build_embedder_from_env,
    build_llm_from_env,
    build_mnemos_config_from_env,
    build_store_from_env,
)
from .types import Interaction
from .utils.embeddings import EmbeddingProvider
from .utils.llm import MockLLMProvider, LLMProvider
from .utils.storage import MemoryStore

VALID_SCOPES = ("project", "workspace", "global")
MemoryAction = Literal["allow", "redact", "block"]
CaptureMode = Literal["all", "manual_only", "hooks_only"]


def _parse_allowed_scopes(raw: str) -> tuple[str, ...]:
    scopes = [scope.strip().lower() for scope in raw.split(",") if scope.strip()]
    if not scopes:
        return VALID_SCOPES
    invalid = [scope for scope in scopes if scope not in VALID_SCOPES]
    if invalid:
        raise ValueError(
            f"Invalid allowed scope(s): {', '.join(invalid)}. "
            f"Expected one or more of: {', '.join(VALID_SCOPES)}."
        )
    deduped: list[str] = []
    for scope in scopes:
        if scope not in deduped:
            deduped.append(scope)
    return tuple(deduped)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _memory_action_from_env(name: str, default: MemoryAction) -> MemoryAction:
    raw = (os.getenv(name, default) or default).strip().lower()
    if raw in {"allow", "redact", "block"}:
        return cast(MemoryAction, raw)
    return default


def _capture_mode_from_env(name: str, default: CaptureMode) -> CaptureMode:
    raw = (os.getenv(name, default) or default).strip().lower()
    if raw in {"all", "manual_only", "hooks_only"}:
        return cast(CaptureMode, raw)
    return default


# ---------------------------------------------------------------------------
# Engine lifecycle: initialize once, share across all tool calls
# ---------------------------------------------------------------------------


def _build_llm_provider() -> LLMProvider:
    """Build LLM provider from environment variables."""
    return build_llm_from_env()


def _build_store() -> MemoryStore:
    """Build storage backend from environment variables."""
    return build_store_from_env(default_store_type="memory")


def _build_embedder() -> EmbeddingProvider:
    """Build embedding provider from environment variables."""
    return build_embedder_from_env(default_provider="simple")


def _build_config() -> MnemosConfig:
    """Build configuration from environment variables."""
    return build_mnemos_config_from_env(default_store_type="memory")


@dataclass
class MnemosContext:
    """Holds the initialized MnemosEngine for the MCP server lifetime."""

    engine: MnemosEngine


# ---------------------------------------------------------------------------
# MCP Server definition using FastMCP
# ---------------------------------------------------------------------------


def create_mcp_server() -> Any:
    """Create and configure the MCP server with all mnemos tools and resources."""
    try:
        from mcp.server.fastmcp import Context, FastMCP
    except ImportError:
        raise ImportError(
            "MCP server requires the 'mcp' package. Install it with:\n"
            "  pip install 'mnemos-memory[mcp]'\n"
            "  # or\n"
            "  pip install mcp"
        )
    globals()["Context"] = Context

    @asynccontextmanager
    async def mnemos_lifespan(server: FastMCP) -> AsyncIterator[MnemosContext]:
        """Initialize the MnemosEngine on startup, clean up on shutdown."""
        _ = server
        engine = MnemosEngine(
            config=_build_config(),
            llm=_build_llm_provider(),
            embedder=_build_embedder(),
            store=_build_store(),
        )
        health = run_health_checks(default_store_type="memory")
        log_event(
            "mnemos.startup",
            transport="mcp_stdio",
            status=health["status"],
            profile=health["profile"],
            store_type=health["store_type"],
            llm_provider=health["llm_provider"],
            embedding_provider=health["embedding_provider"],
        )
        if health["status"] != "ready":
            log_event(
                "mnemos.degraded_mode",
                transport="mcp_stdio",
                status=health["status"],
                summary=health["summary"],
            )
        try:
            yield MnemosContext(engine=engine)
        finally:
            # Future: clean up connections (Neo4j, etc.)
            pass

    mcp = FastMCP(
        "Mnemos",
        dependencies=["mnemos", "numpy", "pydantic"],
        lifespan=mnemos_lifespan,
    )

    def _engine_from_ctx(
        ctx: Context[Any, Any, Any] | None,
        tool_name: str,
    ) -> MnemosEngine:
        if ctx is None:
            raise RuntimeError(f"MCP context was not injected for {tool_name}.")
        return cast(MnemosEngine, ctx.request_context.lifespan_context.engine)

    # -----------------------------------------------------------------------
    # TOOLS
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def mnemos_store(
        content: str,
        role: str = "user",
        metadata: str = "{}",
        scope: str = "project",
        scope_id: str = "default",
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        """Store a memory through the full biomimetic pipeline.

        The memory passes through:
        1. Surprisal Gate — only stores if content is surprising (high prediction error)
        2. Affective Router — tags with emotional/cognitive state
        3. Spreading Activation — connects to related memories in the graph

        Args:
            content: The text content to potentially memorize.
            role: Speaker role — "user", "assistant", or "system".
            metadata: JSON string of additional metadata to attach.

        Returns:
            JSON with storage decision, salience score, and reason.
        """
        engine = _engine_from_ctx(ctx, "mnemos_store")

        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            meta = {}

        interaction = Interaction(role=role, content=content, metadata=meta)
        result = await engine.process(
            interaction,
            scope=scope,
            scope_id=(scope_id or None),
        )

        return json.dumps(
            {
                "stored": result.stored,
                "salience": round(result.salience, 4),
                "reason": result.reason,
                "chunk_id": result.chunk.id if result.chunk else None,
                "content": result.chunk.content if result.chunk else None,
                "scope": (result.chunk.metadata.get("scope") if result.chunk else None),
                "scope_id": (result.chunk.metadata.get("scope_id") if result.chunk else None),
            },
            indent=2,
        )

    @mcp.tool()
    async def mnemos_retrieve(
        query: str,
        top_k: int = 5,
        reconsolidate: bool = True,
        current_scope: str = "project",
        scope_id: str = "default",
        allowed_scopes: str = "project,workspace,global",
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        """Retrieve memories using the full contextual retrieval pipeline.

        Pipeline:
        1. Classifies current emotional/cognitive state from query
        2. Spreading Activation — finds associatively connected context
        3. Affective Router — re-ranks by emotional state match
        4. Mutable RAG — flags retrieved memories for reconsolidation

        Args:
            query: What to search for in memory.
            top_k: Maximum number of memories to return.
            reconsolidate: Whether to update stale memories after retrieval.

        Returns:
            JSON array of matching memories with content, salience, and metadata.
        """
        engine = _engine_from_ctx(ctx, "mnemos_retrieve")
        parsed_allowed_scopes = _parse_allowed_scopes(allowed_scopes)
        chunks = await engine.retrieve(
            query,
            top_k=top_k,
            reconsolidate=reconsolidate,
            current_scope=current_scope,
            scope_id=(scope_id or None),
            allowed_scopes=parsed_allowed_scopes,
        )

        results = []
        for chunk in chunks:
            results.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "salience": round(chunk.salience, 4),
                    "version": chunk.version,
                    "access_count": chunk.access_count,
                    "cognitive_state": (
                        {
                            "valence": round(chunk.cognitive_state.valence, 3),
                            "arousal": round(chunk.cognitive_state.arousal, 3),
                            "complexity": round(chunk.cognitive_state.complexity, 3),
                        }
                        if chunk.cognitive_state
                        else None
                    ),
                    "created_at": chunk.created_at.isoformat(),
                    "updated_at": chunk.updated_at.isoformat(),
                    "scope": chunk.metadata.get("scope", "global"),
                    "scope_id": chunk.metadata.get("scope_id"),
                }
            )

        return json.dumps(results, indent=2)

    @mcp.tool()
    async def mnemos_consolidate(ctx: Context[Any, Any, Any] | None = None) -> str:
        """Trigger sleep consolidation — compress episodic memories into semantic facts.

        Mimics the hippocampal-neocortical transfer during sleep:
        - Replays recent episodic interactions
        - Extracts permanent facts and user preferences
        - Stores distilled knowledge as long-term semantic memory
        - Prunes raw episode buffer to save capacity

        Returns:
            JSON with facts extracted, episodes pruned, and any tools generated.
        """
        engine = _engine_from_ctx(ctx, "mnemos_consolidate")
        result = await engine.consolidate()

        return json.dumps(
            {
                "facts_extracted": result.facts_extracted,
                "chunks_pruned": result.chunks_pruned,
                "tools_generated": result.tools_generated,
                "duration_seconds": round(result.duration_seconds, 3),
            },
            indent=2,
        )

    @mcp.tool()
    async def mnemos_forget(
        chunk_id: str,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        """Delete a specific memory by its ID.

        Permanent removal — the memory cannot be recovered.

        Args:
            chunk_id: The UUID of the memory to delete.

        Returns:
            JSON confirming deletion or reporting the memory was not found.
        """
        engine = _engine_from_ctx(ctx, "mnemos_forget")
        deleted = engine.store.delete(chunk_id)

        # Also remove from spreading activation graph if present
        node = engine.spreading_activation.get_node(chunk_id)
        if node:
            engine.spreading_activation.remove_node(chunk_id)

        return json.dumps(
            {
                "deleted": deleted,
                "chunk_id": chunk_id,
            },
            indent=2,
        )

    @mcp.tool()
    async def mnemos_stats(ctx: Context[Any, Any, Any] | None = None) -> str:
        """Get system-wide statistics from all memory modules.

        Returns stats for: engine, surprisal gate, mutable RAG,
        affective router, sleep daemon, spreading activation, and store.
        """
        engine = _engine_from_ctx(ctx, "mnemos_stats")
        stats = engine.get_stats()

        # Make sure everything is JSON-serializable
        def _clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(item) for item in obj]
            if isinstance(obj, float):
                return round(obj, 4)
            return obj

        return json.dumps(_clean(stats), indent=2)

    @mcp.tool()
    async def mnemos_health(ctx: Context[Any, Any, Any] | None = None) -> str:
        """Run profile readiness diagnostics and dependency checks.

        Returns:
            JSON health report including status, checks, and recommendations.
        """
        engine = _engine_from_ctx(ctx, "mnemos_health")
        report = run_health_checks(default_store_type="memory")
        store_stats = engine.store.get_stats()
        report["runtime"] = {
            "store_backend": store_stats.get("backend", "unknown"),
            "total_chunks": store_stats.get("total_chunks", 0),
        }
        return json.dumps(report, indent=2)

    @mcp.tool()
    async def mnemos_inspect(
        chunk_id: str,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        """Inspect a specific memory chunk by its ID.

        Returns the full memory record including content, metadata,
        salience, cognitive state, version history, and access count.

        Args:
            chunk_id: The UUID of the memory to inspect.
        """
        engine = _engine_from_ctx(ctx, "mnemos_inspect")

        payload = build_chunk_inspection(engine, chunk_id)
        if payload is not None:
            return json.dumps(payload, indent=2)

        return json.dumps({"error": f"Memory chunk {chunk_id!r} not found."})

    @mcp.tool()
    async def mnemos_list(
        limit: int = 20,
        sort_by: str = "created_at",
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        """List all stored memories.

        Args:
            limit: Maximum number of memories to return (default 20).
            sort_by: Sort field — "created_at", "salience", or "access_count".

        Returns:
            JSON array of memory summaries.
        """
        engine = _engine_from_ctx(ctx, "mnemos_list")
        all_chunks = engine.store.get_all()

        # Sort
        if sort_by == "salience":
            all_chunks.sort(key=lambda c: c.salience, reverse=True)
        elif sort_by == "access_count":
            all_chunks.sort(key=lambda c: c.access_count, reverse=True)
        else:
            all_chunks.sort(key=lambda c: c.created_at, reverse=True)

        # Limit
        chunks = all_chunks[:limit]

        results = []
        for chunk in chunks:
            results.append(
                {
                    "id": chunk.id,
                    "content": chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""),
                    "salience": round(chunk.salience, 4),
                    "version": chunk.version,
                    "access_count": chunk.access_count,
                    "created_at": chunk.created_at.isoformat(),
                }
            )

        return json.dumps(
            {
                "total": len(all_chunks),
                "showing": len(chunks),
                "memories": results,
            },
            indent=2,
        )

    # -----------------------------------------------------------------------
    # RESOURCES
    # -----------------------------------------------------------------------

    @mcp.resource("mnemos://stats")
    async def resource_stats() -> str:
        """Live system statistics across all memory modules."""
        # Note: resources don't get Context in the same way —
        # this is a static resource that returns basic info
        return json.dumps(
            {
                "name": "Mnemos",
                "version": "0.1.0",
                "description": "Biomimetic memory architecture for LLMs",
                "modules": [
                    "SurprisalGate (Predictive Coding)",
                    "MutableRAG (Memory Reconsolidation)",
                    "AffectiveRouter (Amygdala Filter)",
                    "SleepDaemon (Hippocampal Consolidation)",
                    "SpreadingActivation (Graph RAG)",
                ],
            },
            indent=2,
        )

    @mcp.resource("mnemos://architecture")
    async def resource_architecture() -> str:
        """Description of the biomimetic memory architecture and its modules."""
        return json.dumps(
            {
                "pipeline": {
                    "encode": [
                        "Input → SurprisalGate (prediction error filter)",
                        "→ AffectiveRouter (emotional state tagging)",
                        "→ SpreadingActivation (graph integration)",
                        "→ SleepDaemon episodic buffer",
                    ],
                    "retrieve": [
                        "Query → SpreadingActivation (associative context)",
                        "→ AffectiveRouter (state-dependent re-ranking)",
                        "→ MutableRAG (reconsolidation of stale facts)",
                        "→ Ranked results",
                    ],
                    "consolidate": [
                        "Idle → SleepDaemon (episodic → semantic transfer)",
                        "→ Fact extraction + episode pruning",
                        "→ Optional proceduralization (tool generation)",
                    ],
                },
                "neuroscience_inspiration": {
                    "surprisal_gate": "Predictive coding / active inference — only encode prediction errors",
                    "mutable_rag": "Memory reconsolidation — recalled memories enter labile state, get rewritten",
                    "affective_router": "Amygdala-mediated state-dependent memory — emotional context shapes retrieval",
                    "sleep_daemon": "Hippocampal-neocortical transfer during sleep — episodic → semantic compression",
                    "spreading_activation": "Collins & Loftus (1975) — activation energy propagates through semantic graph",
                },
            },
            indent=2,
        )

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server via stdio transport (for Claude Code, Cursor, etc.)."""
    configure_logging()
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
