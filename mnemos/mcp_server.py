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
  - mnemos_inspect:     Inspect a specific memory chunk by ID
  - mnemos_list:        List all stored memories (with optional limit)

MCP Resources provided:
  - mnemos://stats         — Live system statistics
  - mnemos://memories      — All stored memory contents

Usage:
  # stdio transport (for Claude Code, Cursor, etc.)
  python -m mnemos.mcp_server

  # Or via the CLI entry point
  mnemos-mcp

Configuration via environment variables:
  MNEMOS_LLM_PROVIDER   — "mock" (default), "ollama", or "openai"
  MNEMOS_LLM_MODEL      — Model name for LLM provider (default: "llama3")
  MNEMOS_OLLAMA_URL     — Ollama API base URL (default: "http://localhost:11434")
  MNEMOS_OPENAI_API_KEY — OpenAI API key (required if provider is "openai")
  MNEMOS_OPENAI_URL     — OpenAI-compatible base URL
  MNEMOS_STORE_TYPE     — "memory" (default) or "sqlite"
  MNEMOS_SQLITE_PATH    — Path for SQLite store (default: "mnemos_memory.db")
  MNEMOS_SURPRISAL_THRESHOLD — Surprisal gate threshold (default: 0.3)
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
from typing import Any

from .config import MnemosConfig, SurprisalConfig
from .engine import MnemosEngine
from .types import Interaction
from .utils.embeddings import SimpleEmbeddingProvider
from .utils.llm import MockLLMProvider, LLMProvider
from .utils.storage import InMemoryStore, MemoryStore

# ---------------------------------------------------------------------------
# Engine lifecycle: initialize once, share across all tool calls
# ---------------------------------------------------------------------------


def _build_llm_provider() -> LLMProvider:
    """Build LLM provider from environment variables."""
    provider = os.getenv("MNEMOS_LLM_PROVIDER", "mock").lower()

    if provider == "mock":
        return MockLLMProvider()

    elif provider == "ollama":
        from .utils.llm import OllamaProvider

        return OllamaProvider(
            base_url=os.getenv("MNEMOS_OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("MNEMOS_LLM_MODEL", "llama3"),
        )

    elif provider == "openai":
        from .utils.llm import OpenAIProvider

        api_key = os.getenv("MNEMOS_OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("MNEMOS_OPENAI_API_KEY must be set when using openai provider")
        return OpenAIProvider(
            api_key=api_key,
            base_url=os.getenv("MNEMOS_OPENAI_URL", "https://api.openai.com/v1"),
            model=os.getenv("MNEMOS_LLM_MODEL", "gpt-4o-mini"),
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'mock', 'ollama', or 'openai'.")


def _build_store() -> MemoryStore:
    """Build storage backend from environment variables."""
    store_type = os.getenv("MNEMOS_STORE_TYPE", "memory").lower()

    if store_type == "memory":
        return InMemoryStore()

    elif store_type == "sqlite":
        from .utils.storage import SQLiteStore

        path = os.getenv("MNEMOS_SQLITE_PATH", "mnemos_memory.db")
        return SQLiteStore(db_path=path)

    else:
        raise ValueError(f"Unknown store type: {store_type!r}. Use 'memory' or 'sqlite'.")


def _build_config() -> MnemosConfig:
    """Build configuration from environment variables."""
    threshold = float(os.getenv("MNEMOS_SURPRISAL_THRESHOLD", "0.3"))
    debug = os.getenv("MNEMOS_DEBUG", "").lower() in ("true", "1", "yes")
    return MnemosConfig(
        surprisal=SurprisalConfig(threshold=threshold),
        debug=debug,
    )


@dataclass
class MnemosContext:
    """Holds the initialized MnemosEngine for the MCP server lifetime."""

    engine: MnemosEngine


# ---------------------------------------------------------------------------
# MCP Server definition using FastMCP
# ---------------------------------------------------------------------------


def create_mcp_server():
    """Create and configure the MCP server with all mnemos tools and resources."""
    try:
        from mcp.server.fastmcp import FastMCP, Context
    except ImportError:
        raise ImportError(
            "MCP server requires the 'mcp' package. Install it with:\n"
            "  pip install 'mnemos[mcp]'\n"
            "  # or\n"
            "  pip install mcp"
        )

    @asynccontextmanager
    async def mnemos_lifespan(server: FastMCP) -> AsyncIterator[MnemosContext]:
        """Initialize the MnemosEngine on startup, clean up on shutdown."""
        engine = MnemosEngine(
            config=_build_config(),
            llm=_build_llm_provider(),
            embedder=SimpleEmbeddingProvider(dim=int(os.getenv("MNEMOS_EMBEDDING_DIM", "384"))),
            store=_build_store(),
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

    # -----------------------------------------------------------------------
    # TOOLS
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def mnemos_store(
        content: str,
        role: str = "user",
        metadata: str = "{}",
        ctx: Context = None,
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
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine

        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            meta = {}

        interaction = Interaction(role=role, content=content, metadata=meta)
        result = await engine.process(interaction)

        return json.dumps(
            {
                "stored": result.stored,
                "salience": round(result.salience, 4),
                "reason": result.reason,
                "chunk_id": result.chunk.id if result.chunk else None,
                "content": result.chunk.content if result.chunk else None,
            },
            indent=2,
        )

    @mcp.tool()
    async def mnemos_retrieve(
        query: str,
        top_k: int = 5,
        reconsolidate: bool = True,
        ctx: Context = None,
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
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine
        chunks = await engine.retrieve(query, top_k=top_k, reconsolidate=reconsolidate)

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
                }
            )

        return json.dumps(results, indent=2)

    @mcp.tool()
    async def mnemos_consolidate(ctx: Context = None) -> str:
        """Trigger sleep consolidation — compress episodic memories into semantic facts.

        Mimics the hippocampal-neocortical transfer during sleep:
        - Replays recent episodic interactions
        - Extracts permanent facts and user preferences
        - Stores distilled knowledge as long-term semantic memory
        - Prunes raw episode buffer to save capacity

        Returns:
            JSON with facts extracted, episodes pruned, and any tools generated.
        """
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine
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
    async def mnemos_forget(chunk_id: str, ctx: Context = None) -> str:
        """Delete a specific memory by its ID.

        Permanent removal — the memory cannot be recovered.

        Args:
            chunk_id: The UUID of the memory to delete.

        Returns:
            JSON confirming deletion or reporting the memory was not found.
        """
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine
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
    async def mnemos_stats(ctx: Context = None) -> str:
        """Get system-wide statistics from all memory modules.

        Returns stats for: engine, surprisal gate, mutable RAG,
        affective router, sleep daemon, spreading activation, and store.
        """
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine
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
    async def mnemos_inspect(chunk_id: str, ctx: Context = None) -> str:
        """Inspect a specific memory chunk by its ID.

        Returns the full memory record including content, metadata,
        salience, cognitive state, version history, and access count.

        Args:
            chunk_id: The UUID of the memory to inspect.
        """
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine

        chunk = engine.store.get(chunk_id)
        if chunk:
            return json.dumps(
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
                    "metadata": chunk.metadata,
                    "created_at": chunk.created_at.isoformat(),
                    "updated_at": chunk.updated_at.isoformat(),
                },
                indent=2,
            )

        return json.dumps({"error": f"Memory chunk {chunk_id!r} not found."})

    @mcp.tool()
    async def mnemos_list(
        limit: int = 20,
        sort_by: str = "created_at",
        ctx: Context = None,
    ) -> str:
        """List all stored memories.

        Args:
            limit: Maximum number of memories to return (default 20).
            sort_by: Sort field — "created_at", "salience", or "access_count".

        Returns:
            JSON array of memory summaries.
        """
        engine: MnemosEngine = ctx.request_context.lifespan_context.engine
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


def main():
    """Run the MCP server via stdio transport (for Claude Code, Cursor, etc.)."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
