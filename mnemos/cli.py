"""
mnemos/cli.py — Command-line interface for mnemos memory operations.

Provides shell-friendly commands that work with Claude Code hooks,
cron jobs, and other automation. Unlike the MCP server (stdio transport),
these commands run as one-shot processes and exit.

Usage:
    mnemos-cli store "I prefer dark mode"
    mnemos-cli retrieve "user preferences" --top-k 5
    mnemos-cli consolidate
    mnemos-cli stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .config import MnemosConfig, SurprisalConfig
from .engine import MnemosEngine
from .runtime import build_embedder_from_env, build_store_from_env
from .types import Interaction
from .utils.llm import MockLLMProvider, LLMProvider


def _build_engine() -> MnemosEngine:
    """Build engine from environment, defaulting to SQLite for persistence."""
    import os

    provider_name = os.getenv("MNEMOS_LLM_PROVIDER", "mock").lower()
    llm: LLMProvider

    if provider_name == "ollama":
        from .utils.llm import OllamaProvider

        llm = OllamaProvider(
            base_url=os.getenv("MNEMOS_OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("MNEMOS_LLM_MODEL", "llama3"),
        )
    elif provider_name == "openai":
        from .utils.llm import OpenAIProvider

        api_key = os.getenv("MNEMOS_OPENAI_API_KEY", "")
        if not api_key:
            print("Error: MNEMOS_OPENAI_API_KEY required for openai provider", file=sys.stderr)
            sys.exit(1)
        llm = OpenAIProvider(
            api_key=api_key,
            base_url=os.getenv("MNEMOS_OPENAI_URL", "https://api.openai.com/v1"),
            model=os.getenv("MNEMOS_LLM_MODEL", "gpt-4o-mini"),
        )
    else:
        llm = MockLLMProvider()

    threshold = float(os.getenv("MNEMOS_SURPRISAL_THRESHOLD", "0.3"))

    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=threshold),
        debug=os.getenv("MNEMOS_DEBUG", "").lower() in ("true", "1", "yes"),
    )

    return MnemosEngine(
        config=config,
        llm=llm,
        embedder=build_embedder_from_env(default_provider="simple"),
        store=build_store_from_env(default_store_type="sqlite"),
    )


async def _cmd_store(args: argparse.Namespace) -> None:
    engine = _build_engine()
    interaction = Interaction(role=args.role, content=args.content)
    result = await engine.process(interaction)
    output = {
        "stored": result.stored,
        "salience": round(result.salience, 4),
        "reason": result.reason,
        "chunk_id": result.chunk.id if result.chunk else None,
    }
    print(json.dumps(output, indent=2))


async def _cmd_retrieve(args: argparse.Namespace) -> None:
    engine = _build_engine()
    chunks = await engine.retrieve(args.query, top_k=args.top_k)
    results = []
    for chunk in chunks:
        results.append(
            {
                "id": chunk.id,
                "content": chunk.content,
                "salience": round(chunk.salience, 4),
                "version": chunk.version,
            }
        )
    print(json.dumps(results, indent=2))


async def _cmd_consolidate(args: argparse.Namespace) -> None:
    engine = _build_engine()
    result = await engine.consolidate()
    output = {
        "facts_extracted": result.facts_extracted,
        "chunks_pruned": result.chunks_pruned,
        "duration_seconds": round(result.duration_seconds, 3),
    }
    print(json.dumps(output, indent=2))


async def _cmd_stats(args: argparse.Namespace) -> None:
    engine = _build_engine()
    stats = engine.get_stats()
    print(json.dumps(stats, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mnemos-cli",
        description="Mnemos memory system CLI — shell-friendly commands for hooks and automation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # store
    sp_store = subparsers.add_parser("store", help="Store a memory through the full pipeline")
    sp_store.add_argument("content", help="Text content to memorize")
    sp_store.add_argument("--role", default="user", help="Speaker role (default: user)")

    # retrieve
    sp_retrieve = subparsers.add_parser("retrieve", help="Retrieve memories by query")
    sp_retrieve.add_argument("query", help="Search query")
    sp_retrieve.add_argument("--top-k", type=int, default=5, help="Max results (default: 5)")

    # consolidate
    subparsers.add_parser("consolidate", help="Trigger sleep consolidation")

    # stats
    subparsers.add_parser("stats", help="Show system statistics")

    args = parser.parse_args()

    if args.command == "store":
        asyncio.run(_cmd_store(args))
    elif args.command == "retrieve":
        asyncio.run(_cmd_retrieve(args))
    elif args.command == "consolidate":
        asyncio.run(_cmd_consolidate(args))
    elif args.command == "stats":
        asyncio.run(_cmd_stats(args))


if __name__ == "__main__":
    main()
