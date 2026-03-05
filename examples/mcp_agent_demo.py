"""
examples/mcp_agent_demo.py — Demonstrates how an agent would use Mnemos via MCP.

This shows the programmatic equivalent of what happens when an agent
like Claude Code calls Mnemos MCP tools. Useful for understanding the
flow without needing a full MCP client setup.
"""

import asyncio
import json

from mnemos import MnemosEngine, MnemosConfig
from mnemos.config import SurprisalConfig, SleepConfig
from mnemos.types import Interaction


async def main():
    # ------------------------------------------------------------------
    # Setup: configure mnemos (same as MCP server does internally)
    # ------------------------------------------------------------------
    engine = MnemosEngine(
        config=MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.2),  # More sensitive
            sleep=SleepConfig(
                consolidation_interval_seconds=0,  # Allow immediate consolidation
                min_episodes_before_consolidation=3,  # Low threshold for demo
            ),
            debug=True,
        )
    )

    print("=" * 60)
    print("MNEMOS MCP AGENT DEMO")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Agent stores memories (mnemos_store equivalent)
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Storing memories through the pipeline ---\n")

    interactions = [
        ("user", "I primarily use Python and TypeScript for my projects."),
        ("user", "My production server runs on AWS us-east-1."),
        ("user", "Hi, how are you doing today?"),  # Mundane — should be filtered
        ("user", "CRITICAL: The database migration failed and data is corrupted!"),
        ("user", "I prefer dark mode and Neovim as my editor."),
        ("user", "I'm thinking about switching from React to Svelte."),
        ("assistant", "I'll help you plan the React to Svelte migration."),
        ("user", "Actually, I've decided to migrate to Rust instead of Svelte."),
    ]

    for role, content in interactions:
        result = await engine.process(Interaction(role=role, content=content))
        status = "STORED" if result.stored else "FILTERED"
        print(f"  [{status}] (salience={result.salience:.3f}) {content[:60]}...")

    # ------------------------------------------------------------------
    # Phase 2: Agent retrieves memories (mnemos_retrieve equivalent)
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Retrieving memories ---\n")

    queries = [
        "What programming languages does the user work with?",
        "What's the current server situation?",
        "What migration is the user planning?",
    ]

    for query in queries:
        chunks = await engine.retrieve(query, top_k=3)
        print(f"  Query: {query}")
        for i, chunk in enumerate(chunks, 1):
            state = chunk.cognitive_state
            state_str = (
                f"v={state.valence:.1f} a={state.arousal:.1f} c={state.complexity:.1f}"
                if state
                else "untagged"
            )
            print(f"    {i}. [{state_str}] {chunk.content[:70]}...")
        print()

    # ------------------------------------------------------------------
    # Phase 3: Sleep consolidation (mnemos_consolidate equivalent)
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Sleep consolidation ---\n")

    result = await engine.consolidate()
    print(f"  Facts extracted: {len(result.facts_extracted)}")
    for fact in result.facts_extracted:
        print(f"    • {fact}")
    print(f"  Episodes pruned: {result.chunks_pruned}")
    print(f"  Duration: {result.duration_seconds:.3f}s")

    # ------------------------------------------------------------------
    # Phase 4: System stats (mnemos_stats equivalent)
    # ------------------------------------------------------------------
    print("\n--- Phase 4: System statistics ---\n")
    stats = engine.get_stats()
    print(json.dumps(stats["engine"], indent=2))
    print(f"  Spreading activation nodes: {stats['spreading_activation'].get('total_nodes', 0)}")
    print(f"  Surprisal gate total processed: {stats['surprisal_gate'].get('total_processed', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
