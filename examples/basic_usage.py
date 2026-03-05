"""
examples/basic_usage.py — Getting started with Mnemos.

This example demonstrates the simplest way to use MnemosEngine:
- Zero external dependencies (MockLLM + SimpleEmbedding + InMemoryStore)
- Process a few interactions
- Retrieve relevant memories
- Inspect engine statistics

Run with: python examples/basic_usage.py
"""

import asyncio
import sys
import os

# Add parent directory to path for running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemos import (
    MnemosEngine,
    MnemosConfig,
    SurprisalConfig,
    Interaction,
)
from mnemos.utils import (
    MockLLMProvider,
    SimpleEmbeddingProvider,
    InMemoryStore,
)


async def main() -> None:
    print("=" * 60)
    print("Mnemos — Biomimetic Memory for LLMs")
    print("Basic Usage Example")
    print("=" * 60)

    # ─── Setup ────────────────────────────────────────────────────────────────
    # Use threshold=0.0 to store everything (for demo purposes).
    # In production, use the default 0.3 to filter mundane inputs.
    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=0.0, min_content_length=5),
        debug=True,
    )

    engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=128),
        store=InMemoryStore(),
    )

    print("\n📝 Step 1: Processing interactions through the memory pipeline...")
    print("-" * 60)

    # ─── Process interactions ─────────────────────────────────────────────────
    interactions = [
        ("user", "I'm a Python developer specializing in machine learning"),
        ("assistant", "Interesting! What frameworks do you use?"),
        ("user", "I mainly use PyTorch and Hugging Face Transformers"),
        ("user", "I deploy my models on AWS SageMaker"),
        ("user", "I'm currently building a RAG system for document Q&A"),
        ("user", "My team uses GitHub Actions for CI/CD"),
    ]

    stored_chunks = []
    for role, content in interactions:
        interaction = Interaction(role=role, content=content)
        result = await engine.process(interaction)
        status = "✅ STORED" if result.stored else "⏭  SKIPPED"
        print(f"{status} | salience={result.salience:.3f} | {content[:60]}")
        if result.stored:
            stored_chunks.append(result.chunk)

    print(f"\n→ {len(stored_chunks)} interactions stored in memory")

    # ─── Retrieve memories ────────────────────────────────────────────────────
    print("\n🔍 Step 2: Retrieving memories relevant to 'ML deployment'...")
    print("-" * 60)

    memories = await engine.retrieve("machine learning deployment infrastructure", top_k=3)

    if memories:
        for i, chunk in enumerate(memories, 1):
            state_info = ""
            if chunk.cognitive_state:
                s = chunk.cognitive_state
                state_info = f" | state: v={s.valence:.1f} a={s.arousal:.1f} c={s.complexity:.1f}"
            print(f"{i}. [salience={chunk.salience:.2f}{state_info}]")
            print(f"   {chunk.content}")
    else:
        print("No memories found.")

    # ─── Inspect stats ────────────────────────────────────────────────────────
    print("\n📊 Step 3: Engine statistics")
    print("-" * 60)
    stats = engine.get_stats()

    print(f"Store: {stats['store']['total_chunks']} chunks stored")
    print(
        f"SurprisalGate: processed={stats['surprisal_gate']['total_processed']}, "
        f"stored={stats['surprisal_gate']['total_stored']}"
    )
    print(f"AffectiveRouter: classified={stats['affective_router']['total_classified']}")
    print(f"SleepDaemon: episodic buffer={stats['sleep_daemon']['episodic_buffer_size']}")
    print(
        f"SpreadingActivation: {stats['spreading_activation']['total_nodes']} nodes, "
        f"{stats['spreading_activation']['total_edges']} edges"
    )

    print("\n✅ Basic usage example complete!")


if __name__ == "__main__":
    asyncio.run(main())
