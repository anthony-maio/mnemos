"""
examples/full_pipeline.py — All five Mnemos modules in action.

This example demonstrates each of the five biomimetic memory modules
explicitly, showing their individual behavior and how they compose together:

  Module 1: SurprisalGate   — Only encode surprising inputs
  Module 2: MutableRAG      — Update stale facts on retrieval
  Module 3: AffectiveRouter — Route by emotional state
  Module 4: SleepDaemon     — Consolidate episodic → semantic memory
  Module 5: SpreadingActivation — Associative graph retrieval

Run with: python examples/full_pipeline.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemos import (
    MnemosEngine,
    MnemosConfig,
    SurprisalConfig,
    MutableRAGConfig,
    AffectiveConfig,
    SleepConfig,
    SpreadingConfig,
    Interaction,
    CognitiveState,
    MemoryChunk,
)
from mnemos.modules.surprisal import SurprisalGate
from mnemos.modules.mutable_rag import MutableRAG
from mnemos.modules.affective import AffectiveRouter
from mnemos.modules.sleep import SleepDaemon
from mnemos.modules.spreading import SpreadingActivation
from mnemos.utils import (
    MockLLMProvider,
    SimpleEmbeddingProvider,
    InMemoryStore,
)


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ─── Module 1: SurprisalGate Demo ─────────────────────────────────────────────


async def demo_surprisal_gate() -> None:
    separator("Module 1: SurprisalGate (Predictive Coding)")
    print("""
Neuroscience: The brain predicts its environment and only encodes memories
when there is a prediction error (surprise). Mundane, expected inputs
are discarded at the gate — saving memory for what actually matters.
""")

    embedder = SimpleEmbeddingProvider(dim=128)
    store = InMemoryStore()

    # Low threshold: store most things (for demo visibility)
    config = SurprisalConfig(threshold=0.0, min_content_length=5)
    gate = SurprisalGate(
        llm=MockLLMProvider(),
        embedder=embedder,
        store=store,
        config=config,
    )

    # Seed history with some context
    gate.add_to_history(Interaction(role="user", content="Tell me about Python"))
    gate.add_to_history(Interaction(role="assistant", content="Python is a versatile language"))

    test_inputs = [
        "What's the weather like?",  # Expected small talk
        "CRITICAL: Our production DB is DOWN!",  # High-surprise crisis
        "Tell me more about Python.",  # Low surprise (expected follow-up)
        "I've decided to quit and start a bakery",  # Highly surprising topic shift
    ]

    print("Processing inputs through surprisal gate:")
    for content in test_inputs:
        result = await gate.process(Interaction(role="user", content=content))
        icon = "✅" if result.stored else "⏭ "
        print(f"  {icon} [{result.salience:.3f}] {content[:55]}")
        print(f"      → {result.reason}")

    stats = gate.get_stats()
    print(
        f"\nStats: {stats['total_stored']}/{stats['total_processed']} stored "
        f"(threshold={stats['threshold']})"
    )


# ─── Module 2: MutableRAG Demo ────────────────────────────────────────────────


async def demo_mutable_rag() -> None:
    separator("Module 2: MutableRAG (Memory Reconsolidation)")
    print("""
Neuroscience: Every time a human recalls a memory, it enters a 'labile'
(unstable) state. New information can be integrated before the memory
re-stabilizes. This is Memory Reconsolidation.

In AI: When a retrieved fact is outdated, we overwrite it in-place
rather than accumulating contradictory facts — the database heals itself.
""")

    embedder = SimpleEmbeddingProvider(dim=128)
    store = InMemoryStore()

    # Initial fact stored in the database
    old_fact = MemoryChunk(
        content="User uses React for frontend development",
        embedding=embedder.embed("User uses React for frontend development"),
        salience=0.8,
        metadata={"source": "initial_encoding"},
    )
    store.store(old_fact)
    print(f"Initial memory: '{old_fact.content}'")
    print(f"Version: {old_fact.version}, ID: {old_fact.id[:8]}...")

    # LLM that recognizes the user has switched to Vue
    llm_with_update = MockLLMProvider(
        responses={
            "has this stored memory": "CHANGED: User migrated from React to Vue.js in mid-2025"
        }
    )
    rag = MutableRAG(llm=llm_with_update, embedder=embedder, store=store)

    # Simulate retrieval (flags as labile)
    query_embedding = embedder.embed("frontend framework")
    retrieved = rag.retrieve(
        query_embedding, top_k=1, current_context="User mentioned Vue.js migration"
    )
    print(f"\nRetrieved: '{retrieved[0].content}'")
    print(f"Labile queue size: {rag.get_labile_count()}")

    # Reconsolidate
    print("\nRunning reconsolidation...")
    results = await rag.process_labile_chunks()
    updated_chunk, changed = results[0]

    if changed:
        print(f"✅ Memory UPDATED!")
        print(f"   Old: 'User uses React for frontend development'")
        print(f"   New: '{updated_chunk.content}'")
        print(f"   Version: {old_fact.version} → {updated_chunk.version}")
    else:
        print(f"→ Memory unchanged: '{updated_chunk.content}'")

    # Verify the store has the updated version
    stored = store.get(old_fact.id)
    if stored:
        print(f"\nVerify store: '{stored.content}' (v{stored.version})")


# ─── Module 3: AffectiveRouter Demo ──────────────────────────────────────────


async def demo_affective_router() -> None:
    separator("Module 3: AffectiveRouter (Amygdala-Inspired Routing)")
    print("""
Neuroscience: The amygdala tags memories with emotional context.
State-dependent memory: memories encoded during stress are preferentially
retrieved when you're stressed again. Emotional context is a retrieval key.

Scoring formula: final_score = (similarity × 0.7) + (state_match × 0.3)
""")

    embedder = SimpleEmbeddingProvider(dim=128)
    store = InMemoryStore()
    config = AffectiveConfig(weight_similarity=0.7, weight_state=0.3)
    router = AffectiveRouter(llm=MockLLMProvider(), embedder=embedder, config=config)

    # Create memories with different emotional contexts
    crisis_memory = MemoryChunk(
        content="Fixed production outage by rolling back deployment v2.3.1",
        embedding=embedder.embed("Fixed production outage by rolling back deployment v2.3.1"),
        cognitive_state=CognitiveState(valence=-0.6, arousal=0.9, complexity=0.7),
        salience=0.95,
    )
    calm_memory = MemoryChunk(
        content="Scheduled maintenance: upgrade server RAM on Saturday",
        embedding=embedder.embed("Scheduled maintenance: upgrade server RAM on Saturday"),
        cognitive_state=CognitiveState(valence=0.3, arousal=0.1, complexity=0.3),
        salience=0.5,
    )
    store.store(crisis_memory)
    store.store(calm_memory)

    query = "server issue needs immediate attention"
    query_embedding = embedder.embed(query)

    # Score both chunks under two different emotional states
    print("Scoring chunks under different user states:")
    print(f"Query: '{query}'\n")

    # Panicking user state (high arousal, negative valence)
    panic_state = CognitiveState(valence=-0.7, arousal=0.9, complexity=0.8)
    score_crisis_panic = router.score_chunk(crisis_memory, query_embedding, panic_state)
    score_calm_panic = router.score_chunk(calm_memory, query_embedding, panic_state)

    # Calm user state
    calm_state = CognitiveState(valence=0.5, arousal=0.1, complexity=0.2)
    score_crisis_calm = router.score_chunk(crisis_memory, query_embedding, calm_state)
    score_calm_calm = router.score_chunk(calm_memory, query_embedding, calm_state)

    print("User state: PANICKING (arousal=0.9, valence=-0.7)")
    print(
        f"  Crisis memory score: {score_crisis_panic:.4f} ← {'WINNER ✅' if score_crisis_panic > score_calm_panic else ''}"
    )
    print(f"  Calm memory score:   {score_calm_panic:.4f}")

    print("\nUser state: CALM (arousal=0.1, valence=0.5)")
    print(f"  Crisis memory score: {score_crisis_calm:.4f}")
    print(
        f"  Calm memory score:   {score_calm_calm:.4f} ← {'WINNER ✅' if score_calm_calm > score_crisis_calm else ''}"
    )

    # Classify emotional state
    state = await router.classify_state(
        Interaction(role="user", content="URGENT: everything is broken!")
    )
    print(f"\nClassified state for 'URGENT: everything is broken!':")
    print(
        f"  valence={state.valence:.2f}, arousal={state.arousal:.2f}, complexity={state.complexity:.2f}"
    )


# ─── Module 4: SleepDaemon Demo ──────────────────────────────────────────────


async def demo_sleep_daemon() -> None:
    separator("Module 4: SleepDaemon (Hippocampal-Neocortical Consolidation)")
    print("""
Neuroscience: During sleep, the hippocampus replays the day's episodic
memories. The neocortex extracts generalizable semantic knowledge.
Raw episodes are pruned. Repeated behaviors may crystallize into
procedural skills (hippocampus → basal ganglia transfer).

In AI: Raw conversation logs → distilled facts in long-term store.
Episodic buffer is cleared after consolidation (active forgetting).
""")

    embedder = SimpleEmbeddingProvider(dim=128)
    store = InMemoryStore()
    config = SleepConfig(
        min_episodes_before_consolidation=3,
        consolidation_interval_seconds=0,
    )
    daemon = SleepDaemon(store=store, config=config)

    # Simulate a session's raw episodic memories
    raw_episodes = [
        ("user", "Hi, I'm working on a Python ML project"),
        ("assistant", "What kind of ML are you doing?"),
        ("user", "Natural language processing, specifically text classification"),
        ("user", "I use BERT via Hugging Face for the encoder"),
        ("user", "The dataset is around 50k labeled samples"),
        ("user", "We deploy the model to AWS Lambda for inference"),
        ("assistant", "How's the latency?"),
        ("user", "About 200ms per request, acceptable for our use case"),
    ]

    print("Adding raw episodes to hippocampal buffer...")
    for role, content in raw_episodes:
        daemon.add_episode(Interaction(role=role, content=content))
    print(f"Buffer size: {daemon.get_episode_count()} episodes")
    print(f"Should consolidate: {daemon.should_consolidate()}")

    print("\n💤 Running sleep consolidation...")
    result = await daemon.consolidate(MockLLMProvider(), embedder)

    print(f"\nConsolidation result:")
    print(f"  Episodes pruned: {result.chunks_pruned}")
    print(f"  Facts extracted: {len(result.facts_extracted)}")
    print(f"  Duration: {result.duration_seconds:.3f}s")
    print(f"\nExtracted facts (now in long-term store):")
    for i, fact in enumerate(result.facts_extracted, 1):
        print(f"  {i}. {fact}")

    print(f"\nBuffer size after consolidation: {daemon.get_episode_count()} (cleared)")
    print(f"Long-term store size: {len(store.get_all())} semantic chunks")


# ─── Module 5: SpreadingActivation Demo ──────────────────────────────────────


async def demo_spreading_activation() -> None:
    separator("Module 5: SpreadingActivation (Energy-Based Graph RAG)")
    print("""
Neuroscience: Collins & Loftus (1975) — concepts in semantic memory are
nodes in a network. Hearing "doctor" unconsciously pre-activates "hospital",
"nurse", "medicine". Activation energy spreads along semantic edges, decaying
with distance, creating a 'spotlight' of primed context.

In AI: Vector search finds the exact match. Spreading activation finds the
associative neighborhood — enabling human-like contextual recall.
""")

    embedder = SimpleEmbeddingProvider(dim=128)
    config = SpreadingConfig(
        initial_energy=1.0,
        decay_rate=0.2,
        activation_threshold=0.1,
        max_hops=3,
        auto_connect_threshold=0.3,
    )
    sa = SpreadingActivation(embedder=embedder, config=config)

    # Build a semantic memory graph about a tech stack
    concepts = [
        "Docker containerization platform",
        "Kubernetes container orchestration",
        "AWS cloud infrastructure services",
        "SageMaker machine learning training",
        "Python machine learning development",
        "NumPy array computation library",
        "TensorFlow deep learning framework",
        "PostgreSQL relational database",
        "Redis in-memory cache",
        "React JavaScript frontend framework",
    ]

    print("Building semantic memory graph...")
    nodes = []
    for concept in concepts:
        node = sa.add_node(concept)
        nodes.append(node)

    # Auto-connect based on embedding similarity
    edges = sa.auto_connect(threshold=0.0)  # Connect all for demo
    print(f"  {len(nodes)} nodes created")
    print(f"  {edges} semantic connections built")

    # Simulate retrieving context for "I have a Docker problem"
    query = "Docker container won't start"
    query_embedding = embedder.embed(query)

    print(f"\nQuery: '{query}'")
    print("Activating spreading from closest matching node...")

    activated_nodes = sa.retrieve(query_embedding, top_k=5)

    print(f"\nActivated context (spreading activation spotlight):")
    for i, node in enumerate(activated_nodes, 1):
        bar_len = int(node.energy * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {i}. [{bar}] energy={node.energy:.3f}")
        print(f"     {node.content[:60]}")

    stats = sa.get_stats()
    print(
        f"\nGraph stats: {stats['total_nodes']} nodes, "
        f"{stats['total_edges']} edges, "
        f"{stats['total_activations']} activations"
    )


# ─── Full MnemosEngine Orchestration Demo ────────────────────────────────────


async def demo_full_engine() -> None:
    separator("Full MnemosEngine: All 5 Modules Orchestrated")
    print("""
Now we demonstrate the complete orchestration:
  1. process() → SurprisalGate → AffectiveRouter → Store + Graph
  2. retrieve() → SpreadingActivation + AffectiveRouter + MutableRAG
  3. consolidate() → SleepDaemon → Long-term semantic memory
""")

    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=0.0, min_content_length=5),
        sleep=SleepConfig(
            min_episodes_before_consolidation=3,
            consolidation_interval_seconds=0,
        ),
    )
    engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=128),
        store=InMemoryStore(),
    )

    # Simulate a user session
    session = [
        ("user", "I'm building an LLM-powered customer support chatbot"),
        ("user", "It uses GPT-4o for the main model and Pinecone for vector storage"),
        ("user", "The biggest challenge is handling conversation context across sessions"),
        ("user", "I need the bot to remember user preferences and past issues"),
        ("user", "Currently latency is around 800ms which is too slow"),
        ("user", "I'm considering switching to Llama 3 running locally via Ollama"),
    ]

    print("Processing session interactions...")
    for role, content in session:
        result = await engine.process(Interaction(role=role, content=content))
        icon = "✅" if result.stored else "⏭ "
        print(f"  {icon} {content[:65]}")

    # Retrieve context for a follow-up query
    print(f"\nRetrieving context for: 'local LLM deployment options'")
    memories = await engine.retrieve("local LLM deployment reduce latency", top_k=3)
    print("Top retrieved memories:")
    for i, chunk in enumerate(memories, 1):
        print(f"  {i}. {chunk.content[:70]}")

    # Trigger sleep consolidation
    print(f"\nTriggering sleep consolidation...")
    consolidation = await engine.consolidate()
    print(f"  Extracted {len(consolidation.facts_extracted)} facts")
    print(f"  Pruned {consolidation.chunks_pruned} raw episodes")

    # Final stats
    stats = engine.get_stats()
    print(f"\nFinal engine stats:")
    print(f"  Long-term memory: {stats['engine']['total_chunks']} chunks")
    print(f"  Spreading activation: {stats['spreading_activation']['total_nodes']} nodes")
    print(f"  Consolidations run: {stats['sleep_daemon']['total_consolidations']}")
    print(f"  Affective classifications: {stats['affective_router']['total_classified']}")


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("MNEMOS — Full Pipeline Demo")
    print("Five Biomimetic Memory Modules for LLMs")

    await demo_surprisal_gate()
    await demo_mutable_rag()
    await demo_affective_router()
    await demo_sleep_daemon()
    await demo_spreading_activation()
    await demo_full_engine()

    print("\n" + "=" * 60)
    print("  All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
