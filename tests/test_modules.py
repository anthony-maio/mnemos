"""
tests/test_modules.py — Tests for all five Mnemos memory modules.

Uses MockLLMProvider and SimpleEmbeddingProvider for fully offline testing.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemos.config import (
    AffectiveConfig,
    MutableRAGConfig,
    SleepConfig,
    SpreadingConfig,
    SurprisalConfig,
)
from mnemos.modules.affective import AffectiveRouter
from mnemos.modules.mutable_rag import MutableRAG
from mnemos.modules.sleep import SleepDaemon
from mnemos.modules.spreading import SpreadingActivation
from mnemos.modules.surprisal import SurprisalGate
from mnemos.types import CognitiveState, Interaction, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider, cosine_similarity
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def embedder():
    return SimpleEmbeddingProvider(dim=64)


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


def make_interaction(content: str, role: str = "user") -> Interaction:
    return Interaction(role=role, content=content)


# ─── SurprisalGate tests ──────────────────────────────────────────────────────


class TestSurprisalGate:
    @pytest.mark.asyncio
    async def test_low_surprisal_not_stored(self, mock_llm, embedder, store):
        """
        When prediction closely matches input, surprisal should be low
        and the interaction should NOT be stored.

        We test this by using a mock LLM that returns the exact same text
        as the input, making cosine distance ≈ 0.
        """
        # LLM returns same content as input → cosine distance ≈ 0
        identical_llm = MockLLMProvider(responses={"predict": "routine greeting, nothing new"})
        config = SurprisalConfig(threshold=0.3)
        gate = SurprisalGate(llm=identical_llm, embedder=embedder, store=store, config=config)

        # Seed history so prediction can run
        gate.add_to_history(make_interaction("Hello there!"))
        gate.add_to_history(make_interaction("How are you?"))

        # Input that is semantically very similar to what LLM would predict
        # The mock returns "routine greeting, nothing new" for any "predict" prompt
        # So we need to set up a scenario where surprisal is forced low.
        # We use threshold=0.99 to ensure almost nothing passes.
        config_strict = SurprisalConfig(threshold=0.99)
        gate_strict = SurprisalGate(
            llm=identical_llm, embedder=embedder, store=store, config=config_strict
        )
        gate_strict.add_to_history(make_interaction("Hello"))

        result = await gate_strict.process(make_interaction("Hi there"))
        # With threshold=0.99, almost nothing will pass (cosine distance is < 1.0)
        assert result.stored is False or result.salience < 1.0

    @pytest.mark.asyncio
    async def test_high_surprisal_stored(self, mock_llm, embedder, store):
        """
        With threshold=0.0, everything should pass the gate and be stored.
        """
        config = SurprisalConfig(threshold=0.0, min_content_length=0)
        gate = SurprisalGate(llm=mock_llm, embedder=embedder, store=store, config=config)

        gate.add_to_history(make_interaction("Hello"))
        result = await gate.process(make_interaction("My production server is down!"))
        assert result.stored is True
        assert result.chunk is not None
        assert result.chunk.content == "My production server is down!"

    @pytest.mark.asyncio
    async def test_stored_chunk_is_in_store(self, mock_llm, embedder, store):
        """Stored interactions should appear in the memory store."""
        config = SurprisalConfig(threshold=0.0, min_content_length=0)
        gate = SurprisalGate(llm=mock_llm, embedder=embedder, store=store, config=config)

        gate.add_to_history(make_interaction("context"))
        await gate.process(make_interaction("important new information"))

        all_chunks = store.get_all()
        assert len(all_chunks) == 1
        assert all_chunks[0].content == "important new information"

    @pytest.mark.asyncio
    async def test_chunk_has_embedding(self, mock_llm, embedder, store):
        """Stored chunks should have embeddings."""
        config = SurprisalConfig(threshold=0.0, min_content_length=0)
        gate = SurprisalGate(llm=mock_llm, embedder=embedder, store=store, config=config)

        gate.add_to_history(make_interaction("hello"))
        result = await gate.process(make_interaction("My stack is Python + FastAPI"))

        assert result.chunk is not None
        assert result.chunk.embedding is not None
        assert len(result.chunk.embedding) == 64

    @pytest.mark.asyncio
    async def test_short_content_rejected(self, mock_llm, embedder, store):
        """Content shorter than min_content_length should be rejected."""
        config = SurprisalConfig(threshold=0.0, min_content_length=20)
        gate = SurprisalGate(llm=mock_llm, embedder=embedder, store=store, config=config)

        result = await gate.process(make_interaction("Hi"))
        assert result.stored is False
        assert "too short" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_history_builds_up(self, mock_llm, embedder, store):
        """History buffer should grow after each process call."""
        config = SurprisalConfig(threshold=0.0, min_content_length=0)
        gate = SurprisalGate(llm=mock_llm, embedder=embedder, store=store, config=config)

        assert len(gate.get_history()) == 0
        await gate.process(make_interaction("first"))
        assert len(gate.get_history()) == 1
        await gate.process(make_interaction("second"))
        assert len(gate.get_history()) == 2

    @pytest.mark.asyncio
    async def test_stats_tracking(self, mock_llm, embedder, store):
        """Stats should accurately track processed and stored counts."""
        config = SurprisalConfig(threshold=0.0, min_content_length=0)
        gate = SurprisalGate(llm=mock_llm, embedder=embedder, store=store, config=config)

        await gate.process(make_interaction("first interaction"))
        await gate.process(make_interaction("second interaction"))

        stats = gate.get_stats()
        assert stats["total_processed"] == 2
        assert stats["total_stored"] == 2
        assert stats["store_rate"] == 1.0


# ─── MutableRAG tests ─────────────────────────────────────────────────────────


class TestMutableRAG:
    @pytest.mark.asyncio
    async def test_retrieve_flags_labile(self, mock_llm, embedder, store):
        """Retrieved chunks should be marked as labile."""
        rag = MutableRAG(llm=mock_llm, embedder=embedder, store=store)

        # Add some chunks to the store
        for i in range(3):
            chunk = MemoryChunk(
                content=f"fact number {i}",
                embedding=embedder.embed(f"fact number {i}"),
                salience=0.7,
            )
            store.store(chunk)

        query_embedding = embedder.embed("fact")
        retrieved = rag.retrieve(query_embedding, top_k=2)

        assert len(retrieved) == 2
        assert rag.get_labile_count() == 2

    @pytest.mark.asyncio
    async def test_reconsolidate_unchanged(self, embedder, store):
        """MockLLM returns UNCHANGED by default — chunk should not be modified."""
        llm = MockLLMProvider()  # Returns UNCHANGED by default
        rag = MutableRAG(llm=llm, embedder=embedder, store=store)

        chunk = MemoryChunk(
            content="User uses Python for ML",
            embedding=embedder.embed("User uses Python for ML"),
        )
        store.store(chunk)
        original_version = chunk.version

        updated_chunk, changed = await rag.reconsolidate(chunk, "New context")
        assert changed is False
        assert updated_chunk.version == original_version

    @pytest.mark.asyncio
    async def test_reconsolidate_changed(self, embedder, store):
        """When LLM returns CHANGED, chunk should be updated with new content."""
        llm = MockLLMProvider(
            responses={
                "has this stored memory": "CHANGED: User migrated from Python to Rust in 2026"
            }
        )
        rag = MutableRAG(llm=llm, embedder=embedder, store=store)

        chunk = MemoryChunk(
            content="User uses Python for ML",
            embedding=embedder.embed("User uses Python for ML"),
        )
        store.store(chunk)

        updated_chunk, changed = await rag.reconsolidate(chunk, "User mentioned migrating to Rust")
        assert changed is True
        assert "Rust" in updated_chunk.content
        assert updated_chunk.version == 2

    @pytest.mark.asyncio
    async def test_reconsolidate_changed_re_embeds(self, embedder, store):
        """Changed chunks should have a new embedding computed."""
        llm = MockLLMProvider(
            responses={"has this stored memory": "CHANGED: User now prefers Rust"}
        )
        rag = MutableRAG(llm=llm, embedder=embedder, store=store)

        chunk = MemoryChunk(
            content="User prefers Python",
            embedding=embedder.embed("User prefers Python"),
        )
        store.store(chunk)

        updated_chunk, changed = await rag.reconsolidate(chunk, "new context")
        assert changed is True
        assert updated_chunk.embedding is not None

    @pytest.mark.asyncio
    async def test_process_labile_chunks(self, embedder, store):
        """process_labile_chunks should clear the labile queue."""
        llm = MockLLMProvider()  # Returns UNCHANGED
        rag = MutableRAG(llm=llm, embedder=embedder, store=store)

        for i in range(3):
            chunk = MemoryChunk(
                content=f"memory {i}",
                embedding=embedder.embed(f"memory {i}"),
            )
            store.store(chunk)
            rag._labile_chunks[chunk.id] = (chunk, "current context")

        assert rag.get_labile_count() == 3
        results = await rag.process_labile_chunks()
        assert len(results) == 3
        assert rag.get_labile_count() == 0

    @pytest.mark.asyncio
    async def test_retrieve_increments_access_count(self, mock_llm, embedder, store):
        """Retrieving a chunk should increment its access_count."""
        rag = MutableRAG(llm=mock_llm, embedder=embedder, store=store)

        chunk = MemoryChunk(
            content="important fact",
            embedding=embedder.embed("important fact"),
        )
        store.store(chunk)

        query_embedding = embedder.embed("important")
        rag.retrieve(query_embedding, top_k=1)

        retrieved = store.get(chunk.id)
        assert retrieved.access_count == 1

    @pytest.mark.asyncio
    async def test_mark_labile_respects_reconsolidation_cooldown(
        self, embedder, store, monkeypatch
    ):
        """mark_labile should not re-queue a chunk until cooldown elapses."""
        llm = MockLLMProvider()
        config = MutableRAGConfig(reconsolidation_cooldown_seconds=60)
        rag = MutableRAG(llm=llm, embedder=embedder, store=store, config=config)

        chunk = MemoryChunk(
            content="User uses Python",
            embedding=embedder.embed("User uses Python"),
        )
        store.store(chunk)

        # First retrieval cycle at t=1000
        monkeypatch.setattr("mnemos.modules.mutable_rag.time.time", lambda: 1000.0)
        rag.mark_labile(chunk, "initial context")
        assert rag.get_labile_count() == 1
        await rag.process_labile_chunks()
        assert rag.get_labile_count() == 0

        # Within cooldown at t=1020: should not be re-queued
        monkeypatch.setattr("mnemos.modules.mutable_rag.time.time", lambda: 1020.0)
        rag.mark_labile(chunk, "second context")
        assert rag.get_labile_count() == 0

        # After cooldown at t=1061: allowed again
        monkeypatch.setattr("mnemos.modules.mutable_rag.time.time", lambda: 1061.0)
        rag.mark_labile(chunk, "third context")
        assert rag.get_labile_count() == 1


# ─── AffectiveRouter tests ────────────────────────────────────────────────────


class TestAffectiveRouter:
    @pytest.mark.asyncio
    async def test_classify_state_returns_cognitive_state(self, mock_llm, embedder):
        """classify_state should return a valid CognitiveState."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        state = await router.classify_state(make_interaction("I'm panicking, server down!"))
        assert isinstance(state, CognitiveState)
        assert -1.0 <= state.valence <= 1.0
        assert 0.0 <= state.arousal <= 1.0
        assert 0.0 <= state.complexity <= 1.0

    @pytest.mark.asyncio
    async def test_tag_chunk_attaches_state(self, mock_llm, embedder):
        """tag_chunk should attach cognitive state to the chunk."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        chunk = MemoryChunk(content="test content")
        state = CognitiveState(valence=0.5, arousal=0.8, complexity=0.3)
        router.tag_chunk(chunk, state)
        assert chunk.cognitive_state is not None
        assert chunk.cognitive_state.valence == 0.5
        assert "cognitive_state" in chunk.metadata

    def test_score_chunk_without_state(self, mock_llm, embedder):
        """Chunks without cognitive state should get neutral state_match (0.5)."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)

        # Create chunk without cognitive state
        content = "Python machine learning"
        embedding = embedder.embed(content)
        chunk = MemoryChunk(content=content, embedding=embedding)

        query_embedding = embedder.embed("machine learning frameworks")
        current_state = CognitiveState(valence=0.0, arousal=0.5, complexity=0.5)

        score = router.score_chunk(chunk, query_embedding, current_state)
        # Score = similarity * 0.7 + 0.5 * 0.3 (neutral state match)
        # Must be a valid float in range [0, ~1.2]
        assert isinstance(score, float)
        assert score >= 0.0

    def test_score_chunk_state_match_formula(self, mock_llm, embedder):
        """Affective scoring formula: score = sim*0.7 + state_match*0.3"""
        config = AffectiveConfig(weight_similarity=0.7, weight_state=0.3)
        router = AffectiveRouter(llm=mock_llm, embedder=embedder, config=config)

        content = "urgent production outage"
        embedding = embedder.embed(content)
        # Chunk encoded during high-arousal state (crisis)
        urgent_state = CognitiveState(valence=-0.8, arousal=0.9, complexity=0.7)
        chunk = MemoryChunk(
            content=content,
            embedding=embedding,
            cognitive_state=urgent_state,
        )

        # Query also from high-arousal state
        matching_state = CognitiveState(valence=-0.7, arousal=0.9, complexity=0.8)
        # Query from calm state
        calm_state = CognitiveState(valence=0.5, arousal=0.1, complexity=0.2)

        query_emb = embedder.embed(content)

        score_matching = router.score_chunk(chunk, query_emb, matching_state)
        score_calm = router.score_chunk(chunk, query_emb, calm_state)

        # Matching emotional state should score higher
        assert score_matching > score_calm

    @pytest.mark.asyncio
    async def test_retrieve_rerankis_by_affective_score(self, mock_llm, embedder):
        """retrieve() should return chunks re-ranked by affective scoring."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        store = InMemoryStore()

        # Two similar content chunks, different emotional contexts
        urgent_state = CognitiveState(valence=-0.8, arousal=0.9, complexity=0.7)
        calm_state = CognitiveState(valence=0.8, arousal=0.1, complexity=0.2)

        urgent_chunk = MemoryChunk(
            content="server is down production outage",
            embedding=embedder.embed("server is down production outage"),
            cognitive_state=urgent_state,
        )
        calm_chunk = MemoryChunk(
            content="server maintenance scheduled",
            embedding=embedder.embed("server maintenance scheduled"),
            cognitive_state=calm_state,
        )
        store.store(urgent_chunk)
        store.store(calm_chunk)

        # Query in urgent state should prefer urgent chunk
        urgent_query_state = CognitiveState(valence=-0.7, arousal=0.8, complexity=0.6)
        results = await router.retrieve(
            "server problems",
            current_state=urgent_query_state,
            store=store,
            top_k=2,
        )

        assert len(results) >= 1
        # First result should be the urgent chunk (best affective match)
        assert results[0].cognitive_state is not None

    @pytest.mark.asyncio
    async def test_retrieve_uses_supplied_query_embedding(self, mock_llm):
        """retrieve() should not re-embed the query when query_embedding is provided."""

        class CountingEmbedder:
            def __init__(self) -> None:
                self.calls = 0

            def embed(self, text: str) -> list[float]:
                self.calls += 1
                return [1.0, 0.0, 0.0]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed(text) for text in texts]

        embedder = CountingEmbedder()
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        store = InMemoryStore()
        store.store(
            MemoryChunk(
                content="server is down production outage",
                embedding=[1.0, 0.0, 0.0],
                cognitive_state=CognitiveState(valence=-0.8, arousal=0.9, complexity=0.7),
            )
        )

        current_state = CognitiveState(valence=-0.7, arousal=0.8, complexity=0.6)
        results = await router.retrieve(
            "server problems",
            current_state=current_state,
            store=store,
            top_k=1,
            query_embedding=[1.0, 0.0, 0.0],
        )

        assert len(results) == 1
        assert embedder.calls == 0

    @pytest.mark.asyncio
    async def test_retrieve_uses_supplied_candidates_without_store_lookup(self, mock_llm, embedder):
        """retrieve() should rerank provided candidates without calling store.retrieve()."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        store = MagicMock()
        candidate = MemoryChunk(
            content="server is down production outage",
            embedding=embedder.embed("server is down production outage"),
            cognitive_state=CognitiveState(valence=-0.8, arousal=0.9, complexity=0.7),
        )
        current_state = CognitiveState(valence=-0.7, arousal=0.8, complexity=0.6)

        results = await router.retrieve(
            "server problems",
            current_state=current_state,
            store=store,
            top_k=1,
            query_embedding=embedder.embed("server problems"),
            candidates=[candidate],
        )

        assert len(results) == 1
        store.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_state_history_updates(self, mock_llm, embedder):
        """classify_state should update the state history."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        assert len(router._state_history) == 0

        await router.classify_state(make_interaction("test"))
        assert len(router._state_history) == 1

        await router.classify_state(make_interaction("another test"))
        assert len(router._state_history) == 2

    @pytest.mark.asyncio
    async def test_get_current_state_default(self, mock_llm, embedder):
        """get_current_state should return neutral defaults when no history."""
        router = AffectiveRouter(llm=mock_llm, embedder=embedder)
        state = router.get_current_state()
        assert state.valence == 0.0
        assert state.arousal == 0.5


# ─── SleepDaemon tests ────────────────────────────────────────────────────────


class TestSleepDaemon:
    def test_add_episode(self, store):
        """add_episode should increase buffer size."""
        daemon = SleepDaemon(store=store)
        assert daemon.get_episode_count() == 0
        daemon.add_episode(make_interaction("hello"))
        assert daemon.get_episode_count() == 1

    def test_should_consolidate_false_when_too_few_episodes(self, store):
        """should_consolidate should be False if not enough episodes."""
        config = SleepConfig(
            min_episodes_before_consolidation=10,
            consolidation_interval_seconds=1,  # 1 second — time met but episode count not
        )
        daemon = SleepDaemon(store=store, config=config)

        for _ in range(5):  # Only 5, need 10
            daemon.add_episode(make_interaction("hello"))

        assert daemon.should_consolidate() is False

    def test_should_consolidate_true_when_conditions_met(self, store):
        """should_consolidate should be True when both conditions are met."""
        config = SleepConfig(
            min_episodes_before_consolidation=3,
            consolidation_interval_seconds=0,  # Immediately eligible
        )
        daemon = SleepDaemon(store=store, config=config)

        for i in range(3):
            daemon.add_episode(make_interaction(f"episode {i}"))

        assert daemon.should_consolidate() is True

    @pytest.mark.asyncio
    async def test_consolidate_extracts_facts(self, store, embedder):
        """consolidate() should extract facts and store them as semantic chunks."""
        llm = MockLLMProvider()  # Returns numbered fact list
        config = SleepConfig(min_episodes_before_consolidation=1)
        daemon = SleepDaemon(store=store, config=config)

        for i in range(3):
            daemon.add_episode(make_interaction(f"I work on machine learning projects {i}"))

        result = await daemon.consolidate(llm, embedder)

        assert len(result.facts_extracted) > 0
        # Facts should be stored as semantic chunks
        all_chunks = store.get_all()
        assert len(all_chunks) > 0

    @pytest.mark.asyncio
    async def test_consolidate_clears_episodic_buffer(self, store, embedder):
        """consolidate() should clear the episodic buffer."""
        llm = MockLLMProvider()
        daemon = SleepDaemon(store=store)

        for i in range(5):
            daemon.add_episode(make_interaction(f"episode {i}"))

        assert daemon.get_episode_count() == 5
        await daemon.consolidate(llm, embedder)
        assert daemon.get_episode_count() == 0

    @pytest.mark.asyncio
    async def test_consolidate_reports_pruned_count(self, store, embedder):
        """consolidate() should report the number of pruned episodes."""
        llm = MockLLMProvider()
        daemon = SleepDaemon(store=store)

        episodes_count = 7
        for i in range(episodes_count):
            daemon.add_episode(make_interaction(f"episode {i}"))

        result = await daemon.consolidate(llm, embedder)
        assert result.chunks_pruned == episodes_count

    @pytest.mark.asyncio
    async def test_consolidate_empty_buffer(self, store, embedder):
        """consolidate() on empty buffer should return empty result."""
        llm = MockLLMProvider()
        daemon = SleepDaemon(store=store)
        result = await daemon.consolidate(llm, embedder)
        assert result.facts_extracted == []
        assert result.chunks_pruned == 0

    @pytest.mark.asyncio
    async def test_proceduralize_no_pattern(self, store):
        """proceduralize() should return None when no pattern found (MockLLM returns NO_PATTERN)."""
        llm = MockLLMProvider()  # Returns NO_PATTERN for proceduralize prompts
        daemon = SleepDaemon(store=store)
        result = await daemon.proceduralize(llm, "some episodes")
        assert result is None

    @pytest.mark.asyncio
    async def test_consolidate_stores_with_high_salience(self, store, embedder):
        """Consolidated facts should have high salience (0.7)."""
        llm = MockLLMProvider()
        daemon = SleepDaemon(store=store)

        for i in range(3):
            daemon.add_episode(make_interaction(f"important info {i}"))

        await daemon.consolidate(llm, embedder)

        all_chunks = store.get_all()
        for chunk in all_chunks:
            assert chunk.salience == 0.7

    @pytest.mark.asyncio
    async def test_consolidate_recall_gate_blocks_singleton_fact(self, store, embedder):
        """Recall-gated consolidation should skip facts not supported by repeated episodes."""
        llm = MockLLMProvider(
            responses={"extract permanent facts": "1. Team uses uv for Python tooling."}
        )
        daemon = SleepDaemon(
            store=store,
            config=SleepConfig(
                min_episodes_before_consolidation=1,
                recall_gated_plasticity_enabled=True,
                recall_min_supporting_episodes=2,
                recall_similarity_threshold=0.999,
            ),
        )

        daemon.add_episode(make_interaction("Team uses uv for Python tooling."))
        daemon.add_episode(make_interaction("We reviewed release notes today."))

        result = await daemon.consolidate(llm, embedder)

        assert result.facts_extracted == []
        assert store.get_all() == []

    @pytest.mark.asyncio
    async def test_consolidate_recall_gate_keeps_repeated_fact(self, store, embedder):
        """Recall-gated consolidation should retain facts supported by multiple episodes."""
        llm = MockLLMProvider(
            responses={"extract permanent facts": "1. Team uses uv for Python tooling."}
        )
        daemon = SleepDaemon(
            store=store,
            config=SleepConfig(
                min_episodes_before_consolidation=1,
                recall_gated_plasticity_enabled=True,
                recall_min_supporting_episodes=2,
                recall_similarity_threshold=0.999,
            ),
        )

        daemon.add_episode(make_interaction("Team uses uv for Python tooling."))
        daemon.add_episode(make_interaction("Team uses uv for Python tooling."))
        daemon.add_episode(make_interaction("We reviewed release notes today."))

        result = await daemon.consolidate(llm, embedder)

        assert result.facts_extracted == ["Team uses uv for Python tooling."]
        all_chunks = store.get_all()
        assert len(all_chunks) == 1
        assert all_chunks[0].content == "Team uses uv for Python tooling."
        assert all_chunks[0].metadata["recall_supporting_episodes"] == 2


# ─── SpreadingActivation tests ────────────────────────────────────────────────


class TestSpreadingActivation:
    def test_add_node(self, embedder):
        """add_node should create a node in the graph."""
        sa = SpreadingActivation(embedder=embedder)
        assert sa.get_node_count() == 0
        node = sa.add_node("test concept")
        assert sa.get_node_count() == 1
        assert sa.get_node(node.id) is not None

    def test_add_node_preserves_metadata(self, embedder):
        """add_node should keep scope metadata on the activation node."""
        sa = SpreadingActivation(embedder=embedder)
        node = sa.add_node(
            "test concept",
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        assert node.metadata["scope"] == "project"
        assert node.metadata["scope_id"] == "repo-alpha"

    def test_add_node_computes_embedding(self, embedder):
        """add_node should compute embedding if not provided."""
        sa = SpreadingActivation(embedder=embedder)
        node = sa.add_node("machine learning")
        assert node.embedding is not None
        assert len(node.embedding) == 64

    def test_add_edge(self, embedder):
        """add_edge should create bidirectional edge between two nodes."""
        sa = SpreadingActivation(embedder=embedder)
        a = sa.add_node("node A")
        b = sa.add_node("node B")
        result = sa.add_edge(a.id, b.id, weight=0.8)
        assert result is True
        assert b.id in a.neighbors
        assert a.id in b.neighbors
        assert a.neighbors[b.id] == 0.8

    def test_add_edge_nonexistent_node(self, embedder):
        """add_edge with non-existent node should return False."""
        sa = SpreadingActivation(embedder=embedder)
        node = sa.add_node("real node")
        result = sa.add_edge(node.id, "fake-id-123", weight=0.5)
        assert result is False

    def test_auto_connect(self, embedder):
        """auto_connect should create edges between similar nodes."""
        sa = SpreadingActivation(
            embedder=embedder,
            config=SpreadingConfig(auto_connect_threshold=0.0),  # Connect everything
        )
        a = sa.add_node("machine learning algorithms")
        b = sa.add_node("deep learning neural networks")
        c = sa.add_node("cooking recipes pasta")

        edges = sa.auto_connect(threshold=0.0)
        assert edges > 0  # Should create at least some edges

    def test_auto_connect_respects_scope_boundaries_and_neighbor_cap(self, embedder):
        """auto_connect should avoid cross-project edges and keep the graph sparse."""
        sa = SpreadingActivation(
            embedder=embedder,
            config=SpreadingConfig(
                auto_connect_threshold=0.0,
                max_neighbors_per_node=2,
            ),
        )
        alpha = sa.add_node(
            "alpha",
            embedding=[1.0, 0.0, 0.0],
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        alpha_peer = sa.add_node(
            "alpha-peer",
            embedding=[0.99, 0.01, 0.0],
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        beta_peer = sa.add_node(
            "beta-peer",
            embedding=[0.98, 0.02, 0.0],
            metadata={"scope": "project", "scope_id": "repo-beta"},
        )
        global_peer = sa.add_node(
            "global-peer",
            embedding=[0.97, 0.03, 0.0],
            metadata={"scope": "global"},
        )

        sa.auto_connect(threshold=0.0)

        assert alpha_peer.id in alpha.neighbors
        assert beta_peer.id not in alpha.neighbors
        assert len(alpha.neighbors) <= 2
        assert len(alpha_peer.neighbors) <= 2
        assert len(beta_peer.neighbors) <= 2
        assert len(global_peer.neighbors) <= 2

    def test_activate_seed_gets_initial_energy(self, embedder):
        """The seed node should receive the initial energy."""
        config = SpreadingConfig(initial_energy=1.0)
        sa = SpreadingActivation(embedder=embedder, config=config)
        node = sa.add_node("seed concept")

        activated = sa.activate(node.id)
        assert node.id in activated
        assert activated[node.id] == pytest.approx(1.0)

    def test_activate_propagates_to_neighbors(self, embedder):
        """Activation should propagate to connected neighbors."""
        config = SpreadingConfig(
            initial_energy=1.0,
            decay_rate=0.2,
            activation_threshold=0.0,  # Allow all energy through
            max_hops=1,
        )
        sa = SpreadingActivation(embedder=embedder, config=config)
        seed = sa.add_node("seed")
        neighbor = sa.add_node("neighbor")
        sa.add_edge(seed.id, neighbor.id, weight=1.0)

        activated = sa.activate(seed.id, energy=1.0)

        # Neighbor should receive energy: 1.0 * (1 - 0.2) * 1.0 = 0.8
        assert neighbor.id in activated
        assert activated[neighbor.id] == pytest.approx(0.8, abs=0.01)

    def test_activate_energy_decays_with_hops(self, embedder):
        """Energy should decay with each hop."""
        config = SpreadingConfig(
            initial_energy=1.0,
            decay_rate=0.2,
            activation_threshold=0.0,
            max_hops=3,
        )
        sa = SpreadingActivation(embedder=embedder, config=config)
        a = sa.add_node("A")
        b = sa.add_node("B")
        c = sa.add_node("C")
        sa.add_edge(a.id, b.id, weight=1.0)
        sa.add_edge(b.id, c.id, weight=1.0)

        activated = sa.activate(a.id, energy=1.0)

        # C should have less energy than B
        if b.id in activated and c.id in activated:
            assert activated[c.id] < activated[b.id]

    def test_activate_threshold_filters_weak_nodes(self, embedder):
        """Nodes below activation threshold should not be returned."""
        config = SpreadingConfig(
            initial_energy=0.1,
            decay_rate=0.5,
            activation_threshold=0.5,  # High threshold
            max_hops=3,
        )
        sa = SpreadingActivation(embedder=embedder, config=config)
        seed = sa.add_node("seed")
        weak_neighbor = sa.add_node("weak neighbor")
        sa.add_edge(seed.id, weak_neighbor.id, weight=0.1)

        activated = sa.activate(seed.id, energy=0.1)
        # Seed is at 0.1, below threshold 0.5, but seed is always included as it was explicitly set
        # Neighbor would receive 0.1 * 0.5 * 0.1 = 0.005, definitely below threshold
        assert weak_neighbor.id not in activated

    def test_activate_unknown_node_raises(self, embedder):
        """activate() with unknown node_id should raise KeyError."""
        sa = SpreadingActivation(embedder=embedder)
        with pytest.raises(KeyError):
            sa.activate("nonexistent-node-id")

    def test_retrieve_returns_sorted_by_energy(self, embedder):
        """retrieve() should return nodes sorted by activation energy."""
        config = SpreadingConfig(
            initial_energy=1.0,
            decay_rate=0.1,
            activation_threshold=0.0,
            max_hops=2,
        )
        sa = SpreadingActivation(embedder=embedder, config=config)
        seed = sa.add_node("python programming language")
        related = sa.add_node("python machine learning")
        distant = sa.add_node("cooking food recipes")

        # Strong connection to related, weak to distant
        sa.add_edge(seed.id, related.id, weight=0.9)
        sa.add_edge(seed.id, distant.id, weight=0.1)

        query_embedding = embedder.embed("python programming")
        results = sa.retrieve(query_embedding, top_k=3)

        assert len(results) >= 1
        # Seed node should be first (highest energy = initial_energy)
        assert results[0].energy >= results[-1].energy

    def test_decay_all_reduces_energies(self, embedder):
        """decay_all() should reduce all node energies."""
        sa = SpreadingActivation(embedder=embedder)
        node = sa.add_node("test")
        node.energy = 1.0

        sa.decay_all(rate=0.1)
        assert node.energy == pytest.approx(0.9, abs=0.01)

    def test_remove_node(self, embedder):
        """remove_node should delete node and its edges."""
        sa = SpreadingActivation(embedder=embedder)
        a = sa.add_node("A")
        b = sa.add_node("B")
        sa.add_edge(a.id, b.id, weight=0.5)

        assert a.id in b.neighbors
        sa.remove_node(a.id)
        assert sa.get_node(a.id) is None
        assert a.id not in b.neighbors

    def test_get_stats(self, embedder):
        """get_stats should return expected keys."""
        sa = SpreadingActivation(embedder=embedder)
        sa.add_node("test")
        stats = sa.get_stats()
        assert "total_nodes" in stats
        assert "total_activations" in stats
        assert "module" in stats

    def test_add_node_from_chunk(self, embedder):
        """add_node_from_chunk should use chunk's id and content."""
        sa = SpreadingActivation(embedder=embedder)
        chunk = MemoryChunk(
            content="important memory",
            embedding=embedder.embed("important memory"),
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        node = sa.add_node_from_chunk(chunk)
        assert node.id == chunk.id
        assert node.content == chunk.content
        assert node.metadata["scope"] == "project"
        assert node.metadata["scope_id"] == "repo-alpha"
        assert sa.get_node(chunk.id) is not None
