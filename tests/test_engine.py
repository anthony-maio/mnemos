"""
tests/test_engine.py — Integration tests for MnemosEngine.

Tests the full pipeline: process → retrieve → consolidate.
All tests use MockLLMProvider + SimpleEmbeddingProvider + InMemoryStore
for fully offline execution.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mnemos.config import (
    AffectiveConfig,
    MemoryGovernanceConfig,
    MnemosConfig,
    SleepConfig,
    SurprisalConfig,
)
from mnemos.engine import MnemosEngine
from mnemos.types import CognitiveState, Interaction, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    """A fully configured MnemosEngine with all-local providers."""
    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
        sleep=SleepConfig(
            min_episodes_before_consolidation=3,
            consolidation_interval_seconds=0,
        ),
    )
    return MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=InMemoryStore(),
    )


@pytest.fixture
def strict_engine():
    """Engine with strict surprisal gate (threshold=0.99 — almost nothing stored)."""
    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=0.99, min_content_length=5),
    )
    return MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=InMemoryStore(),
    )


def make_interaction(content: str, role: str = "user") -> Interaction:
    return Interaction(role=role, content=content)


# ─── Basic process tests ──────────────────────────────────────────────────────


class TestEngineProcess:
    @pytest.mark.asyncio
    async def test_process_stores_interaction(self, engine):
        """process() should store interactions when surprisal gate passes."""
        result = await engine.process(make_interaction("I use Python for machine learning"))
        assert result.stored is True
        assert result.chunk is not None

    @pytest.mark.asyncio
    async def test_process_adds_to_episodic_buffer(self, engine):
        """process() should always add to episodic buffer."""
        initial_count = engine.sleep_daemon.get_episode_count()
        await engine.process(make_interaction("test interaction"))
        assert engine.sleep_daemon.get_episode_count() == initial_count + 1

    @pytest.mark.asyncio
    async def test_process_tags_chunk_with_cognitive_state(self, engine):
        """process() should attach a CognitiveState to stored chunks."""
        result = await engine.process(make_interaction("I'm really stressed about this bug!"))
        assert result.stored is True
        assert result.chunk is not None
        assert result.chunk.cognitive_state is not None

    @pytest.mark.asyncio
    async def test_process_adds_to_spreading_activation_graph(self, engine):
        """process() should add stored chunks to the spreading activation graph."""
        initial_nodes = engine.spreading_activation.get_node_count()
        result = await engine.process(make_interaction("Docker containerization and Kubernetes"))
        if result.stored:
            assert engine.spreading_activation.get_node_count() == initial_nodes + 1

    @pytest.mark.asyncio
    async def test_process_multiple_interactions(self, engine):
        """Processing multiple interactions should accumulate in memory."""
        interactions = [
            "I work with Python and TensorFlow",
            "My production database is PostgreSQL",
            "I deploy to AWS using Terraform",
            "My team follows agile methodology",
        ]
        results = []
        for content in interactions:
            result = await engine.process(make_interaction(content))
            results.append(result)

        stored_count = sum(1 for r in results if r.stored)
        assert stored_count > 0

        all_chunks = engine.store.get_all()
        assert len(all_chunks) == stored_count

    @pytest.mark.asyncio
    async def test_process_returns_process_result(self, engine):
        """process() should return a ProcessResult with expected fields."""
        result = await engine.process(make_interaction("test"))
        assert hasattr(result, "stored")
        assert hasattr(result, "chunk")
        assert hasattr(result, "salience")
        assert hasattr(result, "reason")

    @pytest.mark.asyncio
    async def test_process_chunk_has_embedding(self, engine):
        """Stored chunks should have embeddings."""
        result = await engine.process(make_interaction("I use FastAPI for APIs"))
        if result.stored:
            assert result.chunk.embedding is not None
            assert len(result.chunk.embedding) == 64

    @pytest.mark.asyncio
    async def test_process_blocks_secret_content_via_safety_firewall(self, engine):
        """Secret-like content should be blocked from long-term storage."""
        result = await engine.process(
            make_interaction("Store this token: api_key=supersecretvalue123")
        )
        assert result.stored is False
        assert "safety policy" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_process_redacts_pii_content_via_safety_firewall(self, engine):
        """PII-like content should be redacted before storage under default policy."""
        result = await engine.process(
            make_interaction("My email is jane@example.com for follow-up")
        )
        assert result.stored is True
        assert result.chunk is not None
        assert "REDACTED_EMAIL" in result.chunk.content


# ─── Basic retrieve tests ─────────────────────────────────────────────────────


class TestEngineRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_relevant_chunks(self, engine):
        """retrieve() should return semantically relevant chunks."""
        # Store some facts first
        await engine.process(make_interaction("I use Python for machine learning"))
        await engine.process(make_interaction("I deploy applications with Docker"))
        await engine.process(make_interaction("I prefer PostgreSQL for databases"))

        # Query for Python-related content
        results = await engine.retrieve("Python programming language", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self, engine):
        """retrieve() on empty store should return empty list."""
        results = await engine.retrieve("anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, engine):
        """retrieve() should return at most top_k results."""
        for i in range(10):
            await engine.process(make_interaction(f"fact number {i}: I like programming"))

        results = await engine.retrieve("programming", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_results_are_memory_chunks(self, engine):
        """retrieve() should return MemoryChunk objects."""
        await engine.process(make_interaction("test memory content"))
        results = await engine.retrieve("test", top_k=1)

        for chunk in results:
            assert isinstance(chunk, MemoryChunk)
            assert chunk.content
            assert chunk.id

    @pytest.mark.asyncio
    async def test_retrieve_full_pipeline(self, engine):
        """Test the complete retrieve pipeline with spreading activation."""
        # Store related facts that should cluster
        facts = [
            "Python is a high-level programming language",
            "Python is widely used for machine learning",
            "NumPy and Pandas are key Python ML libraries",
            "JavaScript is used for web development",
            "React is a JavaScript framework",
        ]
        for fact in facts:
            await engine.process(make_interaction(fact))

        # Query for Python-related content
        results = await engine.retrieve("Python ML libraries", top_k=3)
        assert len(results) >= 1
        # All results should be MemoryChunks with content
        for r in results:
            assert isinstance(r, MemoryChunk)
            assert len(r.content) > 0

    @pytest.mark.asyncio
    async def test_retrieve_reuses_single_store_candidate_pool(self):
        """retrieve() should fetch semantic candidates once and reuse them for affective reranking."""

        class CountingInMemoryStore(InMemoryStore):
            def __init__(self) -> None:
                super().__init__()
                self.retrieve_calls = 0

            def retrieve(self, query_embedding, top_k=5, filter_fn=None):
                self.retrieve_calls += 1
                return super().retrieve(query_embedding, top_k=top_k, filter_fn=filter_fn)

        store = CountingInMemoryStore()
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            ),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        await engine.process(make_interaction("Python services deploy with uv and Docker"))
        await engine.process(make_interaction("PostgreSQL backs the production app"))

        results = await engine.retrieve("python deploy tooling", top_k=2, reconsolidate=False)

        assert results
        assert store.retrieve_calls == 1


class TestEngineScopedMemory:
    @pytest.mark.asyncio
    async def test_process_persists_scope_metadata(self, engine):
        """process() should persist scope metadata on stored chunks."""
        result = await engine.process(
            make_interaction("workspace-specific memory"),
            scope="workspace",
            scope_id="ws-1",
        )
        assert result.stored is True
        assert result.chunk is not None
        assert result.chunk.metadata.get("scope") == "workspace"
        assert result.chunk.metadata.get("scope_id") == "ws-1"

    @pytest.mark.asyncio
    async def test_retrieve_filters_by_scope_id_and_allowed_scopes(self, engine):
        """retrieve() should isolate project scope while allowing selected global memory."""
        await engine.process(
            make_interaction("alpha project uses terraform"),
            scope="project",
            scope_id="alpha",
        )
        await engine.process(
            make_interaction("beta project uses ansible"),
            scope="project",
            scope_id="beta",
        )
        await engine.process(
            make_interaction("global policy requires code review"),
            scope="global",
        )

        results = await engine.retrieve(
            "project deployment tooling",
            top_k=10,
            reconsolidate=False,
            current_scope="project",
            scope_id="alpha",
            allowed_scopes=("project", "global"),
        )

        ids = {chunk.metadata.get("scope_id") for chunk in results}
        scopes = {chunk.metadata.get("scope") for chunk in results}
        assert "beta" not in ids
        assert "alpha" in ids
        assert "global" in scopes

    @pytest.mark.asyncio
    async def test_retrieve_prefers_current_scope_over_global(self, engine):
        """Current scope matches should rank ahead of global memories when equally relevant."""
        await engine.process(
            make_interaction("deploy checklist includes smoke tests"),
            scope="project",
            scope_id="alpha",
        )
        await engine.process(
            make_interaction("deploy checklist includes changelog updates"),
            scope="global",
        )

        results = await engine.retrieve(
            "deploy checklist",
            top_k=2,
            reconsolidate=False,
            current_scope="project",
            scope_id="alpha",
            allowed_scopes=("project", "global"),
        )
        assert len(results) == 2
        assert results[0].metadata.get("scope") == "project"

    @pytest.mark.asyncio
    async def test_consolidate_preserves_scope_and_isolates_projects(self) -> None:
        """Sleep consolidation should preserve source scope and never leak facts across projects."""
        llm = MockLLMProvider(
            responses={
                "repo alpha uses terraform": "1. Repo alpha uses Terraform modules.",
                "repo beta uses ansible": "1. Repo beta uses Ansible playbooks.",
            }
        )
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
                sleep=SleepConfig(
                    min_episodes_before_consolidation=1,
                    consolidation_interval_seconds=0,
                ),
            ),
            llm=llm,
            embedder=SimpleEmbeddingProvider(dim=64),
            store=InMemoryStore(),
        )

        await engine.process(
            make_interaction("Repo alpha uses Terraform."),
            scope="project",
            scope_id="repo-alpha",
        )
        await engine.process(
            make_interaction("Repo beta uses Ansible."),
            scope="project",
            scope_id="repo-beta",
        )

        result = await engine.consolidate()
        assert "Repo alpha uses Terraform modules." in result.facts_extracted
        assert "Repo beta uses Ansible playbooks." in result.facts_extracted

        consolidated = [
            chunk
            for chunk in engine.store.get_all()
            if chunk.metadata.get("source") == "sleep_consolidation"
        ]
        assert consolidated
        assert {
            (chunk.metadata.get("scope"), chunk.metadata.get("scope_id")) for chunk in consolidated
        } == {
            ("project", "repo-alpha"),
            ("project", "repo-beta"),
        }

        alpha_results = await engine.retrieve(
            "terraform modules",
            top_k=5,
            reconsolidate=False,
            current_scope="project",
            scope_id="repo-alpha",
            allowed_scopes=("project",),
        )
        assert alpha_results
        assert any("Terraform modules" in chunk.content for chunk in alpha_results)
        assert not any(chunk.metadata.get("scope_id") == "repo-beta" for chunk in alpha_results)

        beta_results = await engine.retrieve(
            "ansible playbooks",
            top_k=5,
            reconsolidate=False,
            current_scope="project",
            scope_id="repo-beta",
            allowed_scopes=("project",),
        )
        assert beta_results
        assert any("Ansible playbooks" in chunk.content for chunk in beta_results)
        assert not any(chunk.metadata.get("scope_id") == "repo-alpha" for chunk in beta_results)


class TestEngineGovernance:
    @pytest.mark.asyncio
    async def test_capture_mode_hooks_only_blocks_manual_process(self) -> None:
        config = MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            governance=MemoryGovernanceConfig(capture_mode="hooks_only"),
        )
        engine = MnemosEngine(
            config=config,
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=InMemoryStore(),
        )

        result = await engine.process(make_interaction("manual memory content"))
        assert result.stored is False
        assert "capture mode policy" in result.reason.lower()

        hook_interaction = Interaction(
            role="user",
            content="hook memory content",
            metadata={"source": "claude_hook", "hook_event": "UserPromptSubmit"},
        )
        hook_result = await engine.process(hook_interaction)
        assert hook_result.stored is True

    @pytest.mark.asyncio
    async def test_max_chunks_per_scope_cap_prunes_oldest(self) -> None:
        config = MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            governance=MemoryGovernanceConfig(max_chunks_per_scope=2),
        )
        engine = MnemosEngine(
            config=config,
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=InMemoryStore(),
        )

        await engine.process(make_interaction("fact one"), scope="project", scope_id="alpha")
        await engine.process(make_interaction("fact two"), scope="project", scope_id="alpha")
        await engine.process(make_interaction("fact three"), scope="project", scope_id="alpha")

        scoped_chunks = [
            chunk
            for chunk in engine.store.get_all()
            if chunk.metadata.get("scope") == "project"
            and chunk.metadata.get("scope_id") == "alpha"
        ]
        assert len(scoped_chunks) <= 2

    @pytest.mark.asyncio
    async def test_retention_ttl_prunes_old_chunks(self) -> None:
        config = MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            governance=MemoryGovernanceConfig(retention_ttl_days=1),
        )
        store = InMemoryStore()
        engine = MnemosEngine(
            config=config,
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        old_time = datetime.now(timezone.utc) - timedelta(days=5)
        stale_chunk = MemoryChunk(
            content="stale memory",
            embedding=engine.embedder.embed("stale memory"),
            metadata={"scope": "project", "scope_id": "alpha"},
            created_at=old_time,
            updated_at=old_time,
        )
        store.store(stale_chunk)

        await engine.process(make_interaction("fresh memory"), scope="project", scope_id="alpha")
        ids = {chunk.id for chunk in store.get_all()}
        assert stale_chunk.id not in ids


# ─── Consolidation tests ──────────────────────────────────────────────────────


class TestEngineConsolidate:
    @pytest.mark.asyncio
    async def test_consolidate_extracts_facts(self, engine):
        """consolidate() should extract facts from episodic buffer."""
        # Add episodes to buffer
        for i in range(3):
            await engine.process(
                make_interaction(f"User preference {i}: I prefer concise responses")
            )

        result = await engine.consolidate()
        assert len(result.facts_extracted) > 0

    @pytest.mark.asyncio
    async def test_consolidate_clears_episodic_buffer(self, engine):
        """consolidate() should clear the episodic buffer."""
        for i in range(5):
            await engine.process(make_interaction(f"episode {i}"))

        assert engine.sleep_daemon.get_episode_count() == 5
        await engine.consolidate()
        assert engine.sleep_daemon.get_episode_count() == 0

    @pytest.mark.asyncio
    async def test_consolidate_stores_semantic_chunks(self, engine):
        """consolidate() should store semantic fact chunks in long-term memory."""
        for i in range(4):
            await engine.process(make_interaction(f"I work on ML project step {i}"))

        initial_store_size = len(engine.store.get_all())
        await engine.consolidate()
        final_store_size = len(engine.store.get_all())

        # Should have added new semantic chunks
        assert final_store_size >= initial_store_size

    @pytest.mark.asyncio
    async def test_consolidate_empty_buffer(self, engine):
        """consolidate() with empty buffer should return empty result."""
        result = await engine.consolidate()
        assert result.facts_extracted == []
        assert result.chunks_pruned == 0


# ─── Full pipeline integration ────────────────────────────────────────────────


class TestEngineFullPipeline:
    @pytest.mark.asyncio
    async def test_full_encode_retrieve_cycle(self, engine):
        """
        Integration test: process interactions, retrieve, verify.

        This test exercises the complete pipeline:
        1. Process several user interactions
        2. Retrieve memories relevant to a query
        3. Verify the retrieved memories are coherent
        """
        # Simulate a conversation session
        session = [
            ("user", "I'm a senior Python developer"),
            ("assistant", "Great! What kind of projects do you work on?"),
            ("user", "I build machine learning pipelines with TensorFlow and PyTorch"),
            ("assistant", "Very interesting! Do you use any cloud providers?"),
            ("user", "Yes, I deploy primarily on AWS, using SageMaker for ML training"),
            ("user", "I also use Docker and Kubernetes for containerization"),
            ("user", "My team of 8 engineers uses GitHub and follows trunk-based development"),
        ]

        stored_count = 0
        for role, content in session:
            result = await engine.process(Interaction(role=role, content=content))
            if result.stored:
                stored_count += 1

        assert stored_count > 0

        # Retrieve memories relevant to ML
        ml_memories = await engine.retrieve("machine learning frameworks", top_k=3)
        assert len(ml_memories) >= 1

        # Retrieve memories relevant to cloud deployment
        cloud_memories = await engine.retrieve("AWS cloud deployment", top_k=3)
        assert len(cloud_memories) >= 1

    @pytest.mark.asyncio
    async def test_process_batch(self, engine):
        """process_batch() should process multiple interactions sequentially."""
        interactions = [
            Interaction(role="user", content="fact one: Python is great"),
            Interaction(role="user", content="fact two: I use PostgreSQL"),
            Interaction(role="user", content="fact three: AWS is my cloud provider"),
        ]

        results = await engine.process_batch(interactions)
        assert len(results) == 3
        for result in results:
            assert hasattr(result, "stored")

    @pytest.mark.asyncio
    async def test_stats_returns_all_module_stats(self, engine):
        """get_stats() should include stats from all 5 modules."""
        await engine.process(make_interaction("test for stats"))
        stats = engine.get_stats()

        assert "surprisal_gate" in stats
        assert "mutable_rag" in stats
        assert "affective_router" in stats
        assert "sleep_daemon" in stats
        assert "spreading_activation" in stats
        assert "store" in stats
        assert "engine" in stats

    @pytest.mark.asyncio
    async def test_engine_default_construction(self):
        """MnemosEngine() with no args should work (uses defaults)."""
        engine = MnemosEngine()
        result = await engine.process(
            Interaction(role="user", content="testing default construction")
        )
        assert hasattr(result, "stored")

    @pytest.mark.asyncio
    async def test_consolidation_in_pipeline(self, engine):
        """Full cycle: process → consolidate → retrieve should work."""
        # Process some interactions
        facts = [
            "My favorite language is Python",
            "I work at a fintech startup",
            "I prefer async programming patterns",
        ]
        for fact in facts:
            await engine.process(make_interaction(fact))

        # Consolidate
        consolidation = await engine.consolidate()
        assert consolidation.chunks_pruned == len(facts)

        # After consolidation, semantic facts should be retrievable
        results = await engine.retrieve("programming language preference", top_k=3)
        # Store should have consolidated facts
        assert len(engine.store.get_all()) > 0

    @pytest.mark.asyncio
    async def test_spreading_activation_builds_graph(self, engine):
        """Processing related concepts should build a connected graph."""
        related_facts = [
            "Python is a programming language",
            "Python has libraries like NumPy",
            "Machine learning uses Python",
        ]
        for fact in related_facts:
            await engine.process(make_interaction(fact))

        # The graph should have nodes
        node_count = engine.spreading_activation.get_node_count()
        stored_count = len(engine.store.get_all())
        # Graph nodes should match stored chunks
        assert node_count == stored_count

    @pytest.mark.asyncio
    async def test_engine_properties(self, engine):
        """Engine should expose store, llm, embedder as properties."""
        assert engine.store is not None
        assert engine.llm is not None
        assert engine.embedder is not None


class TestEngineRetrievePersistence:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
    async def test_retrieve_persists_access_count_and_touch(
        self, store_kind: str, tmp_path
    ) -> None:
        """retrieve() should persist access_count and updated_at for returned chunks."""
        if store_kind == "memory":
            store = InMemoryStore()
        else:
            store = SQLiteStore(db_path=str(tmp_path / "mnemos_access_count.db"))

        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            ),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        try:
            result = await engine.process(
                make_interaction("Use uv for Python tooling"),
                scope="project",
                scope_id="repo-alpha",
            )
            assert result.chunk is not None

            before = store.get(result.chunk.id)
            assert before is not None
            before_updated_at = before.updated_at
            assert before.access_count == 0

            results = await engine.retrieve(
                "python tooling",
                top_k=1,
                reconsolidate=False,
                current_scope="project",
                scope_id="repo-alpha",
                allowed_scopes=("project",),
            )

            assert len(results) == 1
            after = store.get(result.chunk.id)
            assert after is not None
            assert after.access_count == 1
            assert after.updated_at >= before_updated_at
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

    @pytest.mark.asyncio
    async def test_retrieve_does_not_call_store_stats_on_hot_path(self) -> None:
        """retrieve() should not compute full store stats just to log backend metadata."""
        store = InMemoryStore()
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            ),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        result = await engine.process(
            make_interaction("Use uv for Python tooling"),
            scope="project",
            scope_id="repo-alpha",
        )
        assert result.chunk is not None

        original_get_stats = store.get_stats

        def fail_get_stats() -> dict[str, object]:
            raise AssertionError("retrieve() hot path should not call store.get_stats()")

        store.get_stats = fail_get_stats  # type: ignore[assignment]
        try:
            results = await engine.retrieve(
                "python tooling",
                top_k=1,
                reconsolidate=False,
                current_scope="project",
                scope_id="repo-alpha",
                allowed_scopes=("project",),
            )
        finally:
            store.get_stats = original_get_stats  # type: ignore[assignment]

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_uses_backend_managed_touch_increment(self) -> None:
        """retrieve() should let the store apply its own increment semantics."""

        class RecordingStore(InMemoryStore):
            def __init__(self) -> None:
                super().__init__()
                self.touch_access_count_args: list[int | None] = []

            def touch(
                self,
                chunk_id: str,
                *,
                access_count: int | None = None,
                updated_at: datetime | None = None,
            ) -> bool:
                self.touch_access_count_args.append(access_count)
                return super().touch(
                    chunk_id,
                    access_count=access_count,
                    updated_at=updated_at,
                )

        store = RecordingStore()
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            ),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        await engine.process(
            make_interaction("Use uv for Python tooling"),
            scope="project",
            scope_id="repo-alpha",
        )

        results = await engine.retrieve(
            "python tooling",
            top_k=1,
            reconsolidate=False,
            current_scope="project",
            scope_id="repo-alpha",
            allowed_scopes=("project",),
        )

        assert len(results) == 1
        assert store.touch_access_count_args == [None]

    @pytest.mark.asyncio
    async def test_retrieve_top_one_skips_spreading_activation(self) -> None:
        """top_k=1 should stay on the direct semantic path without graph expansion."""
        store = InMemoryStore()
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            ),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        await engine.process(
            make_interaction("Use uv for Python tooling"),
            scope="project",
            scope_id="repo-alpha",
        )

        original_retrieve = engine.spreading_activation.retrieve

        def fail_retrieve(*args, **kwargs):
            raise AssertionError("top_k=1 retrieval should not invoke spreading activation")

        engine.spreading_activation.retrieve = fail_retrieve  # type: ignore[assignment]
        try:
            results = await engine.retrieve(
                "python tooling",
                top_k=1,
                reconsolidate=False,
                current_scope="project",
                scope_id="repo-alpha",
                allowed_scopes=("project",),
            )
        finally:
            engine.spreading_activation.retrieve = original_retrieve  # type: ignore[assignment]

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_top_one_skips_affective_reranking(self) -> None:
        """top_k=1 should not spend time on affective classification or reranking."""
        store = InMemoryStore()
        engine = MnemosEngine(
            config=MnemosConfig(
                surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            ),
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store,
        )

        await engine.process(
            make_interaction("Use uv for Python tooling"),
            scope="project",
            scope_id="repo-alpha",
        )

        original_classify = engine.affective_router.classify_state
        original_retrieve = engine.affective_router.retrieve

        async def fail_classify(*args, **kwargs):
            raise AssertionError("top_k=1 retrieval should not classify affective state")

        async def fail_retrieve(*args, **kwargs):
            raise AssertionError("top_k=1 retrieval should not run affective reranking")

        engine.affective_router.classify_state = fail_classify  # type: ignore[assignment]
        engine.affective_router.retrieve = fail_retrieve  # type: ignore[assignment]
        try:
            results = await engine.retrieve(
                "python tooling",
                top_k=1,
                reconsolidate=False,
                current_scope="project",
                scope_id="repo-alpha",
                allowed_scopes=("project",),
            )
        finally:
            engine.affective_router.classify_state = original_classify  # type: ignore[assignment]
            engine.affective_router.retrieve = original_retrieve  # type: ignore[assignment]

        assert len(results) == 1


class TestEnginePersistence:
    @pytest.mark.asyncio
    async def test_spreading_graph_hydrates_from_persistent_store(self, tmp_path):
        """A restarted engine should rebuild spreading graph nodes from persisted chunks."""
        db_path = tmp_path / "mnemos_persist.db"
        config = MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
        )

        store1 = SQLiteStore(db_path=str(db_path))
        engine1 = MnemosEngine(
            config=config,
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store1,
        )

        await engine1.process(make_interaction("I deploy services on AWS."))
        await engine1.process(make_interaction("I use Docker for packaging."))
        stored_count = len(engine1.store.get_all())
        store1.close()

        store2 = SQLiteStore(db_path=str(db_path))
        engine2 = MnemosEngine(
            config=config,
            llm=MockLLMProvider(),
            embedder=SimpleEmbeddingProvider(dim=64),
            store=store2,
        )

        assert len(engine2.store.get_all()) == stored_count
        assert engine2.spreading_activation.get_node_count() == stored_count
        store2.close()


# ─── Tests for __init__.py exports ────────────────────────────────────────────


class TestPackageExports:
    def test_version_is_exported(self):
        """__version__ should be importable from mnemos."""
        import mnemos

        assert hasattr(mnemos, "__version__")
        assert mnemos.__version__ == "0.1.0"

    def test_all_main_classes_exported(self):
        """All main classes should be importable from mnemos directly."""
        from mnemos import (
            MnemosEngine,
            MnemosConfig,
            SurprisalGate,
            MutableRAG,
            AffectiveRouter,
            SleepDaemon,
            SpreadingActivation,
            MemoryChunk,
            CognitiveState,
            Interaction,
            ProcessResult,
            ActivationNode,
            ConsolidationResult,
            MockLLMProvider,
            SimpleEmbeddingProvider,
            InMemoryStore,
        )

        # Verify they are all classes/functions (not None)
        assert MnemosEngine is not None
        assert MemoryChunk is not None
        assert CognitiveState is not None

    def test_config_classes_exported(self):
        """All config classes should be importable from mnemos."""
        from mnemos import (
            MnemosConfig,
            SurprisalConfig,
            MutableRAGConfig,
            AffectiveConfig,
            SleepConfig,
            SpreadingConfig,
        )

        config = MnemosConfig()
        assert config.surprisal is not None
        assert config.sleep is not None

    def test_utility_functions_exported(self):
        """Utility functions should be importable from mnemos."""
        from mnemos import cosine_similarity, cosine_distance

        sim = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(1.0)
        dist = cosine_distance([1.0, 0.0], [1.0, 0.0])
        assert dist == pytest.approx(0.0)
