from datetime import datetime, timezone

import pytest

from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.types import CognitiveState, MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore


class FixedEmbedding(EmbeddingProvider):
    @property
    def dim(self):
        return 2

    def embed(self, text):
        return [1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def make_engine():
    llm = MockLLMProvider(
        responses={"valence": "-1, 1, 1", "has this stored memory": "CHANGED: Now use Rust"},
        record_calls=True,
    )
    engine = MnemosEngine(llm=llm, embedder=FixedEmbedding(), store=InMemoryStore())
    for identifier, embedding, state in [
        ("semantic", [1.0, 0.0], CognitiveState(valence=1, arousal=0, complexity=0)),
        ("state", [0.8, 0.6], CognitiveState(valence=-1, arousal=1, complexity=1)),
    ]:
        engine.store.store(
            MemoryChunk(
                id=identifier,
                content="Python memory " + identifier,
                embedding=embedding,
                cognitive_state=state,
                metadata={"scope": "project", "scope_id": "default"},
            )
        )
    return engine


@pytest.mark.asyncio
async def test_biomimetic_top_one_uses_actual_state_score():
    engine = make_engine()
    engine.config.retrieval.spreading_enabled = False
    baseline = await engine.retrieve("Python", top_k=1, profile="baseline")
    full = await engine.retrieve("Python", top_k=1, profile="biomimetic")
    assert baseline[0].id == "semantic"
    assert full[0].id == "state"
    trace = full[0].metadata["retrieval_trace"]
    assert trace["semantic"] == pytest.approx(0.8)
    assert trace["state_match"] == 1
    assert trace["total"] == pytest.approx(0.86)
    assert "retrieval_trace" not in engine.store.get("state").metadata


@pytest.mark.asyncio
async def test_question_cannot_become_reconsolidation_evidence():
    engine = make_engine()
    before = engine.store.get("semantic").model_copy(deep=True)
    await engine.retrieve("Did we switch to Rust?", top_k=1)
    await engine.mutable_rag.process_labile_chunks()
    after = engine.store.get("semantic")
    assert after.content == before.content
    assert after.version == before.version
    assert after.updated_at == before.updated_at
    assert after.access_count == before.access_count + 1
    assert not engine.store.list_reconsolidation_jobs()


@pytest.mark.asyncio
async def test_relevance_floor_and_context_budget_are_enforced():
    engine = make_engine()
    engine.store.clear()
    engine.store.store(
        MemoryChunk(
            id="irrelevant",
            content="Unrelated",
            embedding=[0, 1],
            metadata={"scope": "project", "scope_id": "default"},
        )
    )
    assert not await engine.retrieve("Python", profile="baseline")
    engine.store.store(
        MemoryChunk(
            id="long",
            content="X" * 5000,
            embedding=[1, 0],
            metadata={"scope": "project", "scope_id": "default"},
        )
    )
    assert not await engine.retrieve("Python", max_context_tokens=100)
    assert not await engine.retrieve("Python", top_k=0)


@pytest.mark.asyncio
async def test_unresolved_conflict_label_is_included_in_context_budget():
    engine = make_engine()
    engine.store.clear()
    chunk = MemoryChunk(
        id="disputed",
        content="Use Python",
        embedding=[1.0, 0.0],
        metadata={"scope": "project", "scope_id": "default", "assertion_status": "conflict"},
    )
    engine.store.store(chunk)
    unlabelled_bytes = len(f"[{chunk.id} version {chunk.version}] {chunk.content}\n".encode())
    assert not await engine.retrieve("Python", max_context_tokens=unlabelled_bytes)
    assert await engine.retrieve("Python", max_context_tokens=unlabelled_bytes + 100)


@pytest.mark.asyncio
async def test_missing_state_falls_back_to_semantic_scores():
    from mnemos.utils.llm import UnavailableLLMProvider

    engine = make_engine()
    engine.affective_router._llm = UnavailableLLMProvider()
    engine.config.retrieval.spreading_enabled = False
    chunks = await engine.retrieve("Python", top_k=1, profile="biomimetic")
    trace = chunks[0].metadata["retrieval_trace"]
    assert chunks[0].id == "semantic"
    assert trace["total"] == 1
    assert trace["state_match"] is None
    assert trace["state_status"] == "unavailable"
