from datetime import datetime, timezone

import pytest

from mnemos import engine as engine_module
from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore


class FixedEmbedding(EmbeddingProvider):
    @property
    def dim(self):
        return 2

    def embed(self, text):
        return [1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 8, tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("access_count", [0, 1000])
@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is known about Aster ownership?", "semantic"),
        ("Describe unknown ownership fields for Aster.", "semantic"),
        ("Describe concurrent ownership updates for Aster.", "semantic"),
        ("Who owns Aster now?", "newer"),
        ("Who is the CURRENT owner of Aster?", "newer"),
        ("Describe the latest Aster owner.", "newer"),
        ("Who owns Aster today?", "newer"),
        ("Describe recent Aster ownership.", "newer"),
        ("Who currently owns Aster?", "newer"),
        ("Who owned Aster recently?", "newer"),
    ],
)
async def test_temporal_words_not_substrings_control_reranking(
    backend, access_count, query, expected, tmp_path, monkeypatch
):
    monkeypatch.setattr(engine_module, "datetime", FixedClock)
    store = InMemoryStore() if backend == "memory" else SQLiteStore(str(tmp_path / "facts.sqlite"))
    try:
        for key, day, embedding in [("semantic", 1, [1.0, 0.0]), ("newer", 2, [0.8, 0.6])]:
            store.store(
                MemoryChunk(
                    id=key,
                    content=f"Aster owner record {key}.",
                    embedding=embedding,
                    metadata={"scope": "project", "scope_id": "fixed"},
                    created_at=datetime(2026, 11, day, tzinfo=timezone.utc),
                    updated_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
                    access_count=access_count,
                )
            )
        engine = MnemosEngine(
            config=MnemosConfig(), llm=MockLLMProvider(), embedder=FixedEmbedding(), store=store
        )
        selected = await engine.retrieve(
            query,
            top_k=1,
            reconsolidate=False,
            current_scope="project",
            scope_id="fixed",
            allowed_scopes=("project",),
        )
        assert [chunk.id for chunk in selected] == [expected]
        assert {chunk.id: chunk.content_recency_at.day for chunk in store.get_all()} == {
            "semantic": 1,
            "newer": 2,
        }
    finally:
        if isinstance(store, SQLiteStore):
            store.close()
