import pytest

from mnemos.config import MnemosConfig, SpreadingConfig
from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import SQLiteStore


@pytest.mark.asyncio
async def test_persisted_neighbor_beyond_startup_and_semantic_pool_is_loaded(tmp_path):
    path = str(tmp_path / "graph.sqlite")
    config = MnemosConfig(spreading=SpreadingConfig(startup_hydration_limit=1))
    config.retrieval.max_candidates = 2
    config.modules.affective = False

    class QueryEmbedding(SimpleEmbeddingProvider):
        def embed(self, text):
            return [1.0, 0.0]

    store = SQLiteStore(path)
    engine = MnemosEngine(config=config, embedder=QueryEmbedding(dim=2), store=store)
    for identifier, vector in (("a", [1.0, 0.0]), ("b", [0.99, 0.01]), ("z", [0.6, 0.8])):
        store.store(
            MemoryChunk(
                id=identifier,
                content="unrelated " + identifier,
                embedding=vector,
                metadata={"scope": "project", "scope_id": "project"},
            )
        )
    store.replace_graph_neighbors("a", {"z": 0.9})
    store.set_graph_rebuild_cursor(None)
    store.close()
    reopened = SQLiteStore(path)
    try:
        engine = MnemosEngine(config=config, embedder=QueryEmbedding(dim=2), store=reopened)
        assert engine.spreading_activation.get_node("z") is None
        await engine.retrieve(
            "query",
            top_k=1,
            current_scope="project",
            scope_id="project",
            allowed_scopes=["project"],
            profile="biomimetic",
        )
        assert engine.spreading_activation.get_node("z") is not None
        assert engine.spreading_activation.get_node("a").neighbors["z"] == 0.9
    finally:
        reopened.close()


def test_incompatible_graph_is_rebuilt_in_resumable_batches_without_changing_facts(tmp_path):
    path = str(tmp_path / "graph.sqlite")
    embedder = SimpleEmbeddingProvider(dim=8)
    store = SQLiteStore(path)
    engine = MnemosEngine(embedder=embedder, store=store)
    for identifier in ("a", "b", "c"):
        store.store(
            MemoryChunk(
                id=identifier,
                content="Use SQLite",
                embedding=embedder.embed("Use SQLite"),
                metadata={"scope": "project", "scope_id": "project"},
            )
        )
    store.replace_graph_neighbors("a", {"b": 0.01})
    store._conn.execute("UPDATE mnemos_meta SET value='obsolete' WHERE key='graph_contract'")
    store._conn.commit()
    store.close()
    store = SQLiteStore(path)
    engine = MnemosEngine(embedder=embedder, store=store)
    assert store.get_graph_edges() == {}
    assert engine.rebuild_graph_batch(limit=1) == 1
    assert store.graph_rebuild_cursor() == "a"
    store.close()
    store = SQLiteStore(path)
    try:
        engine = MnemosEngine(embedder=embedder, store=store)
        assert store.graph_rebuild_cursor() == "a"
        assert engine.rebuild_graph_batch(limit=1) == 1
        assert store.graph_rebuild_cursor() == "b"
        while engine.rebuild_graph_batch(limit=1):
            pass
        assert store.graph_rebuild_cursor() is None
        assert store.get_graph_edges()["a"]["b"] > 0.9
        assert all(
            chunk.version == 1 and chunk.content == "Use SQLite" for chunk in store.get_all()
        )
    finally:
        store.close()
