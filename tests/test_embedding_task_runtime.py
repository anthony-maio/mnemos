"""Task-aware embedding configuration, persistence, and local-memory integration."""

from pathlib import Path

import pytest

from mnemos.config import MnemosConfig, ModuleConfig, SurprisalConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.modules.sleep import SleepDaemon
from mnemos.modules.surprisal import SurprisalGate
from mnemos.reindex import activate_generation, build_generation
from mnemos.runtime import build_embedder_from_settings, embedding_identity
from mnemos.scopes import ScopeRegistry
from mnemos.settings import AppSettings, load_settings, save_settings
from mnemos.types import Interaction, MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider, PrefixEmbeddingProvider, embed_text_async
from mnemos.utils.llm import MockLLMProvider, UnavailableLLMProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore


class RecordingEmbedding(EmbeddingProvider):
    def __init__(self):
        self.inputs = []

    @property
    def dim(self):
        return 3

    @property
    def should_offload_in_async(self):
        return False

    def embed(self, text):
        self.inputs.append(text)
        return [0.0, 1.0, 0.0] if text.startswith(("D: ", "Q: ")) else [1.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


@pytest.mark.asyncio
async def test_surprisal_compares_both_texts_in_the_symmetric_space():
    recording = RecordingEmbedding()
    embedder = PrefixEmbeddingProvider(
        recording, query_prefix="Q: ", document_prefix="D: ", symmetric_prefix="S: "
    )
    gate = SurprisalGate(
        MockLLMProvider(responses={"predict what": "prediction"}),
        embedder,
        InMemoryStore(),
        SurprisalConfig(min_content_length=0),
    )
    score, status, _, _ = await gate._compute_surprisal(
        Interaction(role="user", content="observed")
    )
    assert status == "measured"
    assert score == 0.0
    assert recording.inputs == ["S: prediction", "S: observed"]


@pytest.mark.asyncio
async def test_sleep_compares_candidate_and_evidence_in_the_symmetric_space():
    recording = RecordingEmbedding()
    embedder = PrefixEmbeddingProvider(
        recording, query_prefix="Q: ", document_prefix="D: ", symmetric_prefix="S: "
    )
    support, similarity = await SleepDaemon(InMemoryStore())._measure_recall_support(
        "candidate", [Interaction(role="user", content="evidence")], embedder
    )
    assert support == 1
    assert similarity == 1.0
    assert recording.inputs == ["S: candidate", "S: evidence"]


def settings(**prefixes):
    return AppSettings.model_validate({"embedding": {"provider": "simple", "dim": 3, **prefixes}})


def test_empty_prefixes_preserve_the_existing_persistent_identity(tmp_path):
    legacy_identity = {
        "provider": "simple",
        "model": None,
        "revision": None,
        "dimension": 3,
        "normalization": "provider_default",
    }
    path = tmp_path / "legacy.sqlite"
    store = SQLiteStore(str(path))
    store.bind_embedding_identity(legacy_identity)
    store.store(MemoryChunk(content="Alpha", embedding=[1.0, 0.0, 0.0]))
    store.close()
    reopened = SQLiteStore(str(path))
    try:
        reopened.bind_embedding_identity(embedding_identity(settings()))
        assert len(reopened.get_all()) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("field", ["query_prefix", "document_prefix", "symmetric_prefix"])
def test_each_task_prefix_change_requires_a_new_index_identity(tmp_path, field):
    path = tmp_path / "identity.sqlite"
    store = SQLiteStore(str(path))
    store.bind_embedding_identity(embedding_identity(settings()))
    fact = MemoryChunk(content="Alpha", embedding=[1.0, 0.0, 0.0])
    store.store(fact)
    store.close()
    reopened = SQLiteStore(str(path))
    try:
        with pytest.raises(ValueError, match="identity changed"):
            reopened.bind_embedding_identity(embedding_identity(settings(**{field: "prefix: "})))
        assert reopened.get(fact.id).embedding == [1.0, 0.0, 0.0]
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_saved_prefix_settings_reach_the_actual_ollama_payload(tmp_path, monkeypatch):
    configured = AppSettings.model_validate(
        {
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest",
                "query_prefix": "search_query: ",
                "document_prefix": "search_document: ",
                "symmetric_prefix": "clustering: ",
            }
        }
    )
    path = tmp_path / "mnemos.toml"
    save_settings(configured, path, scope="global")
    loaded = load_settings(env={}, cwd=tmp_path, global_config_path=path).settings
    payloads = []

    def post(url, *, payload, timeout):
        payloads.append(payload)
        return {"embeddings": [[1.0, 0.0, 0.0]]}

    monkeypatch.setattr("mnemos.utils.embeddings._http_post_json", post)
    embedder = build_embedder_from_settings(loaded)
    for role in ("query", "document", "symmetric"):
        await embed_text_async(embedder, "same text", role=role)
    assert [payload["input"] for payload in payloads] == [
        "search_query: same text",
        "search_document: same text",
        "clustering: same text",
    ]
    assert embedding_identity(loaded)["task_contract"]["query_prefix"] == "search_query: "


@pytest.mark.asyncio
async def test_reindex_uses_document_role_before_atomic_identity_activation(tmp_path):
    store = SQLiteStore(str(tmp_path / "reindex.sqlite"))
    old_identity = embedding_identity(settings())
    new_identity = embedding_identity(settings(query_prefix="Q: ", document_prefix="D: "))
    store.bind_embedding_identity(old_identity)
    fact = MemoryChunk(content="Alpha", embedding=[1.0, 0.0, 0.0])
    store.store(fact)
    recording = RecordingEmbedding()
    provider = PrefixEmbeddingProvider(recording, query_prefix="Q: ", document_prefix="D: ")
    try:
        generation = await build_generation(store, provider, new_identity)
        assert generation["ready"] is True
        assert recording.inputs == ["D: Alpha"]
        assert store.get(fact.id).embedding == [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="identity changed"):
            store.bind_embedding_identity(new_identity)
        activate_generation(store, generation["generation_id"])
        store.bind_embedding_identity(new_identity)
        assert store.get(fact.id).embedding == [0.0, 1.0, 0.0]
        store.close()
        store = SQLiteStore(str(tmp_path / "reindex.sqlite"))
        store.bind_embedding_identity(new_identity)
        assert store.get(fact.id).embedding == [0.0, 1.0, 0.0]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_service_remember_and_restarted_recall_use_distinct_roles(tmp_path: Path):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="codex")
    recording = RecordingEmbedding()
    embedder = PrefixEmbeddingProvider(recording, query_prefix="Q: ", document_prefix="D: ")
    config = MnemosConfig(
        modules=ModuleConfig(
            surprisal=False,
            sleep=False,
            affective=False,
            spreading=False,
        )
    )
    path = tmp_path / "service.sqlite"

    def open_memory():
        return LocalMemoryService(
            MnemosEngine(
                config=config,
                llm=UnavailableLLMProvider(),
                embedder=embedder,
                store=SQLiteStore(str(path)),
            ),
            registry,
        )

    memory = open_memory()
    try:
        await memory.dispatch(
            "remember",
            {
                "context": context.model_dump(),
                "content": "Alpha uses SQLite.",
            },
        )
        await memory.maintenance.run_once(force=True)
        facts = memory.engine.store.get_all()
        assert len(facts) == 1
        assert facts[0].embedding == [0.0, 1.0, 0.0]
        assert "D: Alpha uses SQLite." in recording.inputs
        memory.engine.store.close()
        memory = open_memory()
        recording.inputs.clear()
        result = await memory.dispatch(
            "retrieve",
            {
                "context": context.model_dump(),
                "query": "Alpha storage",
                "profile": "baseline",
            },
        )
        assert result["facts"][0]["content"] == "Alpha uses SQLite."
        assert recording.inputs == ["Q: Alpha storage"]
    finally:
        memory.engine.store.close()
