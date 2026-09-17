from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import mnemos.cli as cli_module
from mnemos.config import MnemosConfig, SleepConfig, SurprisalConfig
from mnemos.engine import MnemosEngine
from mnemos.types import Interaction, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import LLMProvider, MockLLMProvider
from mnemos.utils.storage import SQLiteStore


class FailingLLM(LLMProvider):
    async def predict(self, prompt: str, **kwargs: Any) -> str:
        _ = prompt, kwargs
        raise RuntimeError("provider unavailable")

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        _ = prompt, labels
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_autostore_hook_persists_idempotent_episodic_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "hook-events.sqlite"
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")

    def _unexpected_engine() -> None:
        raise AssertionError("autostore-hook should not build a processing engine")

    monkeypatch.setattr(cli_module, "_build_engine", _unexpected_engine, raising=False)
    monkeypatch.setattr(cli_module, "_build_hook_engine", _unexpected_engine, raising=False)

    payload = json.dumps(
        {
            "prompt": "For this repository, use uv and mypy before opening a PR.",
            "cwd": "/tmp/repo-alpha",
        }
    )
    args = Namespace(
        event="UserPromptSubmit",
        payload=payload,
        scope="project",
        scope_id="",
        max_chars=1200,
        dry_run=False,
    )

    await cli_module._cmd_autostore_hook(args)
    await cli_module._cmd_autostore_hook(args)

    store = SQLiteStore(db_path=str(db_path))
    try:
        events = store.list_episodic_events()
    finally:
        store.close()

    assert len(events) == 1
    assert events[0].event_fingerprint
    assert events[0].scope == "project"
    assert events[0].scope_id == "repo-alpha"
    assert "uv and mypy" in events[0].content


@pytest.mark.asyncio
async def test_consolidate_promotes_persisted_episodic_events_across_restarts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "consolidate.sqlite"
    monkeypatch.setenv("MNEMOS_STORE_TYPE", "sqlite")
    monkeypatch.setenv("MNEMOS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("MNEMOS_LLM_PROVIDER", "mock")
    monkeypatch.setenv("MNEMOS_EMBEDDING_PROVIDER", "simple")

    payload = json.dumps(
        {
            "prompt": "Repo alpha uses uv and mypy for Python quality checks.",
            "cwd": "/tmp/repo-alpha",
        }
    )
    await cli_module._cmd_autostore_hook(
        Namespace(
            event="UserPromptSubmit",
            payload=payload,
            scope="project",
            scope_id="",
            max_chars=1200,
            dry_run=False,
        )
    )

    store = SQLiteStore(db_path=str(db_path))
    engine = MnemosEngine(
        config=MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
            sleep=SleepConfig(
                min_episodes_before_consolidation=1,
                consolidation_interval_seconds=0,
            ),
        ),
        llm=MockLLMProvider(
            responses={
                "extract permanent facts": "1. Repo alpha uses uv and mypy for Python quality checks."
            }
        ),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=store,
    )

    try:
        result = await engine.consolidate()
        chunks = store.get_all()
        events = store.list_episodic_events(status="consolidated")
    finally:
        store.close()

    assert result.facts_extracted == ["Repo alpha uses uv and mypy for Python quality checks."]
    assert len(chunks) == 1
    assert chunks[0].canonical_key is not None
    assert chunks[0].observation_count == 1
    assert len(events) == 1


@pytest.mark.asyncio
async def test_retrieve_does_not_queue_query_reconsolidation_after_restart(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "reconsolidation.sqlite"
    config = MnemosConfig(
        surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
        sleep=SleepConfig(
            min_episodes_before_consolidation=1,
            consolidation_interval_seconds=0,
        ),
    )

    first_store = SQLiteStore(db_path=str(db_path))
    first_engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=first_store,
    )
    try:
        result = await first_engine.process(Interaction(role="user", content="placeholder"))
    finally:
        first_store.close()

    assert result.stored is True
    assert result.chunk is not None

    second_store = SQLiteStore(db_path=str(db_path))
    second_engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=second_store,
    )
    try:
        retrieved = await second_engine.retrieve(
            "placeholder",
            top_k=1,
            reconsolidate=False,
            current_scope="project",
            scope_id="default",
            allowed_scopes=("project", "workspace", "global"),
        )
        jobs = second_store.list_reconsolidation_jobs(status="pending")
    finally:
        second_store.close()

    assert len(retrieved) == 1
    assert jobs == []

    third_store = SQLiteStore(db_path=str(db_path))
    third_engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=third_store,
    )
    try:
        await third_engine.consolidate()
        updated = third_store.get(result.chunk.id)
        remaining_jobs = third_store.list_reconsolidation_jobs(status="pending")
    finally:
        third_store.close()

    assert updated is not None
    assert updated.version == 1
    assert updated.last_retrieved_at is not None
    assert updated.reconsolidation_due_at is None
    assert updated.content == "placeholder"
    assert remaining_jobs == []


@pytest.mark.asyncio
async def test_manual_store_keeps_unknown_scores_null_when_provider_unavailable(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "truthful-scores.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    engine = MnemosEngine(
        config=MnemosConfig(
            surprisal=SurprisalConfig(threshold=0.0, min_content_length=0),
        ),
        llm=FailingLLM(),
        embedder=SimpleEmbeddingProvider(dim=64),
        store=store,
    )

    try:
        result = await engine.process(
            Interaction(role="user", content="Store this exact repo preference.")
        )
        chunk = store.get(result.chunk.id if result.chunk else "")
    finally:
        store.close()

    assert result.stored is True
    assert chunk is not None
    assert chunk.surprisal_score is None
    assert chunk.surprisal_status == "provider_unavailable"
    assert chunk.cognitive_state is None
    assert chunk.cognitive_state_status == "provider_unavailable"


def test_sqlite_store_persists_semantic_similar_edges_and_skips_eager_vec_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "vec-and-edges.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        alpha = MemoryChunk(
            id="alpha",
            content="alpha memory",
            embedding=[1.0, 0.0, 0.0],
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        beta = MemoryChunk(
            id="beta",
            content="beta memory",
            embedding=[0.9, 0.1, 0.0],
            metadata={"scope": "project", "scope_id": "repo-alpha"},
        )
        store.store(alpha)
        store.store(beta)
        store.replace_graph_neighbors(alpha.id, {beta.id: 0.91})
        edge_types = [
            row[0] for row in store._conn.execute("SELECT DISTINCT edge_type FROM memory_edges")
        ]
    finally:
        store.close()

    assert edge_types == ["SEMANTIC_SIMILAR"]

    def _unexpected_rebuild(self: SQLiteStore) -> None:
        raise AssertionError("SQLiteStore startup should not rebuild the vec index eagerly")

    monkeypatch.setattr(SQLiteStore, "_rebuild_vec_index", _unexpected_rebuild)
    reopened = SQLiteStore(db_path=str(db_path))
    reopened.close()


def test_repair_sqlite_db_collapses_exact_duplicates(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite"
    target_path = tmp_path / "target.sqlite"

    source = SQLiteStore(db_path=str(source_path))
    try:
        source.store(
            MemoryChunk(
                id="a1",
                content="Repo alpha uses uv",
                embedding=[1.0, 0.0, 0.0],
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )
        source.store(
            MemoryChunk(
                id="a2",
                content="Repo alpha uses uv",
                embedding=[1.0, 0.0, 0.0],
                metadata={"scope": "project", "scope_id": "repo-alpha"},
            )
        )
    finally:
        source.close()

    summary = cli_module._repair_sqlite_db(str(source_path), str(target_path), dry_run=False)

    repaired = SQLiteStore(db_path=str(target_path))
    try:
        chunks = repaired.get_all()
    finally:
        repaired.close()

    assert summary["source_chunks"] == 2
    assert summary["canonical_chunks"] == 1
    assert summary["duplicates_collapsed"] == 1
    assert len(chunks) == 1
    assert chunks[0].observation_count == 2
    assert chunks[0].canonical_key is not None
