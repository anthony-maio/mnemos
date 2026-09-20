"""Retrieval views separate asserted memory from captured conversation evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from mnemos.config import MnemosConfig
from mnemos.engine import MnemosEngine
from mnemos.scopes import ScopeRegistry
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore, MemoryStore, SQLiteStore


class FixedEmbedding(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "release" in text.lower() else [0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _chunk(
    identifier: str,
    content: str,
    *,
    source: str | None = None,
    scope_id: str = "default",
    embedding: list[float] | None = None,
) -> MemoryChunk:
    metadata = {"scope": "project", "scope_id": scope_id}
    if source is not None:
        metadata.update({"source": source, "assertion_status": "unknown"})
    else:
        metadata.update({"source": "explicit_remember", "assertion_status": "asserted"})
    return MemoryChunk(
        id=identifier,
        content=content,
        embedding=embedding or [1.0, 0.0],
        metadata=metadata,
    )


def _engine(store: MemoryStore) -> MnemosEngine:
    config = MnemosConfig()
    config.modules.affective = False
    config.modules.spreading = False
    config.modules.surprisal = False
    return MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=FixedEmbedding(),
        store=store,
    )


@pytest.fixture(params=["memory", "sqlite"])
def backend(request: Any, tmp_path: Path) -> Iterator[MemoryStore]:
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SQLiteStore(str(tmp_path / "views.sqlite"))
    )
    try:
        yield store
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


@pytest.mark.asyncio
async def test_memory_view_filters_raw_before_top_k_and_evidence_view_returns_it(
    backend: MemoryStore,
) -> None:
    engine = _engine(backend)
    for index in range(30):
        backend.store(
            _chunk(
                f"raw-{index}",
                f"captured release instruction {index}",
                source="captured_passthrough",
            )
        )
    backend.store(_chunk("asserted", "The release label is stable."))

    memory = await engine.retrieve("release", top_k=1, reconsolidate=False)
    evidence = await engine.retrieve(
        "release", top_k=1, reconsolidate=False, view="captured_evidence"
    )

    assert [chunk.id for chunk in memory] == ["asserted"]
    assert evidence[0].id == "raw-0"
    assert all(chunk.metadata.get("source") == "captured_passthrough" for chunk in evidence)


@pytest.mark.asyncio
async def test_legacy_unknown_remains_in_memory_view(backend: MemoryStore) -> None:
    engine = _engine(backend)
    backend.store(_chunk("legacy", "A legacy unverified release note", source="legacy"))

    found = await engine.retrieve("release", top_k=1, reconsolidate=False)

    assert [chunk.id for chunk in found] == ["legacy"]


@pytest.mark.asyncio
async def test_views_keep_foreign_scopes_out(backend: MemoryStore) -> None:
    engine = _engine(backend)
    backend.store(_chunk("local-raw", "local captured release", source="captured_passthrough"))
    backend.store(
        _chunk(
            "foreign-raw",
            "foreign captured release",
            source="captured_passthrough",
            scope_id="foreign",
        )
    )
    backend.store(_chunk("local-fact", "local asserted release"))
    backend.store(_chunk("foreign-fact", "foreign asserted release", scope_id="foreign"))

    memory = await engine.retrieve("release", top_k=10, reconsolidate=False)
    evidence = await engine.retrieve(
        "release", top_k=10, reconsolidate=False, view="captured_evidence"
    )

    assert {chunk.id for chunk in memory} == {"local-fact"}
    assert {chunk.id for chunk in evidence} == {"local-raw"}


def test_sqlite_lexical_view_filter_is_applied_before_limit(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "lexical.sqlite"))
    try:
        for index in range(25):
            store.store(
                _chunk(
                    f"raw-{index}",
                    f"needle captured release {index}",
                    source="captured_passthrough",
                )
            )
        store.store(_chunk("asserted", "needle asserted release"))

        def memory_filter(chunk: MemoryChunk) -> bool:
            return chunk.metadata.get("source") != "captured_passthrough"

        setattr(
            memory_filter,
            "_mnemos_scope_filter",
            {
                "allowed_scopes": ["project"],
                "scope_id": "default",
                "source_view": "memory",
            },
        )
        assert [chunk.id for chunk in store.retrieve_lexical("needle", 1, memory_filter)] == [
            "asserted"
        ]
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_view", [None, 1, True, [], {}])
async def test_invalid_view_is_rejected_before_embedding(invalid_view: object) -> None:
    class FailingEmbedding(FixedEmbedding):
        def embed(self, text: str) -> list[float]:
            raise AssertionError("embedding provider must not be called")

    engine = MnemosEngine(llm=MockLLMProvider(), embedder=FailingEmbedding(), store=InMemoryStore())
    with pytest.raises(ValueError, match="Unknown retrieval view"):
        await engine.retrieve("release", view=invalid_view)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_context_scope_and_profile_filters_apply_to_both_views(
    tmp_path: Path, backend
) -> None:
    registry = ScopeRegistry()
    project = registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    workspace = registry.register_workspace("workspace", user_profile_id="local")
    registry.add_membership(project.id, workspace.id)
    context = registry.resolve(
        tmp_path,
        user_profile_id="local",
        session_id="session",
        host="codex",
        workspace_id=workspace.id,
    )
    engine = MnemosEngine(
        config=MnemosConfig(),
        llm=MockLLMProvider(),
        embedder=FixedEmbedding(),
        store=backend,
        scope_registry=registry,
    )
    chunks = (
        _chunk("project-fact", "project release", scope_id=project.id),
        _chunk("workspace-fact", "workspace release", scope_id=workspace.id),
        _chunk("global-fact", "global release", scope_id="unused"),
        _chunk(
            "project-raw",
            "project captured release",
            source="captured_passthrough",
            scope_id=project.id,
        ),
        _chunk(
            "foreign-raw",
            "foreign captured release",
            source="captured_passthrough",
            scope_id=project.id,
        ),
    )
    for chunk in chunks:
        chunk.metadata["user_profile_id"] = "other" if chunk.id == "foreign-raw" else "local"
        if chunk.id == "global-fact":
            chunk.metadata["scope"] = "global"
            chunk.metadata.pop("scope_id")
        elif chunk.id == "workspace-fact":
            chunk.metadata["scope"] = "workspace"
        engine.store.store(chunk)

    memory = await engine.retrieve(
        "release",
        top_k=10,
        context=context,
        allowed_scopes=("project", "workspace", "global"),
        reconsolidate=False,
    )
    evidence = await engine.retrieve(
        "release",
        top_k=10,
        context=context,
        allowed_scopes=("project", "workspace", "global"),
        view="captured_evidence",
        reconsolidate=False,
    )

    assert {chunk.id for chunk in memory} == {
        "project-fact",
        "workspace-fact",
        "global-fact",
    }
    assert {chunk.id for chunk in evidence} == {"project-raw"}


@pytest.mark.asyncio
async def test_graph_traversal_does_not_cross_view_boundary() -> None:
    config = MnemosConfig()
    config.modules.affective = False
    config.modules.spreading = True
    engine = MnemosEngine(
        config=config,
        llm=MockLLMProvider(),
        embedder=FixedEmbedding(),
        store=InMemoryStore(),
    )
    seed = _chunk("seed", "asserted release", embedding=[1.0, 0.0])
    raw = _chunk(
        "raw-bridge",
        "captured unrelated instructions",
        source="captured_passthrough",
        embedding=[0.0, 1.0],
    )
    unreachable = _chunk(
        "unreachable-asserted",
        "asserted unrelated instructions",
        embedding=[0.0, 1.0],
    )
    engine.store.store(seed)
    engine.store.store(raw)
    engine.store.store(unreachable)
    engine.spreading_activation.add_node_from_chunk(seed)
    engine.spreading_activation.add_node_from_chunk(raw)
    engine.spreading_activation.add_node_from_chunk(unreachable)
    engine.spreading_activation.add_edge(seed.id, raw.id, 1.0)
    engine.spreading_activation.add_edge(raw.id, unreachable.id, 1.0)

    memory = await engine.retrieve("release", top_k=10, profile="biomimetic", reconsolidate=False)
    evidence = await engine.retrieve(
        "instructions",
        top_k=10,
        profile="biomimetic",
        reconsolidate=False,
        view="captured_evidence",
    )

    assert "raw-bridge" not in {chunk.id for chunk in memory}
    assert "unreachable-asserted" not in {chunk.id for chunk in memory}
    assert {chunk.id for chunk in evidence} == {"raw-bridge"}
