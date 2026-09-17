"""Deterministic checks for benchmark embedding task-role boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from benchmarks.local_memory.module_mechanism import (
    _FrozenEmbedder,
    prepare_frozen_embeddings,
)
from benchmarks.local_memory.module_mechanism_run import CheckpointedEmbeddingProvider
from benchmarks.local_memory.ollama_adapter import _TracingEmbedder
from benchmarks.local_memory.lifecycle_target_smoke import run_lifecycle_target_smoke
from mnemos.runtime import embedding_task_contract
from mnemos.utils.embeddings import EmbeddingProvider, PrefixEmbeddingProvider


class RoleProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def _vectors(self, texts: list[str], role: str) -> list[list[float]]:
        self.calls.append((role, tuple(texts)))
        value = {"query": 1.0, "document": 2.0, "symmetric": 3.0}[role]
        return [[value, float(index + 1)] for index, _ in enumerate(texts)]

    def embed(self, text: str) -> list[float]:
        return self._vectors([text], "symmetric")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._vectors(texts, "symmetric")

    def embed_query(self, text: str) -> list[float]:
        return self._vectors([text], "query")[0]

    def embed_document(self, text: str) -> list[float]:
        return self._vectors([text], "document")[0]

    def embed_batch_queries(self, texts: list[str]) -> list[list[float]]:
        return self._vectors(texts, "query")

    def embed_batch_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vectors(texts, "document")

    @property
    def dim(self) -> int:
        return 2

    @property
    def should_offload_in_async(self) -> bool:
        return False


class DriftProvider(RoleProvider):
    def embed_document(self, text: str) -> list[float]:
        self.calls.append(("document", (text,)))
        return [2.0, 1.0, 0.0]

    def embed_batch_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_document(text) for text in texts]


def test_frozen_preparation_keeps_identical_query_and_document_text_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "authored.json"
    destination = tmp_path / "frozen.json"
    source.write_text(
        json.dumps(
            {
                "graph_cases": [{"facts": [{"content": "same text"}], "query": "same text"}],
                "affective_pairs": [],
            }
        ),
        encoding="utf-8",
    )
    provider = RoleProvider()

    prepare_frozen_embeddings(source, destination, provider)

    frozen = json.loads(destination.read_text(encoding="utf-8"))
    assert frozen["graph_cases"][0]["facts"][0]["embedding"] == [2.0, 1.0]
    assert frozen["graph_cases"][0]["query"]["embedding"] == [1.0, 1.0]
    assert [role for role, _ in provider.calls] == ["document", "query"]

    embedder = _FrozenEmbedder(
        {"same text": [2.0, 1.0]},
        query_vectors={"same text": [1.0, 1.0]},
        document_vectors={"same text": [2.0, 1.0]},
    )
    assert embedder.embed_query("same text") == [1.0, 1.0]
    assert embedder.embed_document("same text") == [2.0, 1.0]


def test_roleful_checkpoint_reuses_query_and_document_entries(tmp_path: Path) -> None:
    contract = embedding_task_contract(
        query_prefix="query: ", document_prefix="document: ", symmetric_prefix=""
    )
    first = RoleProvider()
    path = tmp_path / "embeddings.json"
    cache = CheckpointedEmbeddingProvider(first, path, "binding", task_contract=contract)

    assert cache.embed_query("same text") == [1.0, 1.0]
    assert cache.embed_document("same text") == [2.0, 1.0]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["task_contract"] == contract
    assert saved["role_vectors"]["query"]["same text"] == [1.0, 1.0]
    assert saved["role_vectors"]["document"]["same text"] == [2.0, 1.0]

    second = RoleProvider()
    resumed = CheckpointedEmbeddingProvider(second, path, "binding", task_contract=contract)
    assert resumed.embed_query("same text") == [1.0, 1.0]
    assert resumed.embed_document("same text") == [2.0, 1.0]
    assert second.calls == []


def test_empty_contract_retains_legacy_flat_checkpoint(tmp_path: Path) -> None:
    provider = RoleProvider()
    path = tmp_path / "embeddings.json"
    cache = CheckpointedEmbeddingProvider(provider, path, "binding")

    assert cache.embed("same text") == [3.0, 1.0]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"binding": "binding", "vectors": {"same text": [3.0, 1.0]}}


def test_empty_prefixes_still_keep_explicit_query_and_document_roles(tmp_path: Path) -> None:
    provider = RoleProvider()
    cache = CheckpointedEmbeddingProvider(provider, tmp_path / "embeddings.json", "binding")

    assert cache.embed_query("same text") == [1.0, 1.0]
    assert cache.embed_document("same text") == [2.0, 1.0]
    assert [role for role, _ in provider.calls] == ["query", "document"]


def test_role_only_checkpoint_exposes_dimension_without_loading_provider(tmp_path: Path) -> None:
    class UnknownDimensionProvider(RoleProvider):
        @property
        def dim(self) -> int:
            raise RuntimeError("No provider request has run")

    path = tmp_path / "cached-query.json"
    initial = CheckpointedEmbeddingProvider(RoleProvider(), path, "binding")
    initial.embed_query("same text")
    provider = UnknownDimensionProvider()
    resumed = CheckpointedEmbeddingProvider(provider, path, "binding")
    assert resumed.dim == 2
    assert resumed.embed_query("same text") == [1.0, 1.0]
    assert provider.calls == []


def test_roleful_checkpoint_binding_also_validates_contract_and_dimension(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.json"
    first_contract = embedding_task_contract(
        query_prefix="query: ", document_prefix="document: ", symmetric_prefix=""
    )
    first = CheckpointedEmbeddingProvider(
        RoleProvider(), path, "binding", task_contract=first_contract
    )
    first.embed_query("query only")
    assert first.dim == 2

    changed_contract = embedding_task_contract(
        query_prefix="other query: ", document_prefix="document: ", symmetric_prefix=""
    )
    try:
        CheckpointedEmbeddingProvider(
            RoleProvider(), path, "binding", task_contract=changed_contract
        )
    except ValueError as exc:
        assert "task contract" in str(exc)
    else:
        raise AssertionError("changed role prefixes must reject an existing checkpoint")


def test_roleful_checkpoint_rejects_cross_role_dimension_drift(tmp_path: Path) -> None:
    contract = embedding_task_contract(
        query_prefix="query: ", document_prefix="document: ", symmetric_prefix=""
    )
    cache = CheckpointedEmbeddingProvider(
        DriftProvider(), tmp_path / "embeddings.json", "binding", task_contract=contract
    )
    cache.embed_query("same text")
    try:
        cache.embed_document("same text")
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("cross-role dimension drift must be rejected")


def test_tracing_embedder_forwards_each_role_without_double_prefix() -> None:
    trace: dict[str, Any] = {"embed_calls": [], "response_metadata": []}
    wrapped = PrefixEmbeddingProvider(
        RoleProvider(),
        query_prefix="Q:",
        document_prefix="D:",
        symmetric_prefix="S:",
    )
    embedder = _TracingEmbedder(wrapped, trace)

    assert embedder.embed_query("same text") == [3.0, 1.0]
    assert embedder.embed_document("same text") == [3.0, 1.0]
    assert embedder.embed_symmetric("same text") == [3.0, 1.0]
    calls = cast(list[dict[str, Any]], trace["embed_calls"])
    assert [item["role"] for item in calls] == [
        "query",
        "document",
        "symmetric",
    ]
    inner_calls = cast(RoleProvider, wrapped.inner).calls
    assert inner_calls == [
        ("symmetric", ("Q:same text",)),
        ("symmetric", ("D:same text",)),
        ("symmetric", ("S:same text",)),
    ]


def test_lifecycle_launcher_pins_and_forwards_task_contract(tmp_path: Path) -> None:
    factory_calls: list[dict[str, object]] = []

    def factory(*_args: object, **kwargs: object) -> object:
        factory_calls.append(kwargs)
        return object()

    def metadata(_endpoint: str, names: tuple[str, ...]) -> dict[str, dict[str, object]]:
        return {name: {"name": name, "digest": "a" * 64} for name in names}

    class Adapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate_session(self, session: object, profile: object, repetition: int) -> object:
            del profile, repetition
            from tests.test_lifecycle_target_smoke import _observation

            return _observation(getattr(session, "session_id"))

    report = run_lifecycle_target_smoke(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        endpoint="http://example.invalid",
        model="model-for-test",
        embedding_model="embedding-for-test",
        profile_names=("baseline",),
        service_factory_builder=factory,
        adapter_type=Adapter,
        model_metadata_loader=metadata,
        query_prefix="Q:",
        document_prefix="D:",
        symmetric_prefix="S:",
    )
    assert report["status"] == "complete"
    assert factory_calls and all(call["query_prefix"] == "Q:" for call in factory_calls)
    assert report["model_pins"]["options"]["embedding_task_contract"]["version"] == (
        "explicit-prefix-v1"
    )
