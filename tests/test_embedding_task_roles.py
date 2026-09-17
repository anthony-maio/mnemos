import threading
from typing import Any

import pytest

from mnemos.utils.embeddings import (
    EmbeddingProvider,
    PrefixEmbeddingProvider,
    embed_batch_for_role,
    embed_text_for_role,
    embed_batch_async,
    embed_text_async,
)


class RecordingProvider(EmbeddingProvider):
    def __init__(self, *, offload: bool = False) -> None:
        self.single: list[str] = []
        self.batches: list[list[str]] = []
        self.threads: list[int] = []
        self.offload = offload

    @property
    def dim(self) -> int:
        return 2

    @property
    def should_offload_in_async(self) -> bool:
        return self.offload

    def embed(self, text: str) -> list[float]:
        self.single.append(text)
        self.threads.append(threading.get_ident())
        return [float(len(text)), 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        self.threads.append(threading.get_ident())
        return [[float(len(text)), 1.0] for text in texts]


def test_prefix_wrapper_applies_each_role_once_for_single_and_batch_calls() -> None:
    inner = RecordingProvider()
    provider = PrefixEmbeddingProvider(
        inner,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        symmetric_prefix="symmetric: ",
    )

    provider.embed_query("same")
    provider.embed_document("same")
    provider.embed("same")
    provider.embed_batch_queries(["same", "other"])
    provider.embed_batch_documents(["same"])
    provider.embed_batch(["same"])

    assert inner.single == [
        "search_query: same",
        "search_document: same",
        "symmetric: same",
    ]
    assert inner.batches == [
        ["search_query: same", "search_query: other"],
        ["search_document: same"],
        ["symmetric: same"],
    ]


def test_empty_prefix_wrapper_preserves_legacy_payloads_and_provider_properties() -> None:
    inner = RecordingProvider()
    provider = PrefixEmbeddingProvider(inner)

    assert provider.embed_query("text") == inner.embed("text")
    assert provider.embed_document("text") == inner.embed("text")
    assert provider.embed_symmetric("text") == inner.embed("text")
    assert provider.dim == inner.dim
    assert provider.should_offload_in_async == inner.should_offload_in_async
    assert inner.single == ["text", "text", "text", "text", "text", "text"]


def test_sync_role_helpers_dispatch_prefix_wrapper() -> None:
    inner = RecordingProvider()
    provider = PrefixEmbeddingProvider(inner, document_prefix="document: ")

    embed_text_for_role(provider, "fact", role="document")
    embed_batch_for_role(provider, ["one", "two"], role="document")

    assert inner.single == ["document: fact"]
    assert inner.batches == [["document: one", "document: two"]]


class DuckProvider:
    should_offload_in_async = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(("single", text))
        return [1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("batch", list(texts)))
        return [[1.0] for _ in texts]


@pytest.mark.asyncio
async def test_async_helpers_fall_back_to_legacy_duck_methods_and_validate_role() -> None:
    provider = DuckProvider()

    assert await embed_text_async(provider, "query", role="query") == [1.0]
    assert await embed_batch_async(provider, ["document"], role="document") == [[1.0]]
    assert provider.calls == [("single", "query"), ("batch", ["document"])]

    invalid_role: Any = "invalid"
    with pytest.raises(ValueError, match="embedding role"):
        await embed_text_async(provider, "bad", role=invalid_role)
    with pytest.raises(ValueError, match="embedding role"):
        await embed_batch_async(provider, ["bad"], role=invalid_role)


@pytest.mark.asyncio
async def test_async_helpers_dispatch_roles_and_preserve_offload() -> None:
    inner = RecordingProvider(offload=True)
    provider = PrefixEmbeddingProvider(inner, query_prefix="query: ")
    caller_thread = threading.get_ident()

    await embed_text_async(provider, "one", role="query")
    await embed_batch_async(provider, ["two", "three"], role="document")

    assert inner.single == ["query: one"]
    assert inner.batches == [["two", "three"]]
    assert inner.threads
    assert all(thread_id != caller_thread for thread_id in inner.threads)


@pytest.mark.asyncio
async def test_batch_role_fallback_does_not_reapply_a_prefix_per_item() -> None:
    inner = RecordingProvider()
    provider = PrefixEmbeddingProvider(inner, query_prefix="query: ")

    await embed_batch_async(provider, ["a", "b"], role="query")

    assert inner.batches == [["query: a", "query: b"]]
    assert inner.single == []
