"""Existing scoped text remains searchable when the embedding service is down."""

import asyncio
import json
import socket

import pytest

from mnemos.scopes import context_metadata
from mnemos.retrieval import context_token_upper_bound
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import OllamaEmbeddingProvider, SimpleEmbeddingProvider
from mnemos.utils.reliability import (
    MnemosConfigurationError,
    MnemosExternalServiceError,
    MnemosTransientError,
)
from tests.test_memory_service import service


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["baseline", "biomimetic"])
async def test_closed_embedding_endpoint_keeps_existing_sqlite_recall(tmp_path, profile):
    memory, context = service(tmp_path)
    target = MemoryChunk(
        content="Deployment uses the maple-heron release label.",
        embedding=[1.0, 0.0],
        metadata={**context_metadata(context, "project"), "assertion_status": "asserted"},
    )
    memory.engine.store.store(target)
    # A bound but non-listening local socket cannot be taken by another server.
    with socket.socket() as stopped_endpoint:
        stopped_endpoint.bind(("127.0.0.1", 0))
        port = stopped_endpoint.getsockname()[1]
        provider = OllamaEmbeddingProvider(base_url=f"http://127.0.0.1:{port}", timeout=0.1)
        memory.engine._embedder = provider
        try:
            result = await memory.dispatch(
                "retrieve",
                {"context": context.model_dump(), "query": "maple-heron", "profile": profile},
            )
            assert [fact["id"] for fact in result["facts"]] == [target.id]
            assert result["degraded"] is True
            assert result["retrieval"]["mode"] == "lexical_only"
            assert result["retrieval"]["reason"] == "embedding_unavailable"
            assert result["facts"][0]["scores"]["semantic"] is None
            assert result["facts"][0]["scores"]["state_status"] == "not_measured"
            assert result["facts"][0]["scores"]["graph_status"] == "not_used"
        finally:
            provider.close()
            memory.engine.store.close()


class OfflineEmbeddings(SimpleEmbeddingProvider):
    def embed_query(self, text):
        raise MnemosTransientError("private-provider-error-canary")


def seed(memory, context, content, *, metadata=None):
    chunk = MemoryChunk(
        content=content,
        embedding=[1.0, 0.0],
        metadata={
            **context_metadata(context, "project"),
            "assertion_status": "asserted",
            **(metadata or {}),
        },
    )
    memory.engine.store.store(chunk)
    return chunk


@pytest.mark.asyncio
async def test_lexical_outage_preserves_scope_view_budget_and_empty_status(
    tmp_path, monkeypatch, caplog
):
    memory, context = service(tmp_path)
    memory.engine._embedder = OfflineEmbeddings()
    try:
        accepted = seed(memory, context, "maple deployment label")
        capture = seed(
            memory,
            context,
            "maple captured conversation",
            metadata={"source": "captured_passthrough", "assertion_status": "unknown"},
        )
        # More unauthorized matches than the candidate budget cannot crowd out
        # the authorized row or enter the response through fallback.
        for index in range(25):
            seed(
                memory,
                context,
                f"maple foreign-profile {index}",
                metadata={"user_profile_id": "other"},
            )
            seed(
                memory,
                context,
                f"maple foreign-project {index}",
                metadata={"project_id": "other", "scope_id": "other"},
            )
            seed(
                memory,
                context,
                f"maple workspace {index}",
                metadata={"scope": "workspace", "scope_id": "unjoined", "workspace_id": "unjoined"},
            )

        def forbidden(*args, **kwargs):
            raise AssertionError("offline recall must not request state or graph inference")

        monkeypatch.setattr(memory.engine.affective_router, "classify_state", forbidden)
        monkeypatch.setattr(memory.engine.spreading_activation, "retrieve", forbidden)
        payload = {
            "context": context.model_dump(),
            "query": "maple",
            "profile": "biomimetic",
            "top_k": 1,
        }
        result = await memory.dispatch("retrieve", payload)
        assert [item["id"] for item in result["facts"]] == [accepted.id]
        assert "semantic retrieval unavailable" in result["context"]
        assert result["context_token_upper_bound"] == len(result["context"].encode())
        assert result["facts"][0]["scores"]["lexical_rank"] == 1
        assert result["facts"][0]["scores"]["total"] is None
        assert "private-provider-error-canary" not in json.dumps(result)
        assert "private-provider-error-canary" not in caplog.text
        assert memory.engine.store.get(accepted.id).access_count == 1
        assert memory.engine.store.get(capture.id).access_count == 0
        assert "retrieval_trace" not in memory.engine.store.get(accepted.id).metadata
        evidence = await memory.dispatch("retrieve", {**payload, "view": "captured_evidence"})
        assert evidence["facts"] == []
        assert [item["id"] for item in evidence["captured_evidence"]] == [capture.id]
        assert "unverified" in evidence["evidence_context"]
        assert "semantic retrieval unavailable" in evidence["evidence_context"]
        assert evidence["context_token_upper_bound"] == len(evidence["evidence_context"].encode())
        for query, budget in [
            ("nonexistentcanary", 1200),
            ("maple", context_token_upper_bound(accepted)),
        ]:
            empty = await memory.dispatch(
                "retrieve", {**payload, "query": query, "max_context_tokens": budget}
            )
            assert empty["facts"] == [] and empty["context"] == ""
            assert empty["degraded"] and empty["retrieval"]["mode"] == "lexical_only"
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"top_k": 0}, {"max_context_tokens": 0}, {"query": "   "}])
async def test_disabled_retrieval_does_not_call_unavailable_provider(tmp_path, options):
    memory, context = service(tmp_path)

    class Forbidden(OfflineEmbeddings):
        def embed_query(self, text):
            raise AssertionError("disabled query called provider")

    memory.engine._embedder = Forbidden()
    try:
        result = await memory.dispatch(
            "retrieve", {"context": context.model_dump(), "query": "maple", **options}
        )
        assert result["facts"] == []
        assert not result.get("degraded", False)
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_store_without_lexical_capability_preserves_provider_error():
    from mnemos.engine import MnemosEngine
    from mnemos.utils.storage import InMemoryStore
    from mnemos.utils.llm import UnavailableLLMProvider

    engine = MnemosEngine(
        embedder=OfflineEmbeddings(), llm=UnavailableLLMProvider(), store=InMemoryStore()
    )
    try:
        with pytest.raises(MnemosTransientError):
            await engine.retrieve("maple")
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_offline_global_recall_requires_allowed_scope_and_same_profile(tmp_path):
    memory, context = service(tmp_path)
    memory.engine._embedder = OfflineEmbeddings()
    try:
        shared = seed(
            memory,
            context,
            "maple shared preference",
            metadata={"scope": "global", "scope_id": None, "project_id": None},
        )
        seed(
            memory,
            context,
            "maple secret preference",
            metadata={
                "scope": "global",
                "scope_id": None,
                "project_id": None,
                "user_profile_id": "other",
            },
        )
        payload = {"context": context.model_dump(), "query": "maple"}
        result = await memory.dispatch("retrieve", payload)
        assert [fact["id"] for fact in result["facts"]] == [shared.id]
        blocked = await memory.dispatch("retrieve", {**payload, "allowed_scopes": ["project"]})
        assert blocked["facts"] == [] and blocked["degraded"]
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [MnemosConfigurationError, MnemosExternalServiceError, ValueError]
)
async def test_unrelated_provider_errors_do_not_become_lexical_success(tmp_path, error):
    memory, context = service(tmp_path)

    class Broken(OfflineEmbeddings):
        def embed_query(self, text):
            raise error("fixture failure")

    memory.engine._embedder = Broken()
    try:
        seed(memory, context, "maple deployment")
        with pytest.raises(error, match="fixture failure"):
            await memory.dispatch("retrieve", {"context": context.model_dump(), "query": "maple"})
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_lexical_database_error_is_not_swallowed(tmp_path, monkeypatch):
    memory, context = service(tmp_path)
    memory.engine._embedder = OfflineEmbeddings()

    def broken(*args, **kwargs):
        raise RuntimeError("database error")

    monkeypatch.setattr(memory.engine.store, "retrieve_lexical", broken)
    try:
        with pytest.raises(RuntimeError, match="database error"):
            await memory.dispatch("retrieve", {"context": context.model_dump(), "query": "maple"})
    finally:
        memory.engine.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", ["correct", "forget"])
async def test_fallback_reads_current_rows_after_embedding_wait(tmp_path, monkeypatch, revision):
    from mnemos import engine as engine_module

    memory, context = service(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    task = None

    async def unavailable(*args, **kwargs):
        entered.set()
        await release.wait()
        raise MnemosTransientError("fixture outage")

    monkeypatch.setattr(engine_module, "embed_text_async", unavailable)
    try:
        old = seed(memory, context, "maple old label")
        task = asyncio.create_task(
            memory.dispatch("retrieve", {"context": context.model_dump(), "query": "maple"})
        )
        await asyncio.wait_for(entered.wait(), 5)
        if revision == "forget":
            await memory.dispatch(
                "forget",
                {
                    "context": context.model_dump(),
                    "fact_id": old.id,
                    "expected_version": old.version,
                },
            )
        else:
            current = old.model_copy(update={"content": "maple corrected label", "version": 2})
            memory.engine.store.update(old.id, current)
        release.set()
        result = await asyncio.wait_for(task, 5)
        assert result["degraded"]
        if revision == "forget":
            assert result["facts"] == []
        else:
            assert [(row["content"], row["version"]) for row in result["facts"]] == [
                ("maple corrected label", 2)
            ]
        assert "maple old label" not in result["context"]
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        memory.engine.store.close()


@pytest.mark.asyncio
async def test_fallback_diagnostics_do_not_leak_into_recovered_request(tmp_path):
    memory, context = service(tmp_path)
    memory.engine._embedder = OfflineEmbeddings()
    payload = {"context": context.model_dump(), "query": "maple"}
    try:
        target = seed(memory, context, "maple")
        offline = await memory.dispatch("retrieve", payload)
        memory.engine._embedder = SimpleEmbeddingProvider(dim=2)
        target.embedding = memory.engine.embedder.embed_query("maple")
        memory.engine.store.update(target.id, target)
        recovered = await memory.dispatch("retrieve", payload)
        assert offline["degraded"] and offline["retrieval"]["mode"] == "lexical_only"
        assert not recovered.get("degraded", False)
        assert recovered["facts"][0]["scores"]["semantic"] == pytest.approx(1)
        assert "semantic retrieval unavailable" not in recovered["context"]
    finally:
        memory.engine.store.close()
