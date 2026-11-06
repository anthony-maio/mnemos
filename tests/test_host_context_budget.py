"""The retrieval service exposes the budget actually used by an injected host counter."""

import pytest

from mnemos.config import MnemosConfig
from mnemos.context_budget import HostTokenizer
from mnemos.contracts import RetrievalRequest
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.reliability import MnemosTransientError
from mnemos.utils.storage import SQLiteStore
from tests.test_retrieval_comparison import add_fact
from tests.test_retrieval_contract import FixedEmbedding, make_engine


def service(tmp_path, tokenizer):
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="local", enabled=True)
    context = registry.resolve(tmp_path, user_profile_id="local", session_id="s", host="manual")
    config = MnemosConfig()
    for flag in type(config.modules).model_fields:
        setattr(config.modules, flag, False)
    engine = MnemosEngine(
        config=config,
        llm=UnavailableLLMProvider(),
        embedder=FixedEmbedding(),
        store=SQLiteStore(str(tmp_path / "memory.sqlite")),
        context_tokenizer=tokenizer,
    )
    return LocalMemoryService(engine, registry), context


@pytest.mark.asyncio
async def test_host_budget_is_used_for_selection_and_service_reporting(tmp_path):
    tokenizer = HostTokenizer("test-word-counter", "v1", lambda text: len(text.split()))
    memory, context = service(tmp_path, tokenizer)
    try:
        add_fact(memory, context, "a", "Juniper uses Python for its backend.")
        add_fact(memory, context, "b", "Juniper uses SQLite for local storage.")
        result = await memory.retrieve(
            RetrievalRequest(query="Juniper", context=context, max_context_tokens=20)
        )
        assert [fact["id"] for fact in result["facts"]] == ["a", "b"]
        assert not result.get("degraded")
        assert result["context_token_count"] == len(result["context"].split())
        assert result["context_token_upper_bound"] <= 20 < len(result["context"].encode())
        assert result["context_byte_count"] == len(result["context"].encode())
        assert result["token_count_method"] == "host_tokenizer"
        assert result["tokenizer_name"] == "test-word-counter"
        assert result["tokenizer_revision"] == "v1"
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_counter_failure_repacks_before_recording_memory_access(tmp_path):
    def fails_on_second(text):
        if text.count("\n") > 1:
            raise RuntimeError("unavailable")
        return 1

    memory, context = service(tmp_path, HostTokenizer("test", "v1", fails_on_second))
    try:
        add_fact(memory, context, "a", "Juniper " * 100)
        add_fact(memory, context, "b", "Juniper uses Python.")
        result = await memory.retrieve(
            RetrievalRequest(query="Juniper", context=context, max_context_tokens=80)
        )
        assert [fact["id"] for fact in result["facts"]] == ["b"]
        assert result["token_count_method"] == "utf8_bytes_upper_bound"
        assert result["context_token_count"] is None
        assert result["tokenizer_fallback_reason"] == "tokenizer_error"
        assert result["context_token_upper_bound"] == len(result["context"].encode())
        assert memory.engine.store.get("a").access_count == 0
        assert memory.engine.store.get("b").access_count == 1
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_lexical_outage_qualifier_is_counted_with_host_tokenizer(tmp_path):
    memory, context = service(
        tmp_path, HostTokenizer("test-word-counter", "v1", lambda text: len(text.split()))
    )
    try:
        add_fact(memory, context, "a", "Juniper uses Python.")

        class OfflineEmbedding(FixedEmbedding):
            def embed(self, text):
                raise MnemosTransientError("offline")

        memory.engine._embedder = OfflineEmbedding()
        result = await memory.retrieve(
            RetrievalRequest(query="Juniper", context=context, max_context_tokens=40)
        )
        assert result["degraded"] is True
        assert "semantic retrieval unavailable" in result["context"]
        assert result["context_token_count"] == len(result["context"].split())
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_revision_during_access_update_cannot_replace_budgeted_snapshot(monkeypatch):
    engine = make_engine()
    original = engine.store.mark_retrieved_many

    def revise_after_access(ids, **kwargs):
        original(ids, **kwargs)
        current = engine.store.get("semantic").model_copy(deep=True)
        current.content = "Python " * 1000
        current.version += 1
        engine.store.update(current.id, current)

    monkeypatch.setattr(engine.store, "mark_retrieved_many", revise_after_access)
    result = await engine.retrieve("Python", top_k=1, max_context_tokens=150)
    assert result == []
