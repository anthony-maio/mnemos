"""Provider resources follow the configured engine and service lifetime."""

import asyncio
import threading

import pytest

from mnemos import runtime
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.service import MnemosService
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils import embeddings, llm
from mnemos.utils.storage import InMemoryStore


class ClosingEmbedding(SimpleEmbeddingProvider):
    def __init__(self, events, *, fail=False):
        super().__init__(dim=8)
        self.events = events
        self.fail = fail

    def close(self):
        self.events.append("embedding")
        if self.fail:
            raise RuntimeError("embedding cleanup failed")


class ClosingLLM(MockLLMProvider):
    def __init__(self, events):
        super().__init__()
        self.events = events

    async def aclose(self):
        self.events.append("llm")


class ClosingStore(InMemoryStore):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def close(self):
        self.events.append("store")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_runtime_owns_created_providers(monkeypatch, tmp_path, configured):
    events = []
    monkeypatch.setattr(runtime, "build_embedder_from_settings", lambda _: ClosingEmbedding(events))
    monkeypatch.setattr(runtime, "build_llm_from_settings", lambda _: ClosingLLM(events))
    monkeypatch.setattr(runtime, "_configured_store", lambda _: ClosingStore(events))
    env = {"MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml")}
    if configured:
        env["MNEMOS_LLM_PROVIDER"] = "ollama"
    engine = runtime.build_configured_engine(env=env, cwd=tmp_path)
    close = getattr(engine, "aclose", None)
    assert callable(close), "Configured engines must release their provider clients"
    await asyncio.gather(close(), close())
    assert events.count("embedding") == 1
    assert events.count("llm") == int(configured)
    assert events.count("store") == 1
    assert events[-1] == "store"


@pytest.mark.asyncio
async def test_borrowed_providers_remain_caller_owned():
    events = []
    engine = MnemosEngine(
        llm=ClosingLLM(events), embedder=ClosingEmbedding(events), store=ClosingStore(events)
    )
    close = getattr(engine, "aclose", None)
    assert callable(close), "Engine cleanup must distinguish borrowed provider ownership"
    await close()
    assert events == ["store"]
    assert await engine.llm.predict("test")
    assert engine.embedder.embed("test")


@pytest.mark.asyncio
async def test_provider_cleanup_failure_does_not_skip_other_resources(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        runtime, "build_embedder_from_settings", lambda _: ClosingEmbedding(events, fail=True)
    )
    monkeypatch.setattr(runtime, "build_llm_from_settings", lambda _: ClosingLLM(events))
    monkeypatch.setattr(runtime, "_configured_store", lambda _: ClosingStore(events))
    engine = runtime.build_configured_engine(
        env={"MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"), "MNEMOS_LLM_PROVIDER": "ollama"},
        cwd=tmp_path,
    )
    close = getattr(engine, "aclose", None)
    assert callable(close), "Cleanup must release the store even when a provider fails"
    with pytest.raises(RuntimeError, match="embedding cleanup failed"):
        await close()
    assert sorted(events) == ["embedding", "llm", "store"]
    assert events[-1] == "store"

    engine.embedder.fail = False
    await close()
    assert events.count("embedding") == 2, "Failed cleanup must be retryable"
    assert events.count("llm") == 1
    assert events.count("store") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", ["llm", "embedding", "neither"])
async def test_engine_owns_only_the_defaults_it_created(monkeypatch, borrowed):
    events = []
    monkeypatch.setattr(embeddings, "SimpleEmbeddingProvider", lambda **_: ClosingEmbedding(events))
    monkeypatch.setattr(llm, "UnavailableLLMProvider", lambda: ClosingLLM(events))
    kwargs = {"store": ClosingStore(events)}
    if borrowed == "llm":
        kwargs["llm"] = ClosingLLM(events)
    elif borrowed == "embedding":
        kwargs["embedder"] = ClosingEmbedding(events)
    engine = MnemosEngine(**kwargs)
    await engine.aclose()
    assert events.count("llm") == int(borrowed != "llm")
    assert events.count("embedding") == int(borrowed != "embedding")
    assert events[-1] == "store"


def test_service_cancels_operations_before_closing_owned_providers(monkeypatch, tmp_path):
    events = []
    started = threading.Event()
    monkeypatch.setattr(runtime, "build_embedder_from_settings", lambda _: ClosingEmbedding(events))
    monkeypatch.setattr(runtime, "build_llm_from_settings", lambda _: ClosingLLM(events))
    monkeypatch.setattr(runtime, "_configured_store", lambda _: ClosingStore(events))

    class Dispatcher(LocalMemoryService):
        async def unfinished(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append("cancelled")

    def factory():
        engine = runtime.build_configured_engine(
            env={
                "MNEMOS_CONFIG_PATH": str(tmp_path / "missing.toml"),
                "MNEMOS_LLM_PROVIDER": "ollama",
            },
            cwd=tmp_path,
        )
        dispatcher = Dispatcher(engine, ScopeRegistry())
        asyncio.get_event_loop().create_task(dispatcher.unfinished())
        return dispatcher

    service = MnemosService(dispatcher_factory=factory, descriptor_path=tmp_path / "service.json")
    service.start()
    try:
        assert started.wait(3)
    finally:
        service.stop()
    assert events[0] == "cancelled"
    assert events.count("embedding") == 1, "Service shutdown must close provider clients"
    assert events.count("llm") == 1
    assert events[-1] == "store"


@pytest.mark.asyncio
async def test_cancelled_close_drains_provider_before_storage_and_other_close():
    events = []
    started = threading.Event()
    release = threading.Event()

    class BlockingEmbedding(ClosingEmbedding):
        def close(self):
            events.append("embedding-start")
            started.set()
            assert release.wait(5), "Test did not release provider cleanup"
            events.append("embedding-done")

    engine = MnemosEngine(
        embedder=BlockingEmbedding(events),
        llm=ClosingLLM(events),
        store=ClosingStore(events),
        owns_providers=True,
    )
    first = asyncio.create_task(engine.aclose())
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        second = asyncio.create_task(engine.aclose())
        # Let both callers run while the worker remains blocked on the event.
        for _ in range(10):
            await asyncio.sleep(0)
        assert "store" not in events, "Cancellation must not close storage before the provider"
        assert events.count("embedding-start") == 1
    finally:
        release.set()
        results = await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert events == ["embedding-start", "embedding-done", "llm", "store"]
