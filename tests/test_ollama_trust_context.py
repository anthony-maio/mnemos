"""Cognition requests amortize default trust loading without retaining transports."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import ssl
import threading
import time

import httpx
import pytest

from mnemos.utils import llm
from mnemos.utils.reliability import MnemosExternalServiceError, RetryPolicy


@pytest.fixture
def transport_probe(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    real_client = httpx.AsyncClient
    probe = {"contexts": [], "clients": [], "kwargs": [], "handler": None}

    def context_factory(*args, **kwargs):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        probe["contexts"].append(context)
        return context

    async def handler(request):
        if probe["handler"] is not None:
            return await probe["handler"](request)
        return httpx.Response(200, json={"response": "ok", "done": True})

    def client_factory(**kwargs):
        probe["kwargs"].append(kwargs)
        client = real_client(**kwargs, transport=httpx.MockTransport(handler))
        probe["clients"].append(client)
        return client

    monkeypatch.setattr(llm.httpx, "create_ssl_context", context_factory)
    monkeypatch.setattr(llm.httpx, "AsyncClient", client_factory)
    probe["factory"] = context_factory
    probe["real_client"] = real_client
    return probe


def test_default_trust_is_lazy_and_shared_across_separate_event_loops(transport_probe):
    observed = []
    provider = llm.OllamaProvider(timeout=17.0, response_observer=observed.append)
    assert transport_probe["contexts"] == []
    for _ in range(3):
        assert asyncio.run(provider.predict("fixture")) == "ok"
    assert len(transport_probe["contexts"]) == 1
    context = transport_probe["contexts"][0]
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    assert transport_probe["kwargs"] == [{"timeout": 17.0, "verify": context}] * 3
    assert len(transport_probe["clients"]) == len(observed) == 3
    assert all(client.is_closed for client in transport_probe["clients"])


def test_concurrent_first_calls_load_one_default_context(monkeypatch, transport_probe):
    build = transport_probe["factory"]
    ready = threading.Barrier(4)

    def delayed_build(*args, **kwargs):
        time.sleep(0.02)
        return build(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "create_ssl_context", delayed_build)
    provider = llm.OllamaProvider()

    def request(_):
        ready.wait(timeout=5)
        return asyncio.run(provider.predict("fixture"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(request, range(4))) == ["ok"] * 4
    assert len(transport_probe["contexts"]) == 1
    assert all(client.is_closed for client in transport_probe["clients"])


@pytest.mark.parametrize("variable", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_custom_trust_environment_bypasses_default_cache(monkeypatch, transport_probe, variable):
    provider = llm.OllamaProvider()
    assert asyncio.run(provider.predict("default")) == "ok"
    for value in ("first-fixture-path", "second-fixture-path"):
        monkeypatch.setenv(variable, value)
        assert asyncio.run(provider.predict("custom")) == "ok"
        assert transport_probe["kwargs"][-1]["verify"] is True
    monkeypatch.delenv(variable)
    assert asyncio.run(provider.predict("default again")) == "ok"
    assert len(transport_probe["contexts"]) == 1
    assert transport_probe["kwargs"][-1]["verify"] is transport_probe["contexts"][0]


def test_failed_default_trust_initialization_is_not_cached(monkeypatch, transport_probe):
    build = transport_probe["factory"]
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("fixture trust load failed")
        return build(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "create_ssl_context", fail_once)
    provider = llm.OllamaProvider()
    with pytest.raises(MnemosExternalServiceError):
        asyncio.run(provider.predict("first"))
    assert asyncio.run(provider.predict("second")) == "ok"
    assert attempts == 2 and len(transport_probe["contexts"]) == 1


@pytest.mark.asyncio
async def test_retry_reuses_trust_but_closes_each_attempt(monkeypatch, transport_probe):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("fixture transient", request=request)
        return httpx.Response(200, json={"response": "ok"})

    transport_probe["handler"] = handler
    monkeypatch.setattr(RetryPolicy, "backoff_seconds", lambda *args: 0.0)
    assert await llm.OllamaProvider().predict("fixture") == "ok"
    assert calls == len(transport_probe["clients"]) == 2
    assert len(transport_probe["contexts"]) == 1
    assert all(client.is_closed for client in transport_probe["clients"])


@pytest.mark.asyncio
async def test_cancellation_closes_request_transport(transport_probe):
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.Event().wait()

    transport_probe["handler"] = handler
    task = asyncio.create_task(llm.OllamaProvider().predict("fixture"))
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport_probe["clients"][0].is_closed


def test_invalid_custom_ca_is_not_masked_by_cached_default(monkeypatch, transport_probe, tmp_path):
    provider = llm.OllamaProvider(base_url="https://127.0.0.1:9")
    assert asyncio.run(provider.predict("prepare default")) == "ok"
    monkeypatch.setattr(llm.httpx, "AsyncClient", transport_probe["real_client"])
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    with pytest.raises(MnemosExternalServiceError) as caught:
        asyncio.run(provider.predict("must fail before transport"))
    assert isinstance(caught.value.__cause__, FileNotFoundError)
