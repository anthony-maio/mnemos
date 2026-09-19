"""Lifecycle and reuse tests for the Ollama embedding transport."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from mnemos.utils import embeddings as embeddings_module


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[httpx.Client]:
    clients: list[httpx.Client] = []
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        client = real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(embeddings_module.httpx, "Client", factory)
    monkeypatch.setattr(
        embeddings_module.httpx,
        "post",
        lambda *args, **kwargs: pytest.fail("provider used per-request httpx.post"),
    )
    return clients


def test_ollama_embedding_reuses_one_client_for_single_batch_and_fallback_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/embed":
            payload = request.read()
            if b'"input":"first"' in payload:
                return httpx.Response(404, request=request)
            if b'"input":["first","second"]' in payload:
                return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})
            return httpx.Response(200, json={"embedding": [1.0, 0.0]})
        return httpx.Response(200, json={"embedding": [1.0, 0.0]})

    clients = _install_mock_client(monkeypatch, handler)
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")

    assert provider.embed("first") == [1.0, 0.0]
    assert provider.embed_batch(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    provider.close()

    assert len(clients) == 1
    assert clients[0].is_closed
    assert [request.url.path for request in requests] == [
        "/api/embed",
        "/api/embeddings",
        "/api/embed",
    ]


def test_ollama_embedding_lazily_creates_one_client_for_concurrent_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json={"embedding": [1.0, 0.0]})

    clients = _install_mock_client(monkeypatch, handler)
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")

    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(provider.embed, [f"text-{i}" for i in range(24)]))
    provider.close()

    assert vectors == [[1.0, 0.0]] * 24
    assert len(clients) == 1


def test_ollama_embedding_close_waits_for_active_request_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_started = threading.Event()
    release_request = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        request_started.set()
        assert release_request.wait(2.0)
        return httpx.Response(200, json={"embedding": [1.0, 0.0]})

    clients = _install_mock_client(monkeypatch, handler)
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")
    result: list[list[float]] = []
    request_thread = threading.Thread(target=lambda: result.append(provider.embed("active")))
    request_thread.start()
    assert request_started.wait(2.0)

    close_finished = threading.Event()

    def finish_close() -> None:
        provider.close()
        close_finished.set()

    close_thread = threading.Thread(target=finish_close)
    close_thread.start()
    assert not close_finished.wait(0.05)

    release_request.set()
    request_thread.join(2.0)
    close_thread.join(2.0)
    provider.close()

    assert result == [[1.0, 0.0]]
    assert close_finished.is_set()
    assert clients[0].is_closed


def test_ollama_embedding_close_rejects_new_requests_without_network_or_retry_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(200, json={"embedding": [1.0, 0.0]}),
    )
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")
    provider.close()

    with pytest.raises(RuntimeError, match="closed"):
        provider.embed("after close")
    provider.close()
    assert clients == []


def test_ollama_embedding_retries_on_transient_status_without_rebuilding_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"embedding": [1.0, 0.0]})

    clients = _install_mock_client(monkeypatch, handler)
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")

    assert provider.embed("retry") == [1.0, 0.0]
    provider.close()

    assert attempts == 2
    assert len(clients) == 1


def test_prefix_embedding_provider_forwards_close() -> None:
    class Inner(embeddings_module.EmbeddingProvider):
        def __init__(self) -> None:
            self.closed = 0

        @property
        def dim(self) -> int:
            return 2

        def embed(self, text: str) -> list[float]:
            _ = text
            return [1.0, 0.0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed(text) for text in texts]

        def close(self) -> None:
            self.closed += 1

    inner = Inner()
    wrapped = embeddings_module.PrefixEmbeddingProvider(inner)

    wrapped.close()

    assert inner.closed == 1


def test_ollama_embedding_observer_can_close_provider_after_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(200, json={"embedding": [1.0, 0.0]}),
    )
    provider: embeddings_module.OllamaEmbeddingProvider

    def observer(response: dict[str, Any]) -> None:
        _ = response
        provider.close()

    provider = embeddings_module.OllamaEmbeddingProvider(
        base_url="http://ollama", response_observer=observer
    )
    errors: list[BaseException] = []
    thread = threading.Thread(
        target=lambda: _run_and_capture(lambda: provider.embed("observed"), errors),
        daemon=True,
    )
    thread.start()
    thread.join(2.0)

    assert not thread.is_alive()
    assert errors == []
    assert clients[0].is_closed


def test_concurrent_close_waits_for_the_first_client_close_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(200, json={"embedding": [1.0, 0.0]}),
    )
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")
    assert provider.embed("warm") == [1.0, 0.0]

    close_started = threading.Event()
    release_close = threading.Event()
    original_close = clients[0].close

    def blocking_close() -> None:
        close_started.set()
        assert release_close.wait(2.0)
        original_close()

    clients[0].close = blocking_close  # type: ignore[method-assign]
    first_finished = threading.Event()
    second_finished = threading.Event()
    first = threading.Thread(target=lambda: _run_and_signal(provider.close, first_finished))
    second = threading.Thread(target=lambda: _run_and_signal(provider.close, second_finished))
    first.start()
    assert close_started.wait(2.0)
    second.start()
    assert not second_finished.wait(0.05)

    release_close.set()
    first.join(2.0)
    second.join(2.0)

    assert first_finished.is_set()
    assert second_finished.is_set()
    assert clients[0].is_closed


def test_close_failure_remains_visible_after_httpx_marks_client_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTransport(httpx.MockTransport):
        def close(self) -> None:
            raise RuntimeError("fixture transport close failed")

    clients: list[httpx.Client] = []
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        client = real_client(
            *args,
            transport=FailingTransport(
                lambda request: httpx.Response(200, json={"embedding": [1.0, 0.0]})
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(embeddings_module.httpx, "Client", factory)
    provider = embeddings_module.OllamaEmbeddingProvider(base_url="http://ollama")
    assert provider.embed("close failure") == [1.0, 0.0]

    with pytest.raises(RuntimeError, match="fixture transport close failed"):
        provider.close()
    assert clients[0].is_closed

    with pytest.raises(RuntimeError, match="fixture transport close failed"):
        provider.close()


def _run_and_signal(function: Any, finished: threading.Event) -> None:
    try:
        function()
    finally:
        finished.set()


def _run_and_capture(function: Any, errors: list[BaseException]) -> None:
    try:
        function()
    except BaseException as exc:  # pragma: no cover - assertion reports failures
        errors.append(exc)
