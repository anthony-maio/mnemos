"""Opt-in benchmark observer for exact synchronous Ollama chat requests.

This module patches ``httpx.Client.send`` only inside ``observe_requests``.
It does not observe asynchronous clients and is not part of the Mnemos runtime.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/chat/completions"
    ):
        raise ValueError("endpoint must be a credential-free local Ollama chat endpoint")
    if parsed.port not in {None, 11434}:
        raise ValueError("endpoint must use the local Ollama port")
    return endpoint.rstrip("/")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def observe_requests(endpoint: str, output_dir: Path) -> AbstractContextManager[None]:
    """Observe eager synchronous POST bodies sent to one local Ollama endpoint.

    The observer records only request bytes and non-sensitive method/URL/hash
    metadata. Matching streamed bodies are rejected before consumption. Async
    HTTPX clients are outside this observer's scope.
    """

    safe_endpoint = _validate_endpoint(endpoint)
    output = Path(output_dir)
    if output.exists() and any(output.glob("request-*")):
        raise ValueError("output directory already contains request artifacts")

    @contextmanager
    def manager() -> Iterator[None]:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.glob("request-*")):
            raise ValueError("output directory already contains request artifacts")
        counter = 1
        lock = threading.Lock()
        original_send = httpx.Client.send

        def send(
            self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
        ) -> httpx.Response:
            nonlocal counter
            matches = request.method == "POST" and str(request.url) == safe_endpoint
            if not matches:
                return original_send(self, request, *args, **kwargs)
            if not isinstance(request.stream, httpx.ByteStream):
                raise ValueError(
                    "matching Ollama request body is streamed; observer requires eager bytes"
                )
            body = request.content
            with lock:
                ordinal = counter
                counter += 1
            body_file = f"request-{ordinal:04d}.bin"
            metadata_file = f"request-{ordinal:04d}.json"
            with (output / body_file).open("xb") as stream:
                stream.write(body)
            _atomic_json(
                output / metadata_file,
                {
                    "body_file": body_file,
                    "byte_count": len(body),
                    "method": request.method,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "url": safe_endpoint,
                },
            )
            return original_send(self, request, *args, **kwargs)

        httpx.Client.send = send  # type: ignore[method-assign]
        try:
            yield
        finally:
            httpx.Client.send = original_send  # type: ignore[method-assign]

    return manager()
