"""Uncached, loopback-only native token counting for the integration experiment."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

import httpx


class LiveTokenCounter:
    """Own a bounded HTTP client and retain each original count observation."""

    def __init__(self, endpoint: str, *, transport: httpx.BaseTransport | None = None) -> None:
        url = urlsplit(endpoint)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "::1"}
            or url.port is None
            or not 0 < url.port < 65536
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("Tokenizer endpoint must be a literal loopback HTTP origin")
        self.client = httpx.Client(
            base_url=endpoint,
            timeout=2.0,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )
        self.records: list[dict[str, Any]] = []

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        payload = {"content": text, "add_special": False, "parse_special": False}
        record: dict[str, Any] = {"request": payload}
        start = time.perf_counter()
        try:
            response = self.client.post("/tokenize", json=payload)
            record.update(status_code=response.status_code, response_text=response.text)
            response.raise_for_status()
            body = response.json()
            tokens = body.get("tokens") if isinstance(body, dict) else None
            if (
                not isinstance(tokens, list)
                or any(type(token) is not int or token < 0 for token in tokens)
                or (text and not tokens)
            ):
                raise ValueError("Tokenizer returned invalid token IDs")
            record["count"] = len(tokens)
            return len(tokens)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["elapsed_ms"] = (time.perf_counter() - start) * 1000
            self.records.append(record)

    def close(self) -> None:
        self.client.close()
