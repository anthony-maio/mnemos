"""
tests/test_llm.py -- Tests for production LLM providers.
"""

from __future__ import annotations

import httpx
import pytest

from mnemos.utils import llm as llm_module


@pytest.mark.asyncio
async def test_openai_provider_retries_openrouter_with_fallback_key_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _DummyResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self._status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self._status_code < 400:
                return None
            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            response = httpx.Response(self._status_code, request=request)
            raise httpx.HTTPStatusError(
                f"status {self._status_code}",
                request=request,
                response=response,
            )

        def json(self) -> dict[str, object]:
            return self._payload

    class _DummyAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "_DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> _DummyResponse:
            _ = url, json, self.timeout
            calls.append(headers["Authorization"])
            if headers["Authorization"] == "Bearer stale-key":
                return _DummyResponse(401, {})
            return _DummyResponse(
                200,
                {"choices": [{"message": {"content": "ok"}}]},
            )

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _DummyAsyncClient)

    provider = llm_module.OpenAIProvider(
        api_key="stale-key",
        api_key_fallback="fresh-key",
        model="openrouter/auto",
        base_url="https://openrouter.ai/api/v1",
    )

    result = await provider.predict("memory test")

    assert result == "ok"
    assert calls == ["Bearer stale-key", "Bearer fresh-key"]
    assert provider.api_key == "fresh-key"
