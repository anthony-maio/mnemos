"""
tests/test_reliability.py — Retry and error taxonomy behavior.
"""

from __future__ import annotations

import httpx
import pytest

from mnemos.utils.reliability import (
    MnemosConfigurationError,
    RetryPolicy,
    call_with_async_retry,
    call_with_retry,
)


def test_call_with_retry_retries_transient_then_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary timeout")
        return "ok"

    result = call_with_retry(
        provider="sqlite",
        operation="store",
        fn=flaky,
        policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0, jitter_ratio=0.0),
    )

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_call_with_async_retry_maps_auth_error_to_configuration_error() -> None:
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(401, request=request)
    http_error = httpx.HTTPStatusError("unauthorized", request=request, response=response)

    async def fail() -> str:
        raise http_error

    with pytest.raises(MnemosConfigurationError):
        await call_with_async_retry(
            provider="openclaw",
            operation="predict",
            fn=fail,
            policy=RetryPolicy(max_attempts=1),
        )
