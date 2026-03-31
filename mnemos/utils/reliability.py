"""
mnemos/utils/reliability.py — Shared retry/backoff and error taxonomy.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")


class MnemosError(RuntimeError):
    """Base exception for runtime reliability failures."""


class MnemosConfigurationError(MnemosError):
    """Raised for invalid provider configuration/authentication issues."""


class MnemosTransientError(MnemosError):
    """Raised for transient errors that may succeed on retry."""


class MnemosExternalServiceError(MnemosError):
    """Raised for non-retryable external service failures."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy with exponential backoff."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1

    def backoff_seconds(self, attempt: int) -> float:
        if attempt <= 1:
            return 0.0
        base = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt - 2))
        capped = min(base, self.max_backoff_seconds)
        jitter = capped * self.jitter_ratio
        if jitter <= 0:
            return capped
        return capped + random.uniform(0.0, jitter)


def _retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def is_retryable_http_exception(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _retryable_http_status(exc.response.status_code)
    return False


def is_retryable_transient_exception(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = (
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "503",
        "504",
        "429",
    )
    return any(marker in text for marker in retry_markers)


def _wrap_provider_exception(provider: str, operation: str, exc: Exception) -> MnemosError:
    detail = f"{provider} {operation} failed: {exc}"
    if isinstance(exc, MnemosError):
        return exc

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return MnemosConfigurationError(detail)
        if _retryable_http_status(status):
            return MnemosTransientError(detail)
        return MnemosExternalServiceError(detail)

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return MnemosTransientError(detail)

    if is_retryable_transient_exception(exc):
        return MnemosTransientError(detail)

    return MnemosExternalServiceError(detail)


async def call_with_async_retry(
    *,
    provider: str,
    operation: str,
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    """Execute an async call with retries and standardized error mapping."""
    effective_policy = policy or RetryPolicy()
    retry_predicate = should_retry or is_retryable_http_exception

    last_exception: Exception | None = None
    for attempt in range(1, effective_policy.max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # pragma: no cover - exercised by integration paths
            last_exception = exc
            retryable = retry_predicate(exc)
            if not retryable or attempt >= effective_policy.max_attempts:
                raise _wrap_provider_exception(provider, operation, exc) from exc
            await asyncio.sleep(effective_policy.backoff_seconds(attempt + 1))

    assert last_exception is not None
    raise _wrap_provider_exception(provider, operation, last_exception)


def call_with_retry(
    *,
    provider: str,
    operation: str,
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    """Execute a synchronous call with retries and standardized error mapping."""
    effective_policy = policy or RetryPolicy()
    retry_predicate = should_retry or is_retryable_transient_exception

    last_exception: Exception | None = None
    for attempt in range(1, effective_policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - exercised by integration paths
            last_exception = exc
            retryable = retry_predicate(exc)
            if not retryable or attempt >= effective_policy.max_attempts:
                raise _wrap_provider_exception(provider, operation, exc) from exc
            time.sleep(effective_policy.backoff_seconds(attempt + 1))

    assert last_exception is not None
    raise _wrap_provider_exception(provider, operation, last_exception)
