"""TypeSafe Jev as a judgment backend.

Jev is a hosted System One model: it answers typed questions with calibrated
probabilities and does not generate text. This client is a thin httpx wrapper
around ``POST /v1/systemone``. It refuses to exist without a write firewall,
so nothing leaves the process before the same safety policy that guards disk
and outbox has run over it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
import time
from typing import Any, Callable, Mapping

import httpx

from ..judgment import (
    Judgment,
    JudgmentContractError,
    JudgmentRequestTooLarge,
    JudgmentState,
    JudgmentUsage,
    Question,
    estimate_tokens,
    guard_state,
    serialize_question,
    validate_answers,
)
from ..memory_safety import MemoryWriteFirewall
from ..observability import log_event
from .reliability import MnemosError, RetryPolicy, call_with_async_retry

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-latest"
# Published on the run date; recorded with every measurement that uses it.
USD_PER_MILLION_INPUT_TOKENS = 0.042

ResponseObserver = Callable[[dict[str, Any]], None]


class TypeSafeJudge:
    provider = "typesafe"

    def __init__(
        self,
        api_key: str,
        *,
        firewall: MemoryWriteFirewall,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        response_observer: ResponseObserver | None = None,
        max_concurrency: int = 8,
        max_request_tokens: int = 60_000,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("TypeSafe API key is required")
        if not isinstance(firewall, MemoryWriteFirewall):
            raise TypeError("a cloud judge requires the memory write firewall")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._api_key = api_key.strip()
        self._firewall = firewall
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=4, initial_backoff_seconds=0.5, max_backoff_seconds=8.0
        )
        self.response_observer = response_observer
        self.max_request_tokens = max_request_tokens
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._trust_lock = threading.Lock()
        self._trust_context: ssl.SSLContext | None = None
        self._transport: httpx.AsyncBaseTransport | None = None

    def _tls_verification(self) -> ssl.SSLContext | bool:
        with self._trust_lock:
            if self._trust_context is None:
                self._trust_context = httpx.create_ssl_context(verify=True, trust_env=False)
            return self._trust_context

    def _client(self) -> httpx.AsyncClient:
        if self._transport is not None:
            return httpx.AsyncClient(transport=self._transport, timeout=self.timeout)
        return httpx.AsyncClient(timeout=self.timeout, verify=self._tls_verification())

    async def ask(
        self,
        state: JudgmentState,
        questions: Mapping[str, Question],
        *,
        purpose: str,
        model: str | None = None,
    ) -> Judgment:
        if not questions:
            raise JudgmentContractError("at least one question is required")
        safe_state, safe_questions = guard_state(state, questions, self._firewall)
        estimated = estimate_tokens(safe_state, safe_questions)
        if estimated > self.max_request_tokens:
            raise JudgmentRequestTooLarge(
                f"{purpose}: about {estimated} tokens exceeds the {self.max_request_tokens} ceiling"
            )
        body = {
            "model": model or self.model,
            "state": safe_state,
            "questions": {key: serialize_question(q) for key, q in safe_questions.items()},
        }
        started = time.monotonic()
        async with self._semaphore:
            try:
                data = await call_with_async_retry(
                    provider="typesafe",
                    operation=purpose,
                    fn=lambda: self._post(body, purpose),
                    policy=self.retry_policy,
                    should_retry=_retryable,
                )
            except MnemosError as exc:
                # The retry helper wraps everything it does not know. A
                # contract error is ours to keep: it must spend attempts, not
                # wait out a provider backoff.
                if isinstance(exc.__cause__, JudgmentContractError):
                    raise exc.__cause__ from None
                raise
        latency_ms = (time.monotonic() - started) * 1000.0
        if not isinstance(data, dict):
            raise JudgmentContractError("TypeSafe returned a non-object response")
        answers = validate_answers(safe_questions, data.get("answers", {}))
        usage_value = data.get("usage")
        usage_raw: Mapping[str, Any] = usage_value if isinstance(usage_value, Mapping) else {}
        usage = JudgmentUsage(
            input_tokens=_int(usage_raw.get("input_tokens")),
            output_tokens=_int(usage_raw.get("output_tokens")),
        )
        reported_model = data.get("model")
        judgment = Judgment(
            answers=answers,
            model=reported_model if isinstance(reported_model, str) else (model or self.model),
            usage=usage,
            latency_ms=latency_ms,
        )
        if self.response_observer is not None:
            try:
                self.response_observer(
                    {
                        "provider": self.provider,
                        "purpose": purpose,
                        "model": judgment.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "latency_ms": latency_ms,
                        "question_count": len(safe_questions),
                        "cost_usd": usage.input_tokens * USD_PER_MILLION_INPUT_TOKENS / 1e6,
                    }
                )
            except Exception:
                log_event(
                    "mnemos.provider_observer_failure",
                    level=logging.WARNING,
                    provider=self.provider,
                    operation=purpose,
                )
        return judgment

    async def _post(self, body: dict[str, Any], purpose: str) -> Any:
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/systemone",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"),
            )
            if response.status_code == 422:
                # Our request is malformed. Retrying the same body cannot help,
                # and a provider backoff would hide it for up to 15 minutes.
                detail = _error_detail(response)
                log_event(
                    "mnemos.provider_failure",
                    level=logging.WARNING,
                    provider=self.provider,
                    operation=purpose,
                    status_code=422,
                    detail=detail,
                )
                raise JudgmentContractError(f"TypeSafe rejected the request: {detail}")
            response.raise_for_status()
            return response.json()

    async def aclose(self) -> None:
        return None


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, JudgmentContractError):
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    return False


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, Mapping):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value[:500]
            if isinstance(value, Mapping) and isinstance(value.get("message"), str):
                return str(value["message"])[:500]
    return json.dumps(payload)[:500]


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
