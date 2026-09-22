"""The hosted judge: request body, error mapping, retries, observer, firewall."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from mnemos.config import MemorySafetyConfig
from mnemos.judgment import (
    Choice,
    Criteria,
    JudgmentBlocked,
    JudgmentContractError,
    JudgmentRequestTooLarge,
    Noul,
    Score,
)
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.utils.reliability import (
    MnemosConfigurationError,
    MnemosTransientError,
    RetryPolicy,
)
from mnemos.utils.typesafe import TypeSafeJudge

QUESTIONS = {
    "is_urgent": Noul("Does this convey urgency?", Criteria("time-sensitive", "no urgency")),
    "department": Choice("Which team?", (("billing", None), ("technical", None))),
    "frustration": Score("How frustrated?", ("Calm", "Frustrated", "Very angry")),
}
ANSWERS = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.88, "technical": 0.12},
            "confidence": 0.76,
        },
        "frustration": {"type": "score", "score": 1.05, "confidence": 0.92},
    },
    "usage": {"input_tokens": 304, "output_tokens": 18},
}


def _judge(handler: Any, **kwargs: Any) -> tuple[TypeSafeJudge, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    judge = TypeSafeJudge(
        "key-123",
        firewall=MemoryWriteFirewall(MemorySafetyConfig()),
        retry_policy=RetryPolicy(
            max_attempts=3, initial_backoff_seconds=0.0, max_backoff_seconds=0.0
        ),
        response_observer=seen.append,
        **kwargs,
    )
    judge._transport = httpx.MockTransport(handler)
    return judge, seen


def test_a_request_carries_the_key_the_state_and_the_questions() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWERS)

    judge, seen = _judge(handler)
    judgment = asyncio.run(
        judge.ask({"document": "My payouts have failed for 3 days."}, QUESTIONS, purpose="triage")
    )

    (request,) = requests
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["Authorization"] == "Bearer key-123"
    body = json.loads(request.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == {"document": "My payouts have failed for 3 days."}
    assert set(body["questions"]) == set(QUESTIONS)
    assert body["questions"]["frustration"]["criteria"] == ["Calm", "Frustrated", "Very angry"]
    assert judgment.model == "jev-1.13.0"
    assert judgment.answers["department"].choice == "billing"  # type: ignore[union-attr]
    assert judgment.usage.input_tokens == 304
    assert seen[0]["purpose"] == "triage" and seen[0]["question_count"] == 3
    assert seen[0]["cost_usd"] == pytest.approx(304 * 0.042 / 1e6)


def test_an_explicit_model_overrides_the_default() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ANSWERS)

    judge, _ = _judge(handler)
    asyncio.run(judge.ask("s", QUESTIONS, purpose="t", model="jev-preview"))
    assert bodies[0]["model"] == "jev-preview"


def test_a_secret_never_reaches_the_wire() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWERS)

    judge, _ = _judge(handler)
    with pytest.raises(JudgmentBlocked):
        asyncio.run(judge.ask({"text": "password=hunter2"}, QUESTIONS, purpose="t"))
    assert requests == []

    asyncio.run(judge.ask({"text": "ring 555-867-5309"}, QUESTIONS, purpose="t"))
    assert b"555-867-5309" not in requests[0].content
    assert b"[REDACTED_" in requests[0].content


def test_auth_failures_are_configuration_errors_and_are_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "invalid key"})

    judge, _ = _judge(handler)
    with pytest.raises(MnemosConfigurationError):
        asyncio.run(judge.ask("s", QUESTIONS, purpose="t"))
    assert calls == 1


@pytest.mark.parametrize("status", [429, 529, 503])
def test_overload_is_retried_then_reported_transient(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": "busy"})

    judge, _ = _judge(handler)
    with pytest.raises(MnemosTransientError):
        asyncio.run(judge.ask("s", QUESTIONS, purpose="t"))
    assert calls == 3


def test_overload_that_recovers_returns_the_answer() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(529)
        return httpx.Response(200, json=ANSWERS)

    judge, _ = _judge(handler)
    assert asyncio.run(judge.ask("s", QUESTIONS, purpose="t")).answers["is_urgent"].probability == 0.95  # type: ignore[union-attr]


def test_a_malformed_request_is_a_contract_error_not_an_outage() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(422, json={"error": {"message": "questions.x: unknown type"}})

    judge, _ = _judge(handler)
    with pytest.raises(JudgmentContractError, match="unknown type"):
        asyncio.run(judge.ask("s", QUESTIONS, purpose="t"))
    assert calls == 1


def test_an_answer_off_contract_is_refused() -> None:
    bad = {**ANSWERS, "answers": {**ANSWERS["answers"], "department": {"choice": "refunds"}}}
    judge, _ = _judge(lambda request: httpx.Response(200, json=bad))
    with pytest.raises(JudgmentContractError):
        asyncio.run(judge.ask("s", QUESTIONS, purpose="t"))


def test_an_oversized_request_is_refused_before_sending() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ANSWERS)

    judge, _ = _judge(handler, max_request_tokens=50)
    with pytest.raises(JudgmentRequestTooLarge):
        asyncio.run(judge.ask({"text": "word " * 200}, QUESTIONS, purpose="t"))
    assert requests == []


def test_a_cloud_judge_cannot_be_built_without_a_firewall() -> None:
    with pytest.raises(TypeError):
        TypeSafeJudge("key", firewall=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TypeSafeJudge("", firewall=MemoryWriteFirewall(MemorySafetyConfig()))
