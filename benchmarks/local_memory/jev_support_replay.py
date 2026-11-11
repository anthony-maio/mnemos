"""One explicitly selected hosted comparison of frozen support requests."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import httpx

from benchmarks.local_memory.support_request_replay import restore_questions
from mnemos.judgment import (
    Choice,
    JudgmentContractError,
    estimate_tokens,
    guard_state,
    serialize_question,
)
from mnemos.memory_safety import source_firewall
from mnemos.utils.reliability import RetryPolicy
from mnemos.utils.typesafe import TypeSafeJudge

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class RecordedJevJudge(TypeSafeJudge):
    """Record bodies, never authentication headers; disallow alternate hosts."""

    def __init__(self, api_key: str, output: Path, *, transport=None):
        super().__init__(
            api_key,
            firewall=source_firewall(),
            timeout=15,
            retry_policy=RetryPolicy(max_attempts=1),
            max_concurrency=1,
        )
        self.output = output
        self.request_id = "unrelated-warmup"
        self.transport = transport

    async def ask(self, *args, **kwargs):
        # HTTPX timeouts limit socket inactivity, not total operation duration.
        async with asyncio.timeout(15):
            return await super().ask(*args, **kwargs)

    async def _post(self, body, purpose):
        data = await super()._post(body, purpose)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("model"), str)
            or not data["model"]
        ):
            raise JudgmentContractError("Hosted model identity is missing")
        if data["model"] in {"jev-latest", "jev-preview"}:
            raise JudgmentContractError("Hosted model identity is an unresolved alias")
        answers = data.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(body["questions"]):
            raise JudgmentContractError("Required choice answers are incomplete")
        for key, question in body["questions"].items():
            item = answers[key]
            distribution = item.get("probabilities") if isinstance(item, dict) else None
            if not isinstance(distribution, dict) or set(distribution) != set(question["criteria"]):
                raise JudgmentContractError("Required choice probabilities are incomplete")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in distribution.values()
            ) or not math.isclose(sum(distribution.values()), 1.0, abs_tol=0.01):
                raise JudgmentContractError("Required choice probabilities must sum to 1")
            chosen = item.get("choice")
            if chosen not in distribution or distribution[chosen] < max(distribution.values()):
                raise JudgmentContractError("Chosen answer does not have the highest probability")
        usage = data.get("usage")
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        ):
            raise JudgmentContractError("Hosted token usage is missing")
        return data

    def _append(self, filename, row):
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    check(key)
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                if self._api_key in value:
                    raise ValueError("Credential appeared in evidence body")
                # Wire bodies can hide reflected strings behind JSON escapes.
                if value.lstrip().startswith(("{", "[", '"')):
                    try:
                        decoded = json.loads(value)
                    except ValueError:
                        return
                    if decoded != value:
                        check(decoded)

        check(row)
        encoded = json.dumps(row, ensure_ascii=True)
        if self._api_key in encoded:
            raise ValueError("Credential appeared in evidence body")
        with (self.output / filename).open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def _client(self):
        async def request_hook(request):
            if str(request.url) != ENDPOINT or request.method != "POST":
                raise ValueError("Only the explicit Jev HTTPS endpoint is permitted")
            self._append(
                "requests.jsonl",
                {
                    "request_id": self.request_id,
                    "url": str(request.url),
                    "body": request.content.decode("utf-8"),
                },
            )

        async def response_hook(response):
            body = await response.aread()
            self._append(
                "responses.jsonl",
                {
                    "request_id": self.request_id,
                    "status": response.status_code,
                    "body": body.decode("utf-8"),
                },
            )

        return httpx.AsyncClient(
            timeout=15,
            trust_env=False,
            follow_redirects=False,
            transport=self.transport,
            event_hooks={"request": [request_hook], "response": [response_hook]},
        )


def prepare_requests(inputs):
    if not 1 <= len(inputs) <= 44:
        raise ValueError("The bounded comparison permits at most 44 requests")
    seen = set()
    prepared = []
    total_tokens = 0
    for row in inputs:
        if set(row) != {"request_id", "purpose", "state", "questions"}:
            raise ValueError("Replay inputs cannot contain labels or grading fields")
        if row["request_id"] in seen:
            raise ValueError("Duplicate frozen request")
        seen.add(row["request_id"])
        questions = restore_questions(row["questions"])
        safe_state, safe_questions = guard_state(row["state"], questions, source_firewall())
        if (
            safe_state != row["state"]
            or {key: serialize_question(value) for key, value in safe_questions.items()}
            != row["questions"]
        ):
            raise ValueError("Firewall changed the frozen replay request")
        total_tokens += estimate_tokens(safe_state, safe_questions)
        prepared.append((row, questions))
    if total_tokens > 1_000_000:
        raise ValueError("Comparison exceeds the one-million estimated input token bound")
    return prepared


async def run_jev_replays(inputs, api_key: str, output: Path, *, transport=None):
    prepared = prepare_requests(inputs)
    # Exclusive output creation prevents accidental rerun/overwrite.
    output.mkdir(exist_ok=False)
    judge = RecordedJevJudge(api_key, output, transport=transport)
    completed = []
    async with asyncio.timeout(720):
        warmup = await judge.ask(
            {"source": "Ada owns the demonstration service."},
            {
                "owner": Choice(
                    "Who owns the demonstration service?",
                    (
                        ("ada", "Ada"),
                        ("unknown", "The source does not say"),
                    ),
                )
            },
            purpose="unrelated-warmup",
        )
        judge._append("warmup.json", asdict(warmup))
        # Bind subsequent calls to the model identity returned by the warmup.
        judge.model = warmup.model
        for row, questions in prepared:
            judge.request_id = row["request_id"]
            started = time.perf_counter()
            result = {"request_id": row["request_id"], "status": "unavailable"}
            try:
                judgment = await judge.ask(row["state"], questions, purpose=row["purpose"])
                if judgment.model != warmup.model:
                    raise ValueError("Hosted model identity changed during comparison")
                result.update(status="assessed", judgment=asdict(judgment))
            except Exception as exc:
                # Error text may contain provider details. Keep only its class.
                result["error_type"] = type(exc).__name__
            result["milliseconds"] = (time.perf_counter() - started) * 1000
            judge._append("observations.jsonl", result)
            completed.append(result)
            if result["status"] != "assessed":
                break
    return {
        "attempted": len(completed),
        "total": len(inputs),
        "completed": sum(row["status"] == "assessed" for row in completed),
        "observed_model": warmup.model,
        "model_identity_kind": (
            "vendor_alias"
            if warmup.model in {"jev-latest", "jev-preview"}
            else "vendor_version_string"
        ),
    }
