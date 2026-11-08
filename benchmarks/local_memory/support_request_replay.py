"""Replay frozen typed support requests without labels or mutable stores."""
from __future__ import annotations

import asyncio
import math
from dataclasses import asdict
import json
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from mnemos.judgment import PREAMBLE, Choice, guard_state, serialize_question
from mnemos.memory_safety import source_firewall
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.ollama_judge import OllamaJudge, render_prompt
from mnemos.utils.reliability import RetryPolicy

MODEL = "qwen3.5:9b-q4_K_M"
SAMPLING = dict(seed=20261106, top_k=0, top_p=1, min_p=0,
                repeat_penalty=1, presence_penalty=0, frequency_penalty=0)


def direct_recorded_client(output: Path, request_id):
    class DirectRecordedClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            async def request_hook(request):
                if request.url.scheme != "http" or request.url.host != "127.0.0.1":
                    raise ValueError("Replay transport requires direct loopback")
                with (output / "requests.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"request_id": request_id(), "url": str(request.url),
                        "body": request.content.decode("utf-8")}, ensure_ascii=True) + "\n")

            async def response_hook(response):
                body = await response.aread()
                with (output / "responses.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"request_id": request_id(), "status": response.status_code,
                        "body": body.decode("utf-8")}, ensure_ascii=True) + "\n")

            kwargs.update(trust_env=False, event_hooks={"request": [request_hook], "response": [response_hook]})
            super().__init__(**kwargs)
    return DirectRecordedClient


def grade_replays(observations: list[dict], labels: list[dict]) -> dict:
    by_id = {r["request_id"]: r for r in observations}
    if len(by_id) != len(observations) or set(by_id) - {r["request_id"] for r in labels}:
        raise ValueError("Unexpected or duplicate observation")
    summaries = {}
    for arm in sorted({r["strategy"] for r in labels}):
        counts = dict(questions=0, answerable=0, no_answer=0, expected_facts=0,
                      retained=0, selected=0, correct_abstentions=0, answerable_hits=0,
                      clean_hits=0, unavailable=0, unattempted=0)
        durations = []
        for label in (r for r in labels if r["strategy"] == arm):
            counts["questions"] += 1
            expected = set(label["expected_fact_ids"])
            counts["answerable" if expected else "no_answer"] += 1
            counts["expected_facts"] += len(expected)
            row = by_id.get(label["request_id"])
            if row is None:
                counts["unattempted"] += 1
                continue
            durations.append(row["milliseconds"])
            if row["status"] != "assessed":
                counts["unavailable"] += 1
                continue
            answers = row["judgment"]["answers"]
            if set(answers) != {f"support:{i}" for i in range(len(label["candidate_ids"]))}:
                raise ValueError("Missing candidate decision")
            selected = set()
            for i, fact_id in enumerate(label["candidate_ids"]):
                answer = answers[f"support:{i}"]
                if answer["choice"].startswith("evidence:") and answer["probabilities"][answer["choice"]] >= 0.8:
                    selected.add(fact_id)
            counts["selected"] += len(selected)
            counts["retained"] += len(selected & expected)
            counts["answerable_hits"] += bool(expected and selected & expected)
            counts["clean_hits"] += bool(expected and selected == expected)
            counts["correct_abstentions"] += not expected and not selected
        def ratio(n, d):
            return n / d if d else None
        summaries[arm] = dict(**counts,
            precision=ratio(counts["retained"], counts["selected"]),
            retention=ratio(counts["retained"], counts["expected_facts"]),
            recall=ratio(counts["answerable_hits"], counts["answerable"]),
            abstention=ratio(counts["correct_abstentions"], counts["no_answer"]),
            support_p95_ms=sorted(durations)[math.ceil(.95*len(durations))-1] if durations else None)
    return {"stage": "known_development_component", "fresh_acceptance": False,
            "critical_retention": None, "arms": summaries}


def restore_questions(payload):
    result = {}
    for key, value in payload.items():
        if value.get("type") != "choice" or not isinstance(value.get("instructions"), str):
            raise ValueError("Only the frozen choice protocol is supported")
        instructions = value["instructions"]
        if not instructions.startswith(PREAMBLE):
            raise ValueError("Missing frozen safety preamble")
        question = Choice(instructions[len(PREAMBLE):], tuple(value["criteria"].items()))
        if serialize_question(question) != value:
            raise ValueError("Question round trip changed the frozen request")
        result[key] = question
    return result


async def run_replays(inputs: list[dict], endpoint: str, output: Path):
    from urllib.parse import urlparse
    if urlparse(endpoint).hostname != "127.0.0.1":
        raise ValueError("An isolated loopback endpoint is required")
    output.mkdir(exist_ok=False)
    seen = set()
    prepared = []
    for row in inputs:
        if set(row) != {"request_id", "purpose", "state", "questions"}:
            raise ValueError("Replay inputs cannot contain labels or grading fields")
        if row["request_id"] in seen:
            raise ValueError("Duplicate frozen request")
        seen.add(row["request_id"])
        questions = restore_questions(row["questions"])
        safe_state, safe_questions = guard_state(row["state"], questions, source_firewall())
        if safe_state != row["state"] or {
            k: serialize_question(v) for k, v in safe_questions.items()
        } != row["questions"]:
            raise ValueError("Firewall changed the frozen replay request")
        prompt = render_prompt(row["state"], questions, require_choice_probabilities=True)
        if len(prompt.encode("utf-8")) > 5120:
            raise ValueError("Frozen prompt exceeds unchanged byte bound")
        prepared.append((row, questions))

    active_id = "unrelated-warmup"

    provider = OllamaProvider(MODEL, endpoint, timeout=15, temperature=0, max_tokens=1024,
                              think=False, sampling_options=SAMPLING)
    judge = OllamaJudge(provider, firewall=source_firewall(), require_choice_probabilities=True,
                        max_prompt_bytes=5120)
    completed = []
    # The provider's retry policy is selected inside predict. Bind one attempt
    # in this isolated replay process; do not change production defaults.
    client = direct_recorded_client(output, lambda: active_id)
    with (
        patch("mnemos.utils.llm.RetryPolicy", return_value=RetryPolicy(max_attempts=1)),
        patch("mnemos.utils.llm.httpx.AsyncClient", client),
    ):
        async with asyncio.timeout(15):
            warmup = await judge.ask(
                {"source": "Ada owns the demonstration service."},
                {"owner": Choice("Who owns the demonstration service?", (
                    ("ada", "Ada"), ("unknown", "The source does not say"),
                ))}, purpose="unrelated-warmup",
            )
        (output / "warmup.json").write_text(json.dumps(asdict(warmup)), encoding="utf-8")
        for row, questions in prepared:
            active_id = row["request_id"]
            started = time.perf_counter()
            result = {"request_id": row["request_id"], "status": "unavailable"}
            try:
                async with asyncio.timeout(15):
                    judgment = await judge.ask(row["state"], questions, purpose=row["purpose"])
                result.update(status="assessed", judgment=asdict(judgment))
            except Exception as exc:
                result["error_type"] = type(exc).__name__
            result["milliseconds"] = (time.perf_counter() - started) * 1000
            with (output / "observations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=True) + "\n")
            completed.append(result)
            if result["status"] != "assessed":
                break
    return {"attempted": len(completed), "total": len(inputs),
            "completed": sum(r["status"] == "assessed" for r in completed)}
