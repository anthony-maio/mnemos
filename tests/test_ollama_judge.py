"""The local judge answers the shared questions through structured output."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from mnemos.config import MemorySafetyConfig
from mnemos.judgment import Choice, Criteria, JudgmentBlocked, JudgmentContractError, Noul, Score
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.utils.llm import LLMProvider
from mnemos.utils.ollama_judge import OllamaJudge, questions_schema

QUESTIONS = {
    "novel": Noul("New information?", Criteria("yes", "no")),
    "kind:r0": Choice("Kind?", (("asserted_fact", None), ("question", None))),
    "arousal": Score("Urgency?", ("calm", "mild", "urgent")),
}


class _Structured(LLMProvider):
    supports_structured_output = True
    supports_model_override = True
    model = "qwen3.5:9b"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return {label: 0.0 for label in labels}


def test_the_schema_admits_only_offered_answers() -> None:
    schema = questions_schema(QUESTIONS)
    assert schema["required"] == ["novel", "kind:r0", "arousal"]
    assert schema["properties"]["kind:r0"]["properties"]["choice"]["enum"] == [
        "asserted_fact",
        "question",
    ]
    assert schema["properties"]["arousal"]["properties"]["score"]["enum"] == [0, 1, 2]
    assert schema["properties"]["novel"]["properties"]["noul"]["maximum"] == 1.0


def test_answers_come_back_typed_with_the_schema_and_model_forwarded() -> None:
    llm = _Structured(
        json.dumps(
            {"novel": {"noul": 0.8}, "kind:r0": {"choice": "question"}, "arousal": {"score": 2}}
        )
    )
    judge = OllamaJudge(llm, firewall=MemoryWriteFirewall(MemorySafetyConfig()))

    judgment = asyncio.run(judge.ask({"text": "Why?"}, QUESTIONS, purpose="t", model="other"))

    assert judgment.answers["novel"].probability == 0.8  # type: ignore[union-attr]
    assert judgment.answers["kind:r0"].choice == "question"  # type: ignore[union-attr]
    assert judgment.answers["arousal"].value == 2.0  # type: ignore[union-attr]
    assert judgment.model == "other"
    call = llm.calls[0]
    assert call["format"] == questions_schema(QUESTIONS)
    assert call["model"] == "other"
    assert '"text": "Why?"' in call["prompt"]


def test_off_contract_or_unparseable_replies_are_refused() -> None:
    firewall = MemoryWriteFirewall(MemorySafetyConfig())
    for reply in (
        "not json",
        json.dumps({"novel": {"noul": 0.5}}),
        json.dumps(
            {"novel": {"noul": 0.5}, "kind:r0": {"choice": "other"}, "arousal": {"score": 1}}
        ),
    ):
        with pytest.raises(JudgmentContractError):
            asyncio.run(
                OllamaJudge(_Structured(reply), firewall=firewall).ask("s", QUESTIONS, purpose="t")
            )


def test_the_firewall_runs_before_the_local_model_too() -> None:
    llm = _Structured("{}")
    judge = OllamaJudge(llm, firewall=MemoryWriteFirewall(MemorySafetyConfig()))
    with pytest.raises(JudgmentBlocked):
        asyncio.run(judge.ask({"text": "api_key=supersecretvalue"}, QUESTIONS, purpose="t"))
    assert llm.calls == []
