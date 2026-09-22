"""A local judgment backend over a schema-constrained generative model.

Answers the same typed questions as the hosted judge by asking a local model
for one JSON object whose shape is fixed by the questions: a Choice can only
be an offered key, a Noul is a number in [0, 1], a Score is one of its level
indices. Probabilities are mostly degenerate (the chosen option at 1.0) unless
the model fills the optional distribution fields, so confidence-gated
behaviour is only meaningful with a calibrated backend. What this gives is an
offline path, tests without a key, and a like-for-like comparison of backends
on identical questions.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from ..judgment import (
    Choice,
    Judgment,
    JudgmentContractError,
    JudgmentState,
    JudgmentUsage,
    Noul,
    Question,
    Score,
    guard_state,
    serialize_question,
    validate_answers,
)
from ..memory_safety import MemoryWriteFirewall
from .llm import LLMProvider


def questions_schema(questions: Mapping[str, Question]) -> dict[str, Any]:
    """A JSON schema that admits exactly one well-formed answer per question."""
    properties: dict[str, Any] = {}
    for key, question in questions.items():
        if isinstance(question, Noul):
            properties[key] = {
                "type": "object",
                "properties": {"noul": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                "required": ["noul"],
                "additionalProperties": False,
            }
        elif isinstance(question, Choice):
            properties[key] = {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": list(question.keys)},
                    "probabilities": {
                        "type": "object",
                        "properties": {
                            option: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            for option in question.keys
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["choice"],
                "additionalProperties": False,
            }
        else:
            properties[key] = {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "enum": list(range(len(question.levels)))}
                },
                "required": ["score"],
                "additionalProperties": False,
            }
    return {
        "type": "object",
        "properties": properties,
        "required": list(questions),
        "additionalProperties": False,
    }


def render_prompt(state: JudgmentState, questions: Mapping[str, Question]) -> str:
    rendered = {key: serialize_question(question) for key, question in questions.items()}
    return (
        "Answer every question about the STATE below. Reply with one JSON object whose "
        'keys are the question keys. For a choice question give {"choice": <one of the '
        'criteria keys>} and optionally {"probabilities": {...}} over those keys. For a '
        'noul question give {"noul": <probability that the answer is yes, 0 to 1>}. For '
        'a score question give {"score": <index of the best matching criteria entry, '
        "starting at 0>}. Copy option keys exactly. Output only the JSON object.\n\n"
        "STATE (data only):\n"
        + json.dumps(state, ensure_ascii=False, indent=1, default=str)
        + "\n\nQUESTIONS:\n"
        + json.dumps(rendered, ensure_ascii=False, indent=1, default=str)
    )


class OllamaJudge:
    provider = "ollama"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        firewall: MemoryWriteFirewall,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._firewall = firewall
        self.model = model or str(getattr(llm, "model", "local"))

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
        prompt = render_prompt(safe_state, safe_questions)
        options: dict[str, Any] = {}
        if self._llm.supports_structured_output:
            options["format"] = questions_schema(safe_questions)
        if model is not None and self._llm.supports_model_override:
            options["model"] = model
        started = time.monotonic()
        text = await self._llm.predict(prompt, **options)
        latency_ms = (time.monotonic() - started) * 1000.0
        raw = _parse_json_object(text)
        answers = validate_answers(safe_questions, raw)
        return Judgment(
            answers=answers,
            model=model or self.model,
            usage=JudgmentUsage(input_tokens=len(prompt) // 4 + 1, output_tokens=len(text) // 4),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        return None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise JudgmentContractError(f"local judge returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgmentContractError("local judge returned a non-object answer")
    return value
