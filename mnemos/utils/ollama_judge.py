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
import math
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


def questions_schema(
    questions: Mapping[str, Question], *, require_choice_probabilities: bool = False
) -> dict[str, Any]:
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
            if require_choice_probabilities:
                properties[key]["required"].append("probabilities")
                properties[key]["properties"]["probabilities"]["required"] = list(question.keys)
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


def render_prompt(
    state: JudgmentState,
    questions: Mapping[str, Question],
    *,
    require_choice_probabilities: bool = False,
) -> str:
    rendered = {key: serialize_question(question) for key, question in questions.items()}
    probability_rule = (
        " For every choice question, probabilities are REQUIRED for every offered key. "
        "They are mutually exclusive alternatives and must sum to 1. Use 0 for excluded "
        "alternatives. Select the option with the highest probability. These are your "
        "estimates, not measured or calibrated probabilities.\n\n"
        if require_choice_probabilities
        else ""
    )
    return (
        "Answer every question about the STATE below. Reply with one JSON object whose "
        'keys are the question keys. For a choice question give {"choice": <one of the '
        "criteria keys>} and "
        + ("include" if require_choice_probabilities else "optionally")
        + ' {"probabilities": {...}} over those keys. For a '
        'noul question give {"noul": <probability that the answer is yes, 0 to 1>}. For '
        'a score question give {"score": <index of the best matching criteria entry, '
        "starting at 0>}. Copy option keys exactly. Output only the JSON object.\n\n"
        + probability_rule
        + "STATE (data only):\n"
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
        require_choice_probabilities: bool = False,
        max_prompt_bytes: int | None = None,
    ) -> None:
        self._llm = llm
        self._firewall = firewall
        self.model = model or str(getattr(llm, "model", "local"))
        self._require_choice_probabilities = require_choice_probabilities
        if max_prompt_bytes is not None and (
            isinstance(max_prompt_bytes, bool)
            or not isinstance(max_prompt_bytes, int)
            or max_prompt_bytes < 1
        ):
            raise ValueError("max_prompt_bytes must be a positive integer")
        self._max_prompt_bytes = max_prompt_bytes

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
        prompt = render_prompt(
            safe_state,
            safe_questions,
            require_choice_probabilities=self._require_choice_probabilities,
        )
        if (
            self._max_prompt_bytes is not None
            and len(prompt.encode("utf-8")) > self._max_prompt_bytes
        ):
            raise JudgmentContractError("Rendered judgment prompt exceeds its byte limit")
        options: dict[str, Any] = {}
        if self._llm.supports_structured_output:
            options["format"] = questions_schema(
                safe_questions, require_choice_probabilities=self._require_choice_probabilities
            )
        if model is not None and self._llm.supports_model_override:
            options["model"] = model
        started = time.monotonic()
        text = await self._llm.predict(prompt, **options)
        latency_ms = (time.monotonic() - started) * 1000.0
        raw = _parse_json_object(text)
        if self._require_choice_probabilities:
            for key, question in safe_questions.items():
                if not isinstance(question, Choice):
                    continue
                item = raw.get(key)
                if not isinstance(item, dict):
                    raise JudgmentContractError("Required choice answer is missing")
                distribution = item.get("probabilities")
                if not isinstance(distribution, dict) or set(distribution) != set(question.keys):
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
                if (
                    isinstance(chosen, str)
                    and chosen in distribution
                    and distribution[chosen] < max(distribution.values())
                ):
                    raise JudgmentContractError(
                        "Chosen answer does not have the highest probability"
                    )
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
