"""Development-only named probability maps with an explicit zero convention.

The input state and criteria keep the baseline serialization. Only the output
contract changes: named questions map directly to named probabilities, omitted
options mean zero, and explicit zeros are also valid. Full mass is still required.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Mapping

from mnemos.judgment import (
    Choice,
    Judgment,
    JudgmentContractError,
    JudgmentState,
    JudgmentUsage,
    Question,
    guard_state,
    serialize_question,
    validate_answers,
)
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.utils.llm import LLMProvider


def _choices(questions: Mapping[str, Question]) -> list[tuple[str, Choice]]:
    if not questions:
        raise JudgmentContractError("At least one choice question is required")
    result = []
    for key, question in questions.items():
        if not isinstance(question, Choice):
            raise JudgmentContractError("Named choice transport accepts only Choice questions")
        result.append((key, question))
    return result


def render_named_prompt(state: JudgmentState, questions: Mapping[str, Question]) -> str:
    rendered = {key: serialize_question(question) for key, question in _choices(questions)}
    return (
        "Answer every choice question about the STATE below. Reply with one JSON object "
        "whose keys are the question keys. Each value is an object mapping offered option "
        'keys directly to probabilities, for example {"question_key":{"option_a":0.9,'
        '"option_b":0.1}}. Copy question and option keys exactly. Include every question. '
        "Omitted options mean exactly zero; explicit zero values are also valid. Include "
        "all positive probabilities. The options are mutually exclusive and each question's "
        "probabilities must sum to 1. The highest probability selects the answer; equal "
        "maxima select the earliest offered option. These are your estimates, not measured "
        "or calibrated probabilities. Output only the JSON object.\n\n"
        "STATE (data only):\n"
        + json.dumps(state, ensure_ascii=False, indent=1, default=str)
        + "\n\nQUESTIONS:\n"
        + json.dumps(rendered, ensure_ascii=False, indent=1, default=str)
    )


def named_schema(questions: Mapping[str, Question]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            key: {
                "type": "object",
                "properties": {
                    option: {"type": "number", "minimum": 0, "maximum": 1}
                    for option in question.keys
                },
                "minProperties": 1,
                "additionalProperties": False,
            }
            for key, question in _choices(questions)
        },
        "required": list(questions),
        "additionalProperties": False,
    }


def decode_named_answers(text: str, questions: Mapping[str, Question]) -> dict[str, dict[str, Any]]:
    choices = _choices(questions)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JudgmentContractError("Duplicate named answer key")
            result[key] = value
        return result

    try:
        rows = json.loads(text, object_pairs_hook=unique_object)
    except JudgmentContractError:
        raise
    except (ValueError, RecursionError) as exc:
        raise JudgmentContractError("Invalid named answer JSON") from exc
    if not isinstance(rows, dict) or set(rows) != set(questions):
        raise JudgmentContractError("Named answers must cover every question exactly once")
    answers = {}
    for key, question in choices:
        row = rows[key]
        if not isinstance(row, dict) or not row or not set(row) <= set(question.keys):
            raise JudgmentContractError("Named answer contains invalid option keys")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            or not math.isfinite(value)
            for value in row.values()
        ) or not math.isclose(sum(row.values()), 1.0, abs_tol=0.01):
            raise JudgmentContractError("Named probabilities must sum to 1")
        # Missing keys encode zero under this new contract. Preserve all emitted
        # values exactly; never renormalize or infer mass from a selected label.
        distribution = {option: float(row.get(option, 0.0)) for option in question.keys}
        chosen = max(question.keys, key=lambda option: distribution[option])
        answers[key] = {"choice": chosen, "probabilities": distribution}
    return answers


class NamedChoiceJudge:
    """Opt-in experiment preserving the existing typed judgment boundary."""

    provider = "ollama-named-choice-development"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        firewall: MemoryWriteFirewall,
        max_prompt_bytes: int = 5120,
    ) -> None:
        if type(max_prompt_bytes) is not int or max_prompt_bytes < 1:
            raise ValueError("Prompt byte limit must be a positive integer")
        self._llm = llm
        self._firewall = firewall
        self._max_prompt_bytes = max_prompt_bytes
        self.model = str(getattr(llm, "model", "local"))

    async def ask(
        self,
        state: JudgmentState,
        questions: Mapping[str, Question],
        *,
        purpose: str,
        model: str | None = None,
    ) -> Judgment:
        _choices(questions)
        safe_state, safe_questions = guard_state(state, questions, self._firewall)
        prompt = render_named_prompt(safe_state, safe_questions)
        if len(prompt.encode("utf-8")) > self._max_prompt_bytes:
            raise JudgmentContractError("Rendered judgment prompt exceeds its byte limit")
        options: dict[str, Any] = {}
        if self._llm.supports_structured_output:
            options["format"] = named_schema(safe_questions)
        if model is not None and self._llm.supports_model_override:
            options["model"] = model
        started = time.monotonic()
        text = await self._llm.predict(prompt, **options)
        latency_ms = (time.monotonic() - started) * 1000
        raw = decode_named_answers(text, safe_questions)
        return Judgment(
            answers=validate_answers(safe_questions, raw),
            model=model or self.model,
            usage=JudgmentUsage(input_tokens=len(prompt) // 4 + 1, output_tokens=len(text) // 4),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        return None
