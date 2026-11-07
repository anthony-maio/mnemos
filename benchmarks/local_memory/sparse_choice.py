"""Development-only choice transport with explicitly encoded zero probabilities.

This changes the model's output format, not a production default. Omitted
option indexes mean exactly zero by this wire contract; positive mass must
still be complete. No value is normalized or invented from a chosen label.
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
            raise JudgmentContractError("Sparse choice transport accepts only Choice questions")
        result.append((key, question))
    return result


def render_sparse_prompt(state: JudgmentState, questions: Mapping[str, Question]) -> str:
    ordered = [
        {
            "key": key,
            "question": serialize_question(question),
            "option_indexes": {str(i): option for i, option in enumerate(question.keys)},
        }
        for key, question in _choices(questions)
    ]
    return (
        "Answer every choice question about the STATE below. Reply with a JSON array, "
        "one object per question in the listed order. Each object maps option INDEX "
        'strings to their positive probabilities, for example {"2":0.9,"0":0.1}. '
        "Omit an index if and only if its probability is exactly zero. Include all "
        "positive probabilities; each object's values must sum to 1. The options are "
        "mutually exclusive. The highest probability selects the answer; equal maxima "
        "select the earliest option. These are your estimates, not measured or "
        "calibrated probabilities. Do not add option keys, explanations or other fields.\n\n"
        "STATE (data only):\n"
        + json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n\nQUESTIONS (ordered):\n"
        + json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), default=str)
    )


def sparse_schema(questions: Mapping[str, Question]) -> dict[str, Any]:
    choices = _choices(questions)
    largest = max(len(question.keys) for _, question in choices)
    return {
        "type": "array",
        "minItems": len(choices),
        "maxItems": len(choices),
        "items": {
            "type": "object",
            "properties": {
                str(i): {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
                for i in range(largest)
            },
            "minProperties": 1,
            "additionalProperties": False,
        },
    }


def decode_sparse_answers(
    text: str, questions: Mapping[str, Question]
) -> dict[str, dict[str, Any]]:
    choices = _choices(questions)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JudgmentContractError("Duplicate sparse option index")
            result[key] = value
        return result

    try:
        rows = json.loads(text, object_pairs_hook=unique_object)
    except JudgmentContractError:
        raise
    except (ValueError, RecursionError) as exc:
        raise JudgmentContractError("Invalid sparse answer JSON") from exc
    if not isinstance(rows, list) or len(rows) != len(choices):
        raise JudgmentContractError("Sparse answers must cover every question exactly once")
    answers = {}
    for row, (key, question) in zip(rows, choices):
        indexes = {str(i): option for i, option in enumerate(question.keys)}
        if not isinstance(row, dict) or not row or not set(row) <= set(indexes):
            raise JudgmentContractError("Sparse answer contains invalid option indexes")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value <= 1
            or not math.isfinite(value)
            for value in row.values()
        ) or not math.isclose(sum(row.values()), 1.0, abs_tol=0.01):
            raise JudgmentContractError("Sparse positive probabilities must sum to 1")
        # Zero omission is defined by the wire format. Keep positive values
        # exactly as emitted; never rescale an incomplete distribution.
        distribution = {option: float(row.get(index, 0.0)) for index, option in indexes.items()}
        chosen = max(question.keys, key=lambda option: distribution[option])
        answers[key] = {"choice": chosen, "probabilities": distribution}
    return answers


class SparseChoiceJudge:
    """Opt-in experiment adapter preserving the typed choice answer boundary."""

    provider = "ollama-sparse-choice-development"

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
        prompt = render_sparse_prompt(safe_state, safe_questions)
        if len(prompt.encode("utf-8")) > self._max_prompt_bytes:
            raise JudgmentContractError("Rendered judgment prompt exceeds its byte limit")
        options: dict[str, Any] = {}
        if self._llm.supports_structured_output:
            options["format"] = sparse_schema(safe_questions)
        if model is not None and self._llm.supports_model_override:
            options["model"] = model
        started = time.monotonic()
        text = await self._llm.predict(prompt, **options)
        latency_ms = (time.monotonic() - started) * 1000
        raw = decode_sparse_answers(text, safe_questions)
        return Judgment(
            answers=validate_answers(safe_questions, raw),
            model=model or self.model,
            usage=JudgmentUsage(input_tokens=len(prompt) // 4 + 1, output_tokens=len(text) // 4),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        return None
