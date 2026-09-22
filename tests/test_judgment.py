"""Typed judgments: wire format, answer validation, scripting, and the firewall."""

from __future__ import annotations

import asyncio

import pytest

from mnemos.config import MemorySafetyConfig
from mnemos.judgment import (
    PREAMBLE,
    Choice,
    ChoiceAnswer,
    Criteria,
    JudgmentBlocked,
    JudgmentContractError,
    MockJudge,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    UnavailableJudge,
    choice_confidence,
    guard_state,
    serialize_question,
    validate_answers,
)
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.utils.reliability import MnemosConfigurationError

QUESTIONS = {
    "is_urgent": Noul(
        "Does this convey urgency?",
        Criteria(true="Explicitly time-sensitive", false="No urgency expressed"),
    ),
    "department": Choice(
        "Which team should handle this?",
        (
            ("billing", "Payments, invoicing, refunds"),
            ("technical", "Bugs, outages, integrations"),
            ("sales", "Pricing, upgrades, new accounts"),
        ),
    ),
    "frustration": Score("How frustrated is the customer?", ("Calm", "Frustrated", "Very angry")),
}

# The documented response for the questions above.
DOCUMENTED_ANSWERS = {
    "is_urgent": {"type": "noul", "noul": 0.95},
    "department": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
        "confidence": 0.81,
    },
    "frustration": {
        "type": "score",
        "score": 1.05,
        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
        "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
        "confidence": 0.92,
    },
}


def test_questions_serialize_in_the_documented_wire_format() -> None:
    noul = serialize_question(QUESTIONS["is_urgent"])
    choice = serialize_question(QUESTIONS["department"])
    score = serialize_question(QUESTIONS["frustration"])

    assert noul == {
        "type": "noul",
        "instructions": PREAMBLE + "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    }
    assert choice["type"] == "choice"
    assert choice["criteria"] == {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts",
    }
    assert score == {
        "type": "score",
        "instructions": PREAMBLE + "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    }


def test_documented_answers_validate_into_typed_answers() -> None:
    answers = validate_answers(QUESTIONS, DOCUMENTED_ANSWERS)

    assert answers["is_urgent"] == NoulAnswer(probability=0.95)
    assert answers["department"] == ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.88, "technical": 0.12, "sales": 0.0},
        confidence=0.81,
    )
    frustration = answers["frustration"]
    assert isinstance(frustration, ScoreAnswer)
    assert frustration.value == 1.05 and frustration.confidence == 0.92
    assert frustration.scaled(-1.0, 1.0, 3) == pytest.approx(0.05)


def test_confidence_matches_the_vendor_formula() -> None:
    assert choice_confidence({"billing": 0.88, "technical": 0.12, "sales": 0.0}) == pytest.approx(
        0.82
    )
    assert choice_confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert choice_confidence({"a": 0.5, "b": 0.5}) == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        {**DOCUMENTED_ANSWERS, "department": {"choice": "refunds"}},
        {**DOCUMENTED_ANSWERS, "is_urgent": {"noul": 1.5}},
        {**DOCUMENTED_ANSWERS, "frustration": {"score": 7}},
        {key: value for key, value in DOCUMENTED_ANSWERS.items() if key != "frustration"},
        {**DOCUMENTED_ANSWERS, "is_urgent": {"noul": True}},
    ],
    ids=["unoffered-choice", "probability-above-one", "score-past-levels", "missing", "bool"],
)
def test_off_contract_answers_are_refused(raw: dict[str, object]) -> None:
    with pytest.raises(JudgmentContractError):
        validate_answers(QUESTIONS, raw)


def test_question_shapes_are_bounded() -> None:
    with pytest.raises(ValueError):
        Choice("one option", (("only", None),))
    with pytest.raises(ValueError):
        Choice("dup", (("a", None), ("a", None)))
    with pytest.raises(ValueError):
        Score("one level", ("only",))


def test_mock_judge_scripts_by_key_and_pattern_with_conservative_defaults() -> None:
    judge = MockJudge({"department": "technical", "kind:*": "sales", "is_urgent": 0.7})
    questions = {**QUESTIONS, "kind:r0": QUESTIONS["department"]}

    judgment = asyncio.run(judge.ask({"text": "x"}, questions, purpose="test"))

    assert judgment.answers["department"].choice == "technical"  # type: ignore[union-attr]
    assert judgment.answers["kind:r0"].choice == "sales"  # type: ignore[union-attr]
    assert judgment.answers["is_urgent"] == NoulAnswer(probability=0.7)
    # Unscripted Score: the middle level.
    assert judgment.answers["frustration"].value == 1.0  # type: ignore[union-attr]
    assert judge.calls[0][2] == "test"

    unscripted = asyncio.run(MockJudge().ask("s", QUESTIONS, purpose="t")).answers
    assert unscripted["is_urgent"] == NoulAnswer(probability=0.0)
    assert unscripted["department"].choice == "billing"  # type: ignore[union-attr]
    with pytest.raises(JudgmentContractError):
        asyncio.run(MockJudge({"department": "refunds"}).ask("s", QUESTIONS, purpose="t"))


def test_unavailable_judge_raises_a_configuration_error() -> None:
    with pytest.raises(MnemosConfigurationError, match="no judgment provider"):
        asyncio.run(UnavailableJudge().ask("s", QUESTIONS, purpose="t"))


def test_guard_state_blocks_secrets_and_sends_redactions() -> None:
    firewall = MemoryWriteFirewall(MemorySafetyConfig())
    state = {"units": [{"id": "u0", "text": "Call me at 555-867-5309 about the deploy."}]}
    guarded_state, guarded_questions = guard_state(state, QUESTIONS, firewall)

    assert "[REDACTED_" in guarded_state["units"][0]["text"]  # type: ignore[index]
    assert "555-867-5309" not in guarded_state["units"][0]["text"]  # type: ignore[index]
    assert set(guarded_questions) == set(QUESTIONS)

    with pytest.raises(JudgmentBlocked):
        guard_state({"text": "api_key=supersecretvalue"}, QUESTIONS, firewall)
    with pytest.raises(JudgmentBlocked):
        guard_state(
            "fine",
            {"q": Choice("pick", (("a", "token: sk-abcdefghijklmnopqrstuvwxyz"), ("b", None)))},
            firewall,
        )
