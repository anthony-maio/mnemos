"""Sparse probabilities preserve mass and option identity at the judge boundary."""

import asyncio
import json

import pytest

from benchmarks.local_memory.sparse_choice import (
    SparseChoiceJudge,
    decode_sparse_answers,
    render_sparse_prompt,
    sparse_schema,
)
from mnemos.config import MemorySafetyConfig
from mnemos.judgment import (
    Choice,
    ChoiceAnswer,
    JudgmentBlocked,
    JudgmentContractError,
    Noul,
    serialize_question,
    validate_answers,
)
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.utils.llm import LLMProvider

QUESTIONS = {
    "second": Choice(
        "Which source supports the answer?", (("none", "No support"), ("ev:z", "Z"), ("ev:a", "A"))
    ),
    "first": Choice("Does this date apply?", (("outside", "Outside"), ("inside", "Inside"))),
}


class Provider(LLMProvider):
    supports_structured_output = True
    supports_model_override = True
    model = "test-model"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def predict(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply

    async def classify(self, prompt, labels):
        raise AssertionError("Sparse choice does not use a separate classification call")


def test_explicit_zero_omission_preserves_distribution_and_order():
    raw = decode_sparse_answers('[{"2":0.875,"0":0.125},{"1":1}]', QUESTIONS)
    expected = {
        "second": {"choice": "ev:a", "probabilities": {"none": 0.125, "ev:z": 0, "ev:a": 0.875}},
        "first": {"choice": "inside", "probabilities": {"outside": 0, "inside": 1}},
    }
    assert raw == expected
    assert validate_answers(QUESTIONS, raw) == validate_answers(QUESTIONS, expected)


def test_small_sum_tolerance_does_not_rescale_any_probability():
    raw = decode_sparse_answers('[{"0":0.2,"2":0.799},{"1":1}]', QUESTIONS)
    assert raw["second"]["probabilities"] == {"none": 0.2, "ev:z": 0, "ev:a": 0.799}


def test_tie_uses_offered_order_and_cannot_clear_point_eight_threshold():
    raw = decode_sparse_answers('[{"2":0.5,"1":0.5},{"1":1}]', QUESTIONS)
    assert raw["second"]["choice"] == "ev:z"
    assert max(raw["second"]["probabilities"].values()) < 0.8


@pytest.mark.parametrize(
    "wire",
    [
        "{}",
        "[]",
        "[{}]",
        "[{},{}]",
        '[{"0":1}]',
        '[{"0":1},{"0":1},{"0":1}]',
        '[{"3":1},{"0":1}]',
        '[{"0":1},{"2":1}]',
        '[{"01":1},{"0":1}]',
        '[{"-1":1},{"0":1}]',
        '[{"+1":1},{"0":1}]',
        '[{"none":1},{"0":1}]',
        '[{"0":true},{"0":1}]',
        '[{"0":"1"},{"0":1}]',
        '[{"0":null},{"0":1}]',
        '[{"0":0},{"0":1}]',
        '[{"0":-1},{"0":1}]',
        '[{"0":1.1},{"0":1}]',
        '[{"0":0.7},{"0":1}]',
        '[{"0":0.6,"1":0.6},{"0":1}]',
        '[{"0":NaN},{"0":1}]',
        '[{"0":Infinity},{"0":1}]',
        '[{"0":1,"0":1},{"0":1}]',
        '[{"0":1},{"0":1}] trailing',
        '```json\n[{"0":1},{"0":1}]\n```',
        '[null,{"0":1}]',
        '[{"0":' + "9" * 400 + '},{"0":1}]',
    ],
)
def test_malformed_or_incomplete_probability_reply_is_not_an_abstention(wire):
    with pytest.raises(JudgmentContractError):
        decode_sparse_answers(wire, QUESTIONS)


@pytest.mark.parametrize("questions", [{}, {"n": Noul("Answer?")}])
def test_unsupported_question_contract_is_rejected_before_inference(questions):
    provider = Provider("[]")
    judge = SparseChoiceJudge(provider, firewall=MemoryWriteFirewall(MemorySafetyConfig()))
    with pytest.raises(JudgmentContractError):
        asyncio.run(judge.ask("state", questions, purpose="test"))
    assert not provider.calls


def test_rendering_keeps_full_state_and_criteria_with_explicit_index_map():
    state = {"query": "Which bucket now?", "facts": ["old", "new", "unrelated"]}
    prompt = render_sparse_prompt(state, QUESTIONS)
    left, right = prompt.split("STATE (data only):\n", 1)[1].split("\n\nQUESTIONS (ordered):\n")
    assert json.loads(left) == state
    rows = json.loads(right)
    assert [row["key"] for row in rows] == list(QUESTIONS)
    for row, question in zip(rows, QUESTIONS.values()):
        assert row["question"] == serialize_question(question)
        assert row["option_indexes"] == {str(i): option for i, option in enumerate(question.keys)}


def test_actual_typed_judge_forwards_schema_and_preserves_supported_probability():
    provider = Provider('[{"2":0.8,"1":0.2},{"1":1}]')
    judge = SparseChoiceJudge(provider, firewall=MemoryWriteFirewall(MemorySafetyConfig()))
    result = asyncio.run(
        judge.ask({"query": "Where?"}, QUESTIONS, purpose="test", model="override")
    )
    assert result.model == "override"
    assert isinstance(result.answers["second"], ChoiceAnswer)
    assert result.answers["second"].choice == "ev:a"
    assert result.answers["second"].probabilities == {"none": 0, "ev:z": 0.2, "ev:a": 0.8}
    assert provider.calls[0]["format"] == sparse_schema(QUESTIONS)
    assert provider.calls[0]["model"] == "override"


def test_utf8_byte_limit_refuses_full_request_without_truncation_or_model_call():
    provider = Provider("[]")
    state = "\u00e9" * 100
    # The actual rendered byte limit, not its smaller character count.
    limit = len(render_sparse_prompt(state, QUESTIONS))
    judge = SparseChoiceJudge(
        provider, firewall=MemoryWriteFirewall(MemorySafetyConfig()), max_prompt_bytes=limit
    )
    with pytest.raises(JudgmentContractError, match="byte limit"):
        asyncio.run(judge.ask(state, QUESTIONS, purpose="test"))
    assert not provider.calls


def test_firewall_blocks_source_before_any_model_call():
    provider = Provider("[]")
    firewall = MemoryWriteFirewall(MemorySafetyConfig(secret_policy="block"))
    judge = SparseChoiceJudge(provider, firewall=firewall)
    with pytest.raises(JudgmentBlocked):
        asyncio.run(judge.ask("api_key=sk-" + "x" * 40, QUESTIONS, purpose="test"))
    assert not provider.calls
