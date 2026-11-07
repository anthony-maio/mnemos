"""Named probability maps preserve identity, mass and rejection boundaries."""

import asyncio
import pytest

from benchmarks.local_memory.named_choice import (
    NamedChoiceJudge,
    decode_named_answers,
    named_schema,
    render_named_prompt,
)
from mnemos.config import MemorySafetyConfig
from mnemos.judgment import Choice, JudgmentBlocked, JudgmentContractError, Noul, validate_answers
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.utils.ollama_judge import render_prompt
from tests.test_sparse_choice import Provider

QUESTIONS = {
    "second": Choice("Which source?", (("none", "Absent"), ("ev:z", "Z"), ("ev:a", "A"))),
    "first": Choice("Which period?", (("outside", "Outside"), ("inside", "Inside"))),
}


def test_named_keys_preserve_identity_without_question_or_option_order_dependency():
    wire = '{"first":{"inside":1},"second":{"ev:a":0.875,"none":0.125,"ev:z":0}}'
    raw = decode_named_answers(wire, QUESTIONS)
    assert raw == {
        "second": {"choice": "ev:a", "probabilities": {"none": 0.125, "ev:z": 0, "ev:a": 0.875}},
        "first": {"choice": "inside", "probabilities": {"outside": 0, "inside": 1}},
    }
    typed = validate_answers(QUESTIONS, raw)
    assert typed["second"].choice == "ev:a"


def test_explicit_and_omitted_zero_have_the_same_defined_meaning():
    omitted = '{"second":{"ev:a":1},"first":{"inside":1}}'
    explicit = '{"second":{"none":0,"ev:z":0,"ev:a":1},"first":{"outside":0,"inside":1}}'
    assert decode_named_answers(omitted, QUESTIONS) == decode_named_answers(explicit, QUESTIONS)


def test_mass_is_not_normalized_and_ties_use_offered_order():
    raw = decode_named_answers(
        '{"second":{"ev:a":0.499,"ev:z":0.499},"first":{"inside":1}}', QUESTIONS
    )
    assert raw["second"]["choice"] == "ev:z"
    assert raw["second"]["probabilities"] == {"none": 0, "ev:z": 0.499, "ev:a": 0.499}
    assert max(raw["second"]["probabilities"].values()) < 0.8


@pytest.mark.parametrize(
    "wire",
    [
        "[]",
        "{}",
        '{"second":{"none":1}}',
        '{"second":{"none":1},"first":{"inside":1},"foreign":{}}',
        '{"second":{},"first":{"inside":1}}',
        '{"second":{"0":1},"first":{"inside":1}}',
        '{"second":{"inside":1},"first":{"inside":1}}',
        '{"second":{"none":1},"first":{"ev:a":1}}',
        '{"second":{"none":true},"first":{"inside":1}}',
        '{"second":{"none":"1"},"first":{"inside":1}}',
        '{"second":{"none":null},"first":{"inside":1}}',
        '{"second":{"none":0},"first":{"inside":1}}',
        '{"second":{"none":-1},"first":{"inside":1}}',
        '{"second":{"none":1.1},"first":{"inside":1}}',
        '{"second":{"none":0.7},"first":{"inside":1}}',
        '{"second":{"none":0.6,"ev:a":0.6},"first":{"inside":1}}',
        '{"second":{"none":NaN},"first":{"inside":1}}',
        '{"second":{"none":Infinity},"first":{"inside":1}}',
        '{"second":{"none":1,"none":1},"first":{"inside":1}}',
        '{"second":{"none":1},"second":{"ev:a":1},"first":{"inside":1}}',
        '{"second":{"none":1},"first":{"inside":1}} trailing',
        '```json\n{"second":{"none":1},"first":{"inside":1}}\n```',
        '{"second":null,"first":{"inside":1}}',
        '{"second":{"none":' + "9" * 400 + '},"first":{"inside":1}}',
    ],
)
def test_contract_failure_never_becomes_an_empty_abstention(wire):
    with pytest.raises(JudgmentContractError):
        decode_named_answers(wire, QUESTIONS)


@pytest.mark.parametrize("questions", [{}, {"n": Noul("Answer?")}])
def test_unsupported_contract_rejects_before_inference(questions):
    provider = Provider("{}")
    judge = NamedChoiceJudge(provider, firewall=MemoryWriteFirewall(MemorySafetyConfig()))
    with pytest.raises(JudgmentContractError):
        asyncio.run(judge.ask("state", questions, purpose="test"))
    assert not provider.calls


def test_input_body_is_byte_identical_to_baseline_and_schema_uses_named_keys():
    state = {"query": "Which bucket now?", "facts": ["old", "new", "unrelated"]}
    baseline = render_prompt(state, QUESTIONS, require_choice_probabilities=True)
    named = render_named_prompt(state, QUESTIONS)
    marker = "STATE (data only):\n"
    assert baseline.split(marker, 1)[1] == named.split(marker, 1)[1]
    schema = named_schema(QUESTIONS)
    assert schema["required"] == list(QUESTIONS)
    assert not schema["additionalProperties"]
    for key, question in QUESTIONS.items():
        row = schema["properties"][key]
        assert list(row["properties"]) == list(question.keys)
        assert "required" not in row and not row["additionalProperties"]
        assert all(p["minimum"] == 0 and p["maximum"] == 1 for p in row["properties"].values())


def test_actual_judge_passes_schema_override_and_unchanged_probabilities():
    provider = Provider('{"second":{"ev:a":0.8,"ev:z":0.2},"first":{"inside":1}}')
    judge = NamedChoiceJudge(provider, firewall=MemoryWriteFirewall(MemorySafetyConfig()))
    result = asyncio.run(
        judge.ask({"query": "Where?"}, QUESTIONS, purpose="test", model="override")
    )
    assert result.model == "override"
    assert result.answers["second"].probabilities == {"none": 0, "ev:z": 0.2, "ev:a": 0.8}
    assert provider.calls[0]["format"] == named_schema(QUESTIONS)
    assert provider.calls[0]["model"] == "override"


def test_utf8_limit_and_firewall_refuse_without_a_provider_call():
    provider = Provider("{}")
    firewall = MemoryWriteFirewall(MemorySafetyConfig(secret_policy="block"))
    state = "\u00e9" * 100
    judge = NamedChoiceJudge(
        provider, firewall=firewall, max_prompt_bytes=len(render_named_prompt(state, QUESTIONS))
    )
    with pytest.raises(JudgmentContractError, match="byte limit"):
        asyncio.run(judge.ask(state, QUESTIONS, purpose="test"))
    with pytest.raises(JudgmentBlocked):
        asyncio.run(judge.ask("api_key=sk-" + "x" * 40, QUESTIONS, purpose="test"))
    assert not provider.calls
