from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.compact_query_wire import TransportReply
from benchmarks.local_memory.condition_applicability import evaluate_conditions
from benchmarks.local_memory.condition_query_wire import (
    decode_condition_question,
    prepare_condition_question,
    render_condition_question,
)
from tests.test_condition_applicability import fixtures, SOURCE


def request(query="During maintenance active, who handles the task?"):
    source, _, _ = fixtures()
    return prepare_condition_question(source, SOURCE, query, "test-interpreter-v1", lambda: None)


def reply(req, state="positive", quote="maintenance active"):
    return TransportReply(
        req.request_id,
        req.binding_sha256,
        True,
        json.dumps({"p0": {"state": state, "quote": quote}}),
    )


@pytest.mark.parametrize(
    "state,expected",
    [("positive", "applies"), ("negative", "excluded"), ("not_established", "unresolved")],
)
def test_bound_interpretations_feed_three_state_logic(state, expected):
    # Authored interpretation tests the boundary, not natural-language accuracy.
    query = (
        "Outside maintenance active, who handles the task?"
        if state == "negative"
        else "During maintenance active, who handles the task?"
    )
    req = request(query)
    decoded = decode_condition_question(
        req,
        reply(req, state, None if state == "not_established" else "maintenance active"),
        lambda: None,
    )
    assert (
        evaluate_conditions(req.source, decoded, source_text=SOURCE, query_text=req.query)
        == expected
    )


def test_unstated_prerequisite_cannot_use_source_text_as_query_evidence():
    req = request("Who handles the task?")
    with pytest.raises(ValueError, match="absent"):
        decode_condition_question(req, reply(req), lambda: None)
    decoded = decode_condition_question(req, reply(req, "not_established", None), lambda: None)
    assert (
        evaluate_conditions(req.source, decoded, source_text=SOURCE, query_text=req.query)
        == "unresolved"
    )


@pytest.mark.parametrize(
    "state,quote",
    [
        ("positive", None),
        ("positive", ""),
        ("negative", "invented"),
        ("not_established", "maintenance active"),
        ("applies", None),
    ],
)
def test_invalid_or_missing_witness_is_not_an_applicability_decision(state, quote):
    req = request()
    with pytest.raises(ValueError):
        decode_condition_question(req, reply(req, state, quote), lambda: None)


def test_repeated_quote_needs_a_unique_span_not_first_occurrence_guessing():
    req = request("maintenance active or maintenance active: who handles it?")
    with pytest.raises(ValueError, match="ambiguous"):
        decode_condition_question(req, reply(req), lambda: None)


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"p1":{"state":"not_established","quote":null}}',
        '{"p0":{"state":"not_established","quote":null},"p0":{"state":"positive","quote":"maintenance active"}}',
        '{"p0":{"state":"not_established","quote":null,"confidence":1}}',
    ],
)
def test_missing_foreign_duplicate_and_extra_fields_are_rejected(text):
    req = request()
    with pytest.raises(ValueError):
        decode_condition_question(req, replace(reply(req), text=text), lambda: None)


def test_request_correspondence_and_revocation():
    req, other = request(), request()
    for wire in [reply(other), replace(reply(req), completed=False)]:
        with pytest.raises(ValueError):
            decode_condition_question(req, wire, lambda: None)
    changed = replace(req, query="changed")
    with pytest.raises(ValueError):
        render_condition_question(changed)
    with pytest.raises(ValueError):
        decode_condition_question(changed, reply(req), lambda: None)

    def revoked():
        raise PermissionError("source forgotten")

    with pytest.raises(PermissionError):
        decode_condition_question(req, reply(req), revoked)


def test_prompt_has_query_and_catalog_without_expected_judgments_or_source_instructions():
    req = request()
    prompt, schema = render_condition_question(req)
    data = json.loads(prompt.split("\nINPUT\n", 1)[1])
    assert data["question"] == req.query
    assert set(data) == {"question", "predicates"}
    assert data["predicates"]["p0"]["subject"] == "task"
    assert schema["required"] == ["p0"]


def test_incomplete_and_unconditional_sources_do_not_start_query_interpretation():
    source, _, _ = fixtures()
    for invalid in [replace(source, complete=False), replace(source, expression=None)]:
        with pytest.raises(ValueError):
            prepare_condition_question(invalid, SOURCE, "Who handles it?", "test-v1", lambda: None)
