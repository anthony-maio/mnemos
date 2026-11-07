"""Provenance and null-selection contracts independent of model quality."""

from dataclasses import asdict

import pytest

from benchmarks.local_memory.extractive_support import decode_source_span, span_matches_fact


def inputs():
    return dict(
        context="Date: today\nLogs last thirty days.",
        source_region=(12, 34),
        offsets=[(0, 0), (0, 4), (0, 0), (0, 11), (12, 16), (17, 21), (22, 28), (29, 34), (0, 0)],
        sequence_ids=[None, 0, None, 1, 1, 1, 1, 1, None],
        attention_mask=[1] * 8 + [0],
        start_logits=[0.0, 1000.0, 1000.0, 1000.0, -10.0, -10.0, 3.0, -10.0, 1000.0],
        end_logits=[0.0, 1000.0, 1000.0, 1000.0, -10.0, -10.0, -10.0, 4.0, 1000.0],
        cls_index=0,
    )


def test_exact_source_offsets_mask_question_prefix_special_and_padding():
    values = inputs()
    result = decode_source_span(**values)
    assert result.selected and result.margin == 7
    assert asdict(result.best_span) == dict(
        text="thirty days.", start=22, end=34, start_token=6, end_token=7, score=7.0
    )
    assert span_matches_fact(result, "The logs last thirty days.")
    assert not span_matches_fact(result, "The logs last seven days.")


@pytest.mark.parametrize("null_score", [7, 8])
def test_null_tie_and_higher_null_abstain(null_score):
    values = inputs()
    values["start_logits"][0] = null_score
    result = decode_source_span(**values)
    assert not result.selected
    assert not span_matches_fact(result, values["context"])
    assert result.best_span.text == "thirty days."


def test_maximum_length_and_no_reversed_span():
    values = inputs()
    assert not decode_source_span(**values, max_answer_tokens=1).selected
    values["start_logits"][6:8] = [-10, 10]
    values["end_logits"][6:8] = [10, -10]
    assert not decode_source_span(**values).selected


def test_masked_middle_cannot_be_crossed():
    values = inputs()
    values["attention_mask"][6] = 0
    values["start_logits"][5] = 10
    assert not decode_source_span(**values).selected


def test_span_ties_choose_earliest_then_shortest():
    values = inputs()
    values["start_logits"][4:8] = [2] * 4
    values["end_logits"][4:8] = [2] * 4
    assert decode_source_span(**values).best_span.text == "Logs"


def test_no_source_tokens_is_a_valid_null_result():
    values = inputs()
    values["source_region"] = (0, 1)
    result = decode_source_span(**values)
    assert result.best_span is None and result.margin is None and not result.selected


def test_unicode_offsets_are_characters_and_overlap_is_literal():
    values = dict(
        context="Caf\u00e9 \u6771",
        source_region=(0, 6),
        offsets=[(0, 0), (0, 4), (5, 6)],
        sequence_ids=[None, 1, 1],
        attention_mask=[1, 1, 1],
        start_logits=[0, 3, -5],
        end_logits=[0, 4, -5],
        cls_index=0,
    )
    result = decode_source_span(**values)
    assert result.best_span.text == "Caf\u00e9"
    assert span_matches_fact(result, "Caf\u00e9 is the title.")
    assert not span_matches_fact(result, "Cafe is the title.")


@pytest.mark.parametrize(
    "change",
    [
        {"start_logits": [0]},
        {"offsets": []},
        {"max_answer_tokens": 0},
        {"max_answer_tokens": True},
        {"cls_index": -1},
        {"cls_index": 1},
        {"null_margin": float("nan")},
        {"source_region": (0, 35)},
        {"source_region": (3, 3)},
        {"source_region": (-1, 3)},
    ],
)
def test_invalid_request_does_not_count_as_abstention(change):
    values = inputs()
    values.update(change)
    with pytest.raises(ValueError):
        decode_source_span(**values)


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("start_logits", 1, float("nan")),
        ("end_logits", 7, float("inf")),
        ("attention_mask", 4, 2),
        ("sequence_ids", 4, 9),
        ("offsets", 6, (20, 19)),
        ("offsets", 7, (29, 99)),
        ("offsets", 7, (12, 34)),
        ("offsets", 6, (22.0, 28)),
    ],
)
def test_malformed_tokens_raise(field, index, value):
    values = inputs()
    values[field][index] = value
    with pytest.raises(ValueError):
        decode_source_span(**values)


@pytest.mark.parametrize("prefix,suffix", [("\n", ""), (" \t", "\n"), ("", " ")])
def test_opt_in_clips_only_boundary_whitespace(prefix, suffix):
    context = prefix + "Lem" + suffix
    values = dict(
        context=context,
        source_region=(len(prefix), len(prefix) + 3),
        offsets=[(0, 0), (0, len(context))],
        sequence_ids=[None, 1],
        attention_mask=[1, 1],
        start_logits=[0, 5],
        end_logits=[0, 6],
        cls_index=0,
    )
    assert not decode_source_span(**values).selected
    result = decode_source_span(**values, clip_boundary_whitespace=True)
    assert result.selected and result.best_span.text == "Lem"
    assert result.best_span.start == len(prefix) and result.best_span.end == len(prefix) + 3
    assert span_matches_fact(result, "Lem owns the service.")


@pytest.mark.parametrize("prefix,suffix", [("x\n", ""), ("", " x"), (":", "")])
def test_clipping_never_admits_nonwhitespace_outside_source(prefix, suffix):
    context = prefix + "Lem" + suffix
    result = decode_source_span(
        context,
        (len(prefix), len(prefix) + 3),
        [(0, 0), (0, len(context))],
        [None, 1],
        [1, 1],
        [0, 5],
        [0, 6],
        cls_index=0,
        clip_boundary_whitespace=True,
    )
    assert not result.selected


def test_clipping_does_not_repair_malformed_offsets():
    values = inputs()
    values["offsets"][7] = (29, 99)
    with pytest.raises(ValueError):
        decode_source_span(**values, clip_boundary_whitespace=True)
