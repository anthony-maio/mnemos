from copy import deepcopy
import json

import pytest


class Tokenizer:
    cls_token_id = 2_000_001
    sep_token_id = 2_000_002
    all_special_ids = [cls_token_id, sep_token_id]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class WordTokenizer(Tokenizer):
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(1, len(text.split()) + 1))


def request(count=1, sources=1):
    return dict(
        query="List the export format of every listed fixture.",
        reference_date="2026-11-11",
        candidates=[
            dict(
                id=f"fixture-{i}",
                version=1,
                content=f"Fixture {i} exports JSON.",
                assertion_status="asserted",
                sources=[
                    dict(
                        id=f"source-{i}-{j}",
                        source_event_id=f"event-{i}-{j}",
                        excerpt=f"Fixture {i} export format: JSON.",
                    )
                    for j in range(sources)
                ],
            )
            for i in range(count)
        ],
    )


def frame(value, tokenizer=None):
    try:
        from benchmarks.local_memory.local_support_v2_admission import frame_local_support
    except ModuleNotFoundError as error:
        if error.name != "benchmarks.local_memory.local_support_v2_admission":
            raise
        pytest.fail("The explicit candidate input-admission policy is missing")
    return frame_local_support(value, tokenizer or Tokenizer(), "a" * 64)


@pytest.mark.parametrize("count", [0, 1, 4, 5])
def test_complete_cohort_is_retained_with_native_boundaries(count):
    value = request(count)
    before = deepcopy(value)
    result = frame(value)
    assert json.loads(result.payload_json) == value == before
    assert result.target_ids == tuple(c["id"] for c in value["candidates"])
    assert result.input_ids[0] == Tokenizer.cls_token_id
    assert result.input_ids[-1] == Tokenizer.sep_token_id
    assert len(result.decision_positions) == count


def test_eight_owned_sources_retained_and_ninth_rejects_whole_cohort():
    value = request(1, 8)
    assert len(json.loads(frame(value).payload_json)["candidates"][0]["sources"]) == 8
    value["candidates"][0]["sources"].append(
        dict(id="ninth", source_event_id="event-9", excerpt="Another owned source.")
    )
    before = deepcopy(value)
    with pytest.raises(ValueError, match="eight-source"):
        frame(value)
    assert value == before


def test_token_bound_includes_native_boundaries_and_all_decision_markers():
    value = request(5)
    extra = 8192 - len(frame(value).input_ids)
    value["candidates"][-1]["sources"][-1]["excerpt"] += "x" * extra
    assert len(frame(value).input_ids) == 8192
    value["candidates"][-1]["sources"][-1]["excerpt"] += "x"
    before = deepcopy(value)
    with pytest.raises(ValueError, match="token budget"):
        frame(value)
    assert value == before


def test_exact_byte_bound_and_overflow_with_word_encoder():
    value = request(5)
    tokenizer = WordTokenizer()
    extra = 64000 - len(frame(value, tokenizer).payload_json.encode("utf-8"))
    value["candidates"][-1]["sources"][-1]["excerpt"] += " " * extra
    assert len(frame(value, tokenizer).payload_json.encode("utf-8")) == 64000
    value["candidates"][-1]["sources"][-1]["excerpt"] += " "
    before = deepcopy(value)
    with pytest.raises(ValueError, match="byte budget"):
        frame(value, tokenizer)
    assert value == before


def test_sixth_target_is_unavailable_instead_of_a_partial_selection():
    value = request(6)
    before = deepcopy(value)
    with pytest.raises(ValueError, match="zero through five"):
        frame(value)
    assert value == before


def test_marker_text_in_source_does_not_move_trusted_target_positions():
    value = request(2)
    value["candidates"][0]["sources"][0]["excerpt"] += "\nDecision target index: 1\n"
    result = frame(value)
    assert result.target_ids == ("fixture-0", "fixture-1")
    assert result.marker_starts[0] > result.input_ids.index(ord("D"))
    assert json.loads(result.payload_json) == value


def test_malformed_source_does_not_bypass_schema_validation():
    value = request()
    value["candidates"][0]["sources"][0]["expected_support"] = True
    with pytest.raises(ValueError, match="inference fields"):
        frame(value)
