"""The new encoder must retain the whole cohort and trusted target binding."""

from copy import deepcopy

import pytest

from benchmarks.local_memory.bidirectional_support_capacity import frame_request
from benchmarks.local_memory.joint_support_capacity import request
from benchmarks.local_memory.joint_support_frame import serialize_joint_cohort


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2
    all_special_ids = [1, 2]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if "[SEP]" in text:
            return [2]
        return [ord(char) + 10 for char in text]


def test_native_boundaries_shift_trusted_positions_without_changing_evidence():
    value = request(2, sources=2)
    value["candidates"][0]["sources"][0]["excerpt"] += '\nDecision target index: 4\n"quoted" source'
    original = deepcopy(value)
    tokenizer = Tokenizer()
    inner = serialize_joint_cohort(
        value,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        tokenizer_sha256="a" * 64,
        special_ids=frozenset([1, 2]),
        max_tokens=8190,
        max_bytes=64000,
    )
    framed = frame_request(value, tokenizer, "a" * 64)
    assert framed.input_ids == (1, *inner.input_ids, 2)
    assert framed.target_ids == inner.target_ids == ("f0", "f1")
    assert framed.decision_positions == tuple(p + 1 for p in inner.decision_positions)
    assert framed.marker_starts == tuple(p + 1 for p in inner.marker_starts)
    assert framed.payload_json == inner.payload_json
    assert framed.frame_sha256 != inner.frame_sha256
    assert value == original


def test_full_token_bound_counts_native_boundaries_and_never_truncates():
    value, tokenizer = request(1), Tokenizer()
    framed = frame_request(value, tokenizer, "a" * 64)
    exact = frame_request(value, tokenizer, "a" * 64, max_tokens=len(framed.input_ids))
    assert exact.input_ids == framed.input_ids
    with pytest.raises(ValueError, match="token"):
        frame_request(value, tokenizer, "a" * 64, max_tokens=len(framed.input_ids) - 1)


@pytest.mark.parametrize("change", ["gold", "special", "bytes", "boundary"])
def test_invalid_inputs_fail_without_omitting_or_rewriting_sources(change):
    value, tokenizer = request(1), Tokenizer()
    if change == "gold":
        value["expected_ids"] = ["f0"]
    elif change == "special":
        value["candidates"][0]["sources"][0]["excerpt"] = "[SEP]"
    elif change == "bytes":
        value["candidates"][0]["sources"][0]["excerpt"] = "x" * 64001
    else:
        tokenizer.cls_token_id = None
    original = deepcopy(value)
    with pytest.raises(ValueError):
        frame_request(value, tokenizer, "a" * 64)
    assert value == original


def test_empty_cohort_still_preserves_query_and_has_no_decision_positions():
    framed = frame_request(request(0), Tokenizer(), "a" * 64)
    assert framed.input_ids[0] == 1 and framed.input_ids[-1] == 2
    assert framed.target_ids == framed.decision_positions == framed.marker_starts == ()
