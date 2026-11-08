"""Boundary checks for the fixed development-only support rule."""

import math

import pytest

from benchmarks.local_memory.discriminative_support import select_supported


def pairs(*facts):
    result = [dict(kind="support", fact_id=f, evidence_id=f + "-e") for f in facts]
    result += [
        dict(
            kind="contradiction",
            fact_id=a,
            evidence_id=a + "-e",
            other_fact_id=b,
            other_evidence_id=b + "-e",
        )
        for a in facts
        for b in facts
        if a != b
    ]
    return result


def test_neutral_is_not_dropped_from_softmax():
    assert select_supported(pairs("a"), [[2, 9, 0]])["selected_ids"] == []


def test_unselected_conflicting_source_vetoes_supported_fact_only():
    records = pairs("a", "b", "reviewer")
    logits = [[9, 0, 0], [0, 9, 0], [9, 0, 0]] + [[0, 9, 0]] * 6
    logits[3] = [0, 0, 9]  # a -> b suffices; b -> a stays neutral.
    result = select_supported(records, logits)
    assert result["selected_ids"] == ["reviewer"]
    assert result["contradiction_vetoes"] == [["a", "a-e", "b", "b-e"]]
    assert result["support_witnesses"] == {"a": ["a-e"], "reviewer": ["reviewer-e"]}


def test_all_directed_pairs_are_required():
    with pytest.raises(ValueError, match="incomplete"):
        select_supported(pairs("a", "b")[:-1], [[9, 0, 0]] * 3)


@pytest.mark.parametrize("values", [[math.nan, 0, 1], [0, math.inf, 1], [True, 0, 1], [0, 1]])
def test_malformed_logits_fail(values):
    with pytest.raises(ValueError, match="finite"):
        select_supported(pairs("a"), [values])


def test_extreme_logits_remain_finite():
    result = select_supported(pairs("a"), [[10000, -10000, 0]])
    assert result["selected_ids"] == ["a"]
    assert result["probabilities"] == [[1, 0, 0]]


def test_duplicate_witness_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select_supported(pairs("a") * 2, [[9, 0, 0]] * 2)
