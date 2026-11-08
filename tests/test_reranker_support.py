import math

import pytest

from benchmarks.local_memory.reranker_support import select_witnesses


def witness(fact="a", evidence="source"):
    return {"fact_id": fact, "evidence_id": evidence}


def test_selects_only_passing_witnesses_and_deduplicates_fact():
    result = select_witnesses(
        [witness(), witness(evidence="second"), witness("b")],
        [[0.0, 3.0], [0.0, 1.0], [2.0, 0.0]],
    )
    assert result["selected_ids"] == ["a"]
    assert result["witnesses"] == {"a": ["source"]}


def test_score_is_two_label_softmax_and_handles_extreme_logits():
    result = select_witnesses([witness(), witness("b")], [[-10000, 10000], [0, 0]])
    assert [r["score"] for r in result["scores"]] == [1.0, 0.5]
    assert result["selected_ids"] == ["a"]
    assert select_witnesses([], []) == {"selected_ids": [], "witnesses": {}, "scores": []}


@pytest.mark.parametrize("bad", [[math.nan, 1], [0, math.inf], [True, 1], [0], [0, 1, 2]])
def test_invalid_later_witness_invalidates_entire_question(bad):
    with pytest.raises(ValueError, match="logits"):
        select_witnesses([witness(), witness("b")], [[0, 3], bad])


def test_rejects_duplicate_missing_and_oversized_witnesses():
    with pytest.raises(ValueError, match="Duplicate"):
        select_witnesses([witness(), witness()], [[0, 3], [0, 3]])
    with pytest.raises(ValueError, match="identity"):
        select_witnesses([witness(evidence="")], [[0, 3]])
    with pytest.raises(ValueError, match="size"):
        select_witnesses([witness(str(i)) for i in range(33)], [[0, 3]] * 33)
    with pytest.raises(ValueError, match="size"):
        select_witnesses([witness()], [])
