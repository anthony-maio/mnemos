from __future__ import annotations

import pytest

from benchmarks.local_memory import scoring


@pytest.mark.parametrize(
    ("selected", "expected"),
    [([0, 1], 0.25), ([0, 0], 0.0), ([1, 1], 1.0)],
)
def test_cluster_bootstrap_keeps_every_member_of_unequal_clusters(
    monkeypatch: pytest.MonkeyPatch, selected: list[int], expected: float
) -> None:
    choices = iter(selected)

    class FixedDraws:
        def __init__(self, seed: int) -> None:
            assert seed == 20260916

        def randrange(self, count: int) -> int:
            assert count == 2
            return next(choices)

    monkeypatch.setattr(scoring.random, "Random", FixedDraws)
    result = scoring.paired_bootstrap_interval(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        cluster_ids=["large", "large", "large", "small"],
        repetitions=1,
    )
    assert result.estimate == 0.25
    assert result.lower == result.upper == expected
    assert result.sample_count == 4
    assert result.cluster_count == 2
    assert list(choices) == []
