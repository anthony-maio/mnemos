"""Offline regressions for the recovered evaluation reporting and source boundaries."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmarks.local_memory.colab_heldout import aggregate
from mnemos.extraction.judged_classification import classify_source_memories_with_judge
from mnemos.judgment import MockJudge
from mnemos.types import EpisodicEvent


def test_full_only_summary_preserves_failed_correction_attempts(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    report = {
        "status": "incomplete",
        "raw_outputs": [
            {
                "execution_status": "complete",
                "supported_precision": [True, False, None],
                "raw_output": {"lifecycle_summary": {"correct": {"attempted": 1, "committed": 1}}},
            },
            {
                "execution_status": "partial",
                "supported_precision": [True],
                "raw_output": {"lifecycle_summary": {"correct": {"attempted": 1, "committed": 0}}},
            },
        ],
    }
    (reports / "full.json").write_text(json.dumps(report), encoding="utf-8")

    summary = aggregate(mirror=tmp_path, repetitions=3)

    assert summary["statuses"] == {"full": "incomplete"}
    assert "baseline" in summary["missing_profiles"]
    assert summary["paired_comparisons"] == {}
    assert not (reports / "baseline.json").exists()
    metrics = summary["profile_metrics"]["full"]
    assert metrics["supported_precision"] == {
        "estimate": 0.5,
        "sample_count": 2,
        "complete_observations": 1,
        "excluded_incomplete_observations": 1,
    }
    assert metrics["correction_commit_rate"] == {
        "estimate": 0.5,
        "committed": 1,
        "attempted": 2,
    }


@pytest.mark.parametrize(
    ("kind", "retained"),
    [("asserted_fact", True), ("quoted_material", False), ("temporary_instruction", False)],
)
def test_retention_prefix_does_not_override_the_judged_kind(kind: str, retained: bool) -> None:
    # A scripted judge tests the rendering contract, not real-model classification quality.
    text = "For this project, keep this in mind: The deployment uses the blue region."
    event = EpisodicEvent(
        id="prefix-contract",
        event_fingerprint="prefix-contract",
        role="user",
        content=text,
        normalized_content=text.lower(),
        scope="project",
        scope_id="test-project",
        source="manual",
        metadata={"user_profile_id": "test-profile", "project_id": "test-project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    judge = MockJudge({"kind:r0": kind, "retention_request:r0": "u0"})

    result = asyncio.run(classify_source_memories_with_judge([event], judge))

    assert bool(result.memories) is retained
    if retained:
        assert result.memories[0].content == text
        assert result.memories[0].source_role == "user"
    assert result.decisions[0].kind == kind
