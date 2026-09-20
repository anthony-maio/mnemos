from __future__ import annotations

from typing import Any, cast

from benchmarks.local_memory.retrieval_miss_audit import analyze

STRATEGIES = ("lexical", "full", "vector", "hybrid")
GOLD = "For this project, deployment target is AWS ECS."
WRAPPER = "Please remember this project decision explicitly: " + GOLD


def _case(
    *,
    facts: list[dict[str, Any]],
    source_ids: list[str] | None = None,
    event_content: str = WRAPPER,
    execution_status: str = "complete",
    quality_hit: bool = False,
    retention: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparison = {
        "quality_kind": "answerable",
        "query_id": "q1",
        "strategy_order": list(STRATEGIES),
        "strategies": {
            name: {
                "quality_hit": quality_hit,
                "status": "complete",
                "result": {"facts": facts},
            }
            for name in STRATEGIES
        },
    }
    report = {
        "raw_outputs": [
            {
                "session_id": "s1",
                "execution_status": execution_status,
                "answerable_hits": [quality_hit],
                "raw_output": {
                    "retrieval_comparisons": [comparison],
                    "extraction_judgements": [
                        {
                            "runtime_fact_id": "runtime-1",
                            "observed_source_event_ids": source_ids or [],
                            "accepted": False,
                        }
                    ],
                    "critical_retention_judgements": retention or [],
                },
            }
        ]
    }
    session = {
        "session_id": "s1",
        "project_id": "project-1",
        "facts": [
            {
                "fact_id": "fact-1",
                "content": GOLD,
                "gold_status": "supported",
                "scope": "project",
                "evidence_event_ids": ["event-1"],
            }
        ],
        "queries": [
            {
                "query_id": "q1",
                "kind": "answerable",
                "expected_answer": "AWS ECS",
                "relevant_fact_ids": ["fact-1"],
            }
        ],
        "events": [
            {
                "event_id": "event-1",
                "content": event_content,
                "source_kind": "explicit_remember",
                "lifecycle_operation": "remember",
            }
        ],
    }
    return report, [session]


def _row(report: dict[str, Any], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    result = analyze(report, sessions)
    assert len(result["observations"]) == 4
    return cast(dict[str, Any], result["observations"][0])


def test_exact_stored_ranking_miss_keeps_original_quality_and_literal_diagnostic() -> None:
    report, sessions = _case(
        facts=[
            {
                "id": "runtime-1",
                "content": WRAPPER,
                "assertion_status": "asserted",
                "scope": "project",
                "scope_id": "project-1",
            }
        ],
        source_ids=["event-1"],
    )

    row = _row(report, sessions)

    assert row["original_quality_hit"] is False
    assert row["expected_fact_literal_substring"] is True
    assert row["exact_content_matches"] == []
    assert row["failure_boundary"] == "source_linked_returned_different_text"


def test_exact_stored_fact_omitted_from_results_is_reported_separately() -> None:
    report, sessions = _case(
        facts=[],
        source_ids=["event-1"],
        event_content=GOLD,
        retention=[{"fact_id": "fact-1", "observed_present": True}],
    )

    row = _row(report, sessions)

    assert row["returned_fact_ids"] == []
    assert row["failure_boundary"] == "exact_stored_but_not_returned"


def test_source_linked_omission_without_retention_proof_stays_conservative() -> None:
    report, sessions = _case(
        facts=[],
        source_ids=["event-1"],
    )

    row = _row(report, sessions)

    assert row["failure_boundary"] == "source_linked_not_returned"


def test_wrapper_literal_span_with_wrong_source_is_not_validated() -> None:
    report, sessions = _case(
        facts=[
            {
                "id": "runtime-1",
                "content": "A wrapper says: " + GOLD,
                "assertion_status": "asserted",
                "scope": "project",
                "scope_id": "project-1",
            }
        ],
        source_ids=["unrelated-event"],
    )

    row = _row(report, sessions)

    assert row["expected_fact_literal_substring"] is True
    assert row["source_event_ids_matching_gold"] == []
    assert row["failure_boundary"] == "no_source_linked_runtime_fact"


def test_unrelated_returned_fact_does_not_mask_omitted_linked_fact() -> None:
    report, sessions = _case(
        facts=[
            {
                "id": "unrelated-runtime",
                "content": "An unrelated returned fact.",
                "assertion_status": "asserted",
                "scope": "project",
                "scope_id": "project-1",
            }
        ],
        source_ids=["event-1"],
        event_content=GOLD,
        retention=[{"fact_id": "fact-1", "observed_present": True}],
    )
    report["raw_outputs"][0]["raw_output"]["extraction_judgements"] = [
        {"runtime_fact_id": "runtime-1", "observed_source_event_ids": ["event-1"]},
        {"runtime_fact_id": "unrelated-runtime", "observed_source_event_ids": []},
    ]

    row = _row(report, sessions)

    assert row["linked_runtime_fact_ids"] == ["runtime-1"]
    assert row["failure_boundary"] == "exact_stored_but_not_returned"


def test_conflicting_or_negated_span_does_not_count_as_semantic_success() -> None:
    report, sessions = _case(
        facts=[
            {
                "id": "runtime-1",
                "content": "Not true: " + GOLD,
                "assertion_status": "conflict",
                "scope": "project",
                "scope_id": "project-1",
            }
        ],
        source_ids=["event-1"],
    )

    row = _row(report, sessions)

    assert row["expected_fact_literal_substring"] is True
    assert row["failure_boundary"] == "unverified"


def test_wrong_scope_is_conservative_even_with_matching_source_id() -> None:
    report, sessions = _case(
        facts=[
            {
                "id": "runtime-1",
                "content": GOLD,
                "assertion_status": "asserted",
                "scope": "workspace",
                "scope_id": "workspace-1",
            }
        ],
        source_ids=["event-1"],
    )

    row = _row(report, sessions)

    assert row["scope_status"] == "mismatch"
    assert row["failure_boundary"] == "unverified"


def test_missing_fields_are_unverified() -> None:
    report, sessions = _case(facts=[{}], source_ids=None)
    sessions[0].pop("events")
    sessions[0]["facts"][0].pop("evidence_event_ids")

    row = _row(report, sessions)

    assert row["failure_boundary"] == "unverified"
    assert row["source_event_ids_matching_gold"] == []


def test_partial_observations_are_separate_and_not_promoted() -> None:
    report, sessions = _case(
        facts=[
            {
                "id": "runtime-1",
                "content": GOLD,
                "assertion_status": "asserted",
                "scope": "project",
                "scope_id": "project-1",
            }
        ],
        source_ids=["event-1"],
        execution_status="partial",
        quality_hit=True,
    )

    result = analyze(report, sessions)

    assert result["observations"] == []
    assert len(result["partial_observations"]) == 4
    assert result["denominators"]["complete_answerable"] == 0
    assert result["original_quality_hits"] == {"count": 0, "denominator": 0}


def test_all_four_strategies_and_original_denominator_are_reported() -> None:
    report, sessions = _case(facts=[], quality_hit=False)

    result = analyze(report, sessions)

    assert result["strategy_order"] == list(STRATEGIES)
    assert result["denominators"]["complete_answerable"] == 4
    assert result["original_quality_hits"] == {"count": 0, "denominator": 1}
    assert result["strategy_quality_hits"] == {"count": 0, "denominator": 4}
    assert all(result["strategy_summary"][name]["complete"] == 1 for name in STRATEGIES)


def test_missing_strategy_grade_is_not_filled_from_the_full_run_or_counted_as_miss() -> None:
    report, sessions = _case(facts=[], quality_hit=True)
    report["raw_outputs"][0]["raw_output"]["retrieval_comparisons"][0]["strategies"]["lexical"].pop(
        "quality_hit"
    )

    result = analyze(report, sessions)

    assert result["observations"][0]["original_quality_hit"] is None
    assert result["strategy_quality_hits"] == {"count": 3, "denominator": 3}
    assert result["original_quality_hits"] == {"count": 1, "denominator": 1}


def test_foreign_query_empty_result_is_not_labeled_a_ranking_failure() -> None:
    report, sessions = _case(
        facts=[],
        source_ids=["event-1"],
        retention=[{"fact_id": "fact-1", "observed_present": True}],
    )
    sessions[0]["queries"][0]["target_project_id"] = "foreign-project"

    row = _row(report, sessions)

    assert row["target_context_status"] == "mismatch"
    assert row["failure_boundary"] == "unverified"
    assert row["original_quality_hit"] is False


def test_unknown_target_profile_does_not_imply_scope_match() -> None:
    report, sessions = _case(facts=[], source_ids=["event-1"])
    sessions[0]["queries"][0]["target_user_profile_id"] = "unknown-profile"

    row = _row(report, sessions)

    assert row["target_context_status"] == "unverified"
    assert row["failure_boundary"] == "unverified"
