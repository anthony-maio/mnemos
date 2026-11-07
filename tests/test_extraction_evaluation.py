"""Scoring must distinguish extraction quality, review, and failed operations."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.local_memory.extraction_evaluation import (
    ExpectedOutcome,
    ExtractionCase,
    grade_extraction,
    validate_case,
)
from mnemos.extraction.source_classification import ClassifiedInventory
from mnemos.contracts import FactProposal
from mnemos.extraction.source_pipeline import ReviewOutcome, SourcePipelineResult
from mnemos.extraction.source_reconciliation import StagedReconciliation
from mnemos.extraction.source_selection import ExtractiveMemory
from mnemos.types import EpisodicEvent, MemoryChunk


def case(content: str = "The worker limit is twelve.") -> ExtractionCase:
    event = EpisodicEvent(
        id="source-1",
        event_fingerprint="fixture-source-1",
        role="user",
        content=content,
        normalized_content=content.casefold(),
        scope="project",
        scope_id="project-1",
        source="codex_hook",
        metadata={"user_profile_id": "profile-1"},
        occurred_at=datetime(2026, 11, 7, tzinfo=timezone.utc),
    )
    return ExtractionCase(
        case_id="limit",
        cluster_id="limit-family",
        events=(event,),
        targets=(),
        expected=(
            ExpectedOutcome(
                source_id=event.id,
                start=0,
                end=len(content),
                kind="asserted_fact",
                operations=("add",),
                critical=True,
            ),
        ),
    )


def result(
    c: ExtractionCase,
    *,
    start: int = 0,
    end: int | None = None,
    operation: str = "add",
    kind: str = "asserted_fact",
) -> SourcePipelineResult:
    event = c.events[0]
    candidate = ExtractiveMemory(
        source_id=event.id,
        source_role=event.role,
        source_fingerprint=event.event_fingerprint,
        source_content_sha256=hashlib.sha256(event.content.encode()).hexdigest(),
        source_channel=event.source,
        occurred_at=event.occurred_at.isoformat(),
        profile_id="profile-1",
        scope="project",
        scope_id="project-1",
        start=start,
        end=end or len(event.content),
        content=event.content[start:end],
        kind=kind,
        retention_request_quote=None,
    )
    staged = StagedReconciliation(
        candidate_index=0,
        source_id=event.id,
        source_role=event.role,
        source_kind=kind,
        content=candidate.content,
        operation=operation,
        reason="Needs a known target." if operation in {"unresolved", "conflict"} else None,
    )
    if operation in {"support", "correct"}:
        staged.target_fact_id = c.targets[0].id
        staged.expected_version = c.targets[0].version
    proposals = []
    reviews = []
    if operation in {"add", "support", "correct"}:
        proposals.append(
            FactProposal(
                content=candidate.content,
                evidence_ids=[event.id],
                operation=operation,
                target_fact_id=staged.target_fact_id,
                expected_version=staged.expected_version,
            )
        )
    else:
        reviews.append(ReviewOutcome.model_validate(staged.model_dump(exclude={"reason_code"})))
    return SourcePipelineResult(
        inventory=ClassifiedInventory(selections=[]),
        candidates=[candidate],
        reconciliations=[staged],
        proposals=proposals,
        reviews=reviews,
    )


def test_exact_source_outcome_retains_critical_fact() -> None:
    c = case()
    grade = grade_extraction(c, result(c), error=None, provider_calls=2)
    assert grade["critical_retained"] == grade["exact_expected_outcomes"] == 1
    assert grade["accepted_outcomes"] == grade["exact_supported_accepted"] == 1
    assert grade["case_exact"] and grade["operational_success"]


@pytest.mark.parametrize("error", ["TimeoutError", "ValueError", "MnemosTransientError"])
def test_failed_empty_extraction_never_gets_abstention_credit(error: str) -> None:
    c = replace(case(), expected=())
    grade = grade_extraction(c, None, error=error, provider_calls=1)
    assert not grade["correct_abstention"] and not grade["case_exact"]
    assert not grade["operational_success"]


def test_valid_empty_extraction_can_abstain() -> None:
    c = replace(case(), expected=())
    empty = result(c).model_copy(update={"candidates": [], "reconciliations": [], "proposals": []})
    assert grade_extraction(c, empty, error=None, provider_calls=1)["correct_abstention"]


def test_retaining_extra_instruction_is_not_an_exact_critical_match() -> None:
    c = case("The worker limit is twelve. Use bullets for this reply.")
    c = replace(c, expected=(replace(c.expected[0], end=len("The worker limit is twelve. ")),))
    grade = grade_extraction(c, result(c), error=None, provider_calls=2)
    assert grade["critical_retained"] == 0
    assert grade["critical_span_contained"] == 1
    assert grade["unmatched_accepted"] == 1 and not grade["case_exact"]


def test_duplicate_proposals_do_not_double_credit_one_expected_fact() -> None:
    c = case()
    actual = result(c)
    actual.reconciliations.append(actual.reconciliations[0].model_copy(deep=True))
    actual.proposals.append(actual.proposals[0].model_copy(deep=True))
    grade = grade_extraction(c, actual, error=None, provider_calls=2)
    assert grade["critical_retained"] == grade["exact_supported_accepted"] == 1
    assert grade["accepted_outcomes"] == 2 and grade["unmatched_accepted"] == 1
    assert not grade["case_exact"]


def test_unresolved_review_cannot_satisfy_critical_fact_retention() -> None:
    c = case()
    grade = grade_extraction(c, result(c, operation="unresolved"), error=None, provider_calls=2)
    assert grade["review_outcomes"] == 1 and grade["accepted_outcomes"] == 0
    assert grade["critical_retained"] == 0 and not grade["case_exact"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_role", "assistant"),
        ("scope_id", "foreign-project"),
        ("profile_id", "other-profile"),
        ("source_content_sha256", "0" * 64),
        ("content", "The worker limit is thirty."),
    ],
)
def test_invalid_source_provenance_cannot_match(field: str, value: str) -> None:
    c = case()
    actual = result(c)
    actual.candidates[0] = ExtractiveMemory(**{**asdict(actual.candidates[0]), field: value})
    grade = grade_extraction(c, actual, error=None, provider_calls=2)
    assert grade["invalid_provenance_outcomes"] == 1
    assert grade["critical_retained"] == 0 and not grade["case_exact"]


def test_correction_requires_exact_target_and_observed_version() -> None:
    c = case("Correction: the worker limit is twelve.")
    target = MemoryChunk(
        id="old-limit",
        content="The worker limit is eight.",
        version=4,
        metadata={"user_profile_id": "profile-1", "scope": "project", "scope_id": "project-1"},
    )
    c = replace(
        c,
        targets=(target,),
        expected=(
            replace(
                c.expected[0],
                operations=("correct",),
                kind="correction_or_rejection",
                target_fact_id=target.id,
                expected_version=4,
            ),
        ),
    )
    actual = result(c, operation="correct", kind="correction_or_rejection")
    actual.reconciliations[0].target_fact_id = target.id
    actual.reconciliations[0].expected_version = 3
    bad = grade_extraction(c, actual, error=None, provider_calls=2)
    assert bad["invalid_provenance_outcomes"] == 1 and bad["critical_retained"] == 0
    actual.reconciliations[0].expected_version = 4
    assert grade_extraction(c, actual, error=None, provider_calls=2)["case_exact"]


def test_guard_rejection_is_separate_from_quality_and_requires_zero_calls() -> None:
    c = replace(case(), expected=(), expected_preflight_error="foreign scope")
    correct = grade_extraction(c, None, error="ValueError: foreign scope", provider_calls=0)
    assert correct["guard_passed"] and not correct["included_in_quality"]
    assert not correct["correct_abstention"]
    assert not grade_extraction(c, None, error="ValueError: foreign scope", provider_calls=1)[
        "guard_passed"
    ]


def test_case_rejects_impossible_source_span_before_execution() -> None:
    c = case()
    c = replace(c, expected=(replace(c.expected[0], end=1000),))
    with pytest.raises(ValueError, match="span"):
        validate_case(c)


def test_critical_expectation_cannot_be_only_a_review() -> None:
    c = case()
    c = replace(c, expected=(replace(c.expected[0], operations=("unresolved",)),))
    with pytest.raises(ValueError, match="critical"):
        validate_case(c)


def test_staged_acceptance_without_a_proposal_is_not_retention() -> None:
    c = case()
    actual = result(c)
    actual.proposals.clear()
    grade = grade_extraction(c, actual, error=None, provider_calls=2)
    assert grade["invalid_provenance_outcomes"] == 1
    assert grade["critical_retained"] == 0 and not grade["case_exact"]


def test_development_dataset_has_separate_guard_and_quality_denominators() -> None:
    from benchmarks.local_memory.extraction_evaluation_run import load_cases

    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/local_memory/fixtures/extraction-isolated-development-v1.json"
    )
    data, cases = load_cases(path)
    assert data["stage"] == "development"
    assert len(cases) == 16
    assert sum(case.expected_preflight_error is not None for case in cases) == 1
    assert sum(not case.expected and case.expected_preflight_error is None for case in cases) == 4


@pytest.mark.asyncio
async def test_dataset_drift_rejected_before_any_runtime_or_output(tmp_path, monkeypatch) -> None:
    from benchmarks.local_memory.extraction_evaluation_run import run

    path = tmp_path / "changed.json"
    path.write_text("{}", encoding="utf-8")

    def forbidden_client(*args, **kwargs):
        raise AssertionError("Must reject changed inputs before contacting a runtime")

    monkeypatch.setattr("httpx.AsyncClient", forbidden_client)
    args = Namespace(
        dataset=str(path),
        dataset_sha256="0" * 64,
        endpoint="http://127.0.0.1:11437",
        tokenizer_url="http://127.0.0.1:11438",
        source_root=str(tmp_path),
        protocol=str(tmp_path / "protocol.md"),
        output=str(tmp_path / "result"),
        repetitions=1,
    )
    with pytest.raises(ValueError, match="frozen digest"):
        await run(args)
    assert not (tmp_path / "result").exists()
