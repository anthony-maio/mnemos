"""Independent scores and mechanical preflight for the reconciliation study."""

from dataclasses import asdict
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from benchmarks.local_memory.reconciliation_evaluation import (
    ARMS,
    grade_reconciliation,
    load_reconciliation_cases,
)
from mnemos.extraction.source_reconciliation import StagedReconciliation
from mnemos.extraction.source_relationship import RELATIONSHIPS, reconcile_source_relationships
from tests.test_source_reconciliation import Provider

CORPUS = (
    Path(__file__).parents[1]
    / "benchmarks/local_memory/fixtures/reconciliation-development-v1.json"
)


def staged(case, alternative=0):
    candidate = case.candidate
    return StagedReconciliation(
        **case.alternatives[alternative].model_dump(),
        source_id=candidate.source_id,
        source_role=candidate.source_role,
        source_kind=candidate.kind,
        content=candidate.content,
        revises_existing_requirement=candidate.revises_existing_requirement,
    )


@pytest.mark.asyncio
async def test_all_frozen_alternatives_are_mechanically_eligible_in_every_arm() -> None:
    _, cases = load_reconciliation_cases(CORPUS)
    assert len(cases) == 20
    checks = 0
    for case in cases:
        before = asdict(case.candidate)
        for style, relevant in ARMS.values():
            for alternative in case.alternatives:
                raw = alternative.model_dump()
                raw["relationship"] = RELATIONSHIPS[raw.pop("operation")]
                response = {"decisions": [raw]}
                provider = Provider(response)
                result = await reconcile_source_relationships(
                    [case.candidate],
                    list(case.source.events),
                    list(case.source.targets),
                    provider,
                    prompt_style=style,
                    relevant_revision_flag=relevant,
                )
                schema = provider.calls[0][1]["format"]
                Draft202012Validator(schema).validate(response)
                assert grade_reconciliation(case, result, error=None)["case_exact"]
                checks += 1
        assert asdict(case.candidate) == before
    assert checks == 84


@pytest.mark.parametrize(
    "change",
    [
        {"source_id": "other-source"},
        {"source_role": "assistant"},
        {"source_kind": "asserted_fact"},
        {"content": "invented text"},
        {"candidate_index": 1},
        {"revises_existing_requirement": True},
        {"expected_version": 99},
        {"target_fact_id": "unseen-target"},
        {"operation": "unresolved", "target_fact_id": None, "expected_version": None},
    ],
)
def test_wrong_source_or_target_does_not_pass(change) -> None:
    _, cases = load_reconciliation_cases(CORPUS)
    case = cases[0]
    result = [staged(case).model_copy(update=change)]
    assert not grade_reconciliation(case, result, error=None)["case_exact"]


def test_empty_or_duplicate_results_are_not_successful_abstentions() -> None:
    _, cases = load_reconciliation_cases(CORPUS)
    case = cases[1]
    for result, error in [(None, "TimeoutError"), ([], None), ([staged(case)] * 2, None)]:
        grade = grade_reconciliation(case, result, error=error)
        assert not grade["case_exact"] and not grade["correct_abstention"]
    assert grade_reconciliation(case, [staged(case)], error=None)["correct_abstention"]


def test_conflict_accepts_only_the_two_prespecified_target_forms() -> None:
    _, cases = load_reconciliation_cases(CORPUS)
    case = cases[-1]
    assert len(case.alternatives) == 2
    for index in range(2):
        assert grade_reconciliation(case, [staged(case, index)], error=None)["case_exact"]
    wrong = staged(case).model_copy(
        update={"operation": "add", "target_fact_id": None, "expected_version": None}
    )
    assert not grade_reconciliation(case, [wrong], error=None)["case_exact"]


def test_loader_rejects_unobserved_expected_target_version(tmp_path) -> None:
    data = json.loads(CORPUS.read_text())
    data["cases"][0]["expected_alternatives"][0]["expected_version"] = 99
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unobserved target version"):
        load_reconciliation_cases(path)


def test_loader_rejects_invalid_supplied_classification(tmp_path) -> None:
    data = json.loads(CORPUS.read_text())
    data["cases"][0]["selections"][0]["revises_existing_requirement"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Requirement revision intent"):
        load_reconciliation_cases(path)
