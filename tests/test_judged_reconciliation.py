"""Judged reconciliation keeps every authority rule and adds paraphrase support."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mnemos.config import JudgmentConfig
from mnemos.extraction.judged_reconciliation import (
    REASON_TEXT,
    reconcile_source_candidates_with_judge,
)
from mnemos.judgment import ChoiceAnswer, MockJudge
from tests.test_source_reconciliation import candidate, event, target


def _relation(choice: str, probability: float = 0.95) -> ChoiceAnswer:
    others = (1.0 - probability) / 2.0
    keys = ("unrelated", "same_information", "different_on_same_subject")
    return ChoiceAnswer(
        choice=choice,
        probabilities={key: (probability if key == choice else others) for key in keys},
        confidence=probability,
    )


def run(candidates, events, targets, judge, **kwargs):
    return asyncio.run(
        reconcile_source_candidates_with_judge(candidates, events, targets, judge, **kwargs)
    )


def test_no_targets_means_no_call_and_code_only_decisions() -> None:
    fact = event("e1", "The marker is cedar.")
    correction = event("e2", "Scrap the cedar marker.")
    judge = MockJudge()

    result = run(
        [candidate(fact), candidate(correction, kind="correction_or_rejection")],
        [fact, correction],
        [],
        judge,
    )

    assert judge.calls == []
    assert [(s.operation, s.reason_code) for s in result.staged] == [
        ("add", None),
        ("unresolved", "no_target_for_correction"),
    ]
    assert result.staged[1].reason == REASON_TEXT["no_target_for_correction"]


def test_a_reworded_restatement_supports_the_existing_fact() -> None:
    source = event("e1", "Cedar is the marker we use.")
    existing = target("fact-1", "The marker is cedar.")
    judge = MockJudge({"relation:c0:t0": _relation("same_information", 0.9)})

    result = run([candidate(source)], [source], [existing], judge)

    (staged,) = result.staged
    assert staged.operation == "support"
    assert staged.target_fact_id == "fact-1" and staged.expected_version == 1
    assert staged.content == "Cedar is the marker we use."
    state, questions, purpose = judge.calls[0]
    assert purpose == "reconcile_candidates"
    assert state["existing_facts"] == [{"id": "t0", "text": "The marker is cedar."}]
    assert "fact-1" not in str(state)  # aliases, never fact ids, reach the judge


def test_exact_text_skips_the_question_and_still_supports() -> None:
    source = event("e1", "The marker is cedar.")
    judge = MockJudge()
    result = run([candidate(source)], [source], [target()], judge)
    assert judge.calls == []
    assert result.staged[0].operation == "support"


def test_two_matching_targets_are_unresolved() -> None:
    source = event("e1", "Cedar is the marker we use.")
    targets = [target("fact-1", "The marker is cedar."), target("fact-2", "Our marker: cedar.")]
    judge = MockJudge({"relation:c0:*": _relation("same_information", 0.9)})
    result = run([candidate(source)], [source], targets, judge)
    assert (result.staged[0].operation, result.staged[0].reason_code) == (
        "unresolved",
        "multiple_matching_targets",
    )


def test_a_contradiction_becomes_a_conflict_review_with_a_reason() -> None:
    source = event("e1", "The marker is maple.")
    judge = MockJudge({"relation:c0:t0": _relation("different_on_same_subject", 0.8)})
    result = run([candidate(source)], [source], [target()], judge)
    (staged,) = result.staged
    assert staged.operation == "conflict"
    assert staged.target_fact_id == "fact-1"
    assert staged.reason == REASON_TEXT["contradicts_target"]
    assert staged.reason_code == "contradicts_target"


def test_a_user_correction_targets_the_referent_or_stays_unresolved() -> None:
    source = event("e1", "Scrap the cedar marker; it was wrong.")
    existing = target()
    found = MockJudge({"referent:c0": "t0"})
    result = run([candidate(source, kind="correction_or_rejection")], [source], [existing], found)
    (staged,) = result.staged
    assert staged.operation == "correct" and staged.target_fact_id == "fact-1"
    # A correction candidate never gets relation questions, only the referent one.
    assert list(found.calls[0][1]) == ["referent:c0"]

    missing = MockJudge({"referent:c0": "none"})
    result = run([candidate(source, kind="correction_or_rejection")], [source], [existing], missing)
    assert (result.staged[0].operation, result.staged[0].reason_code) == (
        "unresolved",
        "referent_not_found",
    )


def test_ambiguous_duplicate_targets_are_never_offered_as_referents() -> None:
    source = event("e1", "Scrap the cedar marker.")
    duplicates = [target("fact-1"), target("fact-2")]
    judge = MockJudge()
    result = run([candidate(source, kind="correction_or_rejection")], [source], duplicates, judge)
    # Only "none" remained, so no question was asked at all.
    assert judge.calls == []
    assert result.staged[0].reason_code == "referent_not_found"


def test_a_probability_just_under_threshold_is_unresolved_not_guessed() -> None:
    source = event("e1", "Cedar is the marker we use.")
    judge = MockJudge({"relation:c0:t0": _relation("same_information", 0.55)})
    result = run([candidate(source)], [source], [target()], judge)
    assert (result.staged[0].operation, result.staged[0].reason_code) == (
        "unresolved",
        "low_confidence",
    )
    strict = run(
        [candidate(source)],
        [source],
        [target()],
        MockJudge({"relation:c0:t0": _relation("same_information", 0.55)}),
        config=JudgmentConfig(support_min_probability=0.5),
    )
    assert strict.staged[0].operation == "support"


def test_non_user_reports_can_conflict_but_never_support() -> None:
    report = event("e1", "The marker is maple.", role="assistant")
    history = candidate(report, kind="attributed_history")
    conflicting = MockJudge({"relation:c0:t0": _relation("different_on_same_subject", 0.9)})
    result = run([history], [report], [target()], conflicting)
    assert result.staged[0].operation == "conflict"

    same = MockJudge({"relation:c0:t0": _relation("same_information", 0.9)})
    result = run([history], [report], [target()], same)
    assert result.staged[0].operation == "add"


def test_unrelated_candidates_are_added() -> None:
    source = event("e1", "Deploys happen on Thursdays.")
    judge = MockJudge({"relation:c0:t0": _relation("unrelated", 0.97)})
    result = run([candidate(source)], [source], [target()], judge)
    assert result.staged[0].operation == "add"
    assert result.trace.calls[0]["question_count"] == 1


def test_a_revision_with_a_referent_corrects_that_fact() -> None:
    source = event("e1", "Forget cedar; the marker is now maple for this project.")
    revision = replace(
        candidate(source, kind="continuing_requirement"), revises_existing_requirement=True
    )
    judge = MockJudge(
        {
            "relation:c0:t0": _relation("different_on_same_subject", 0.9),
            "referent:c0": "t0",
        }
    )
    result = run([revision], [source], [target()], judge)
    (staged,) = result.staged
    assert staged.operation == "correct" and staged.target_fact_id == "fact-1"
    assert staged.revises_existing_requirement is True


@pytest.mark.parametrize("bad", ["support", "correct"])
def test_the_validator_still_refuses_what_the_rules_forbid(bad: str) -> None:
    # A judge cannot make a non-user report support or correct anything: the
    # decision mapping never emits those operations for it, and the validator
    # is the second guard. Exercise the guard directly.
    from mnemos.extraction.source_reconciliation import ReconciliationDecision, _validate_decisions

    report = event("e1", "The marker is cedar.", role="assistant")
    history = candidate(report, kind="attributed_history")
    decision = ReconciliationDecision(
        candidate_index=0, operation=bad, target_fact_id="fact-1", expected_version=1
    )
    with pytest.raises(ValueError):
        _validate_decisions(
            [decision],
            [history],
            [target()],
            approved_paraphrase_support=frozenset({(0, "fact-1")}),
        )
