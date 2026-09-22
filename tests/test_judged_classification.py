"""Judged classification: runs from continuation answers, role-restricted kinds, audit."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from mnemos.config import JudgmentConfig
from mnemos.extraction.judged_classification import (
    NON_USER_KINDS,
    USER_KINDS,
    classify_source_memories_with_judge,
)
from mnemos.judgment import Choice, ChoiceAnswer, MockJudge, Noul, NoulAnswer, Score, ScoreAnswer
from mnemos.types import EpisodicEvent


def source(content: str, role: str = "user", event_id: str = "source") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=event_id,
        role=role,
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def run(events, judge, **kwargs):
    return asyncio.run(classify_source_memories_with_judge(events, judge, **kwargs))


def test_continuations_form_ranges_and_kinds_label_them() -> None:
    # Units: 0 "Release acks use ACK-731." 1 "This stays our rule." 2 "For this reply only, be brief."
    text = "Release acks use ACK-731. This stays our rule. For this reply only, be brief."
    judge = MockJudge(
        {
            "continues:u1": 0.9,
            "continues:u2": 0.1,
            "kind:r0": "continuing_requirement",
            "kind:r1": "temporary_instruction",
        }
    )

    result = run([source(text)], judge)

    assert [(d.first_unit, d.last_unit, d.kind, d.retained) for d in result.decisions] == [
        (0, 1, "continuing_requirement", True),
        (2, 2, "temporary_instruction", False),
    ]
    # Units keep the whitespace after their punctuation, as the renderer always has.
    assert [m.content.strip() for m in result.memories] == [
        "Release acks use ACK-731. This stays our rule."
    ]
    assert [s.kind for s in result.inventory.selections] == [
        "continuing_requirement",
        "temporary_instruction",
    ]
    purposes = [call["purpose"] for call in result.trace.calls]
    assert purposes == ["segment_source", "classify_source"]
    # Kind questions offer only the role's kinds, in the fixed order.
    _, questions, _ = judge.calls[1]
    kind_question = questions["kind:r0"]
    assert isinstance(kind_question, Choice) and kind_question.keys == USER_KINDS


def test_a_single_unit_needs_no_segmentation_call() -> None:
    judge = MockJudge({"kind:r0": "asserted_fact"})
    result = run([source("The marker is cedar.")], judge)
    assert [call["purpose"] for call in result.trace.calls] == ["classify_source"]
    assert [m.kind for m in result.memories] == ["asserted_fact"]


def test_non_user_sources_can_only_be_attributed_history() -> None:
    judge = MockJudge({"kind:r0": "attributed_history"})
    result = run([source("The build failed twice on Tuesday.", role="assistant")], judge)
    _, questions, _ = judge.calls[0]
    assert questions["kind:r0"].keys == NON_USER_KINDS  # type: ignore[union-attr]
    assert "revises:r0" not in questions
    assert [m.kind for m in result.memories] == ["attributed_history"]


def test_low_confidence_retained_kinds_are_audited_not_retained() -> None:
    def script(key, question, state):
        if key == "kind:r0":
            return ChoiceAnswer(
                choice="asserted_fact",
                probabilities={k: (0.4 if k == "asserted_fact" else 0.06) for k in question.keys},
                confidence=0.3,
            )
        if isinstance(question, Noul):
            return NoulAnswer(0.0)
        return ChoiceAnswer(
            choice=question.keys[0],
            probabilities={k: (1.0 if k == question.keys[0] else 0.0) for k in question.keys},
            confidence=1.0,
        )

    result = run([source("The marker might be cedar.")], MockJudge(script))

    assert result.memories == []
    assert result.inventory.selections == []
    (decision,) = result.decisions
    assert decision.kind == "asserted_fact" and decision.note == "low_confidence"
    assert decision.probability == 0.4


def test_retained_uncertainty_requires_a_cited_request_unit() -> None:
    text = "The cache size might need to change. Please remember that open question."
    with_request = MockJudge(
        {
            "continues:u1": 0.9,
            "kind:r0": "explicitly_retained_uncertainty",
            "retention_request:r0": "u1",
        }
    )
    result = run([source(text)], with_request)
    (memory,) = result.memories
    assert memory.kind == "explicitly_retained_uncertainty"
    assert memory.retention_request_quote == "Please remember that open question."

    without = MockJudge({"continues:u1": 0.9, "kind:r0": "explicitly_retained_uncertainty"})
    result = run([source(text)], without)
    assert result.memories == []
    assert result.decisions[0].kind == "unrequested_speculation"
    assert result.decisions[0].note == "uncertainty_without_request"


def test_revision_intent_is_recorded_only_for_user_requirements() -> None:
    judge = MockJudge({"kind:r0": "continuing_requirement", "revises:r0": 0.95})
    result = run([source("Forget the old rule; release acks now use ACK-900.")], judge)
    (memory,) = result.memories
    assert memory.revises_existing_requirement is True


def test_ranges_never_cross_a_batch_boundary() -> None:
    text = " ".join(f"Sentence {index} is here." for index in range(6))
    judge = MockJudge({"continues:*": 0.99, "kind:*": "asserted_fact"})
    result = run([source(text)], judge, config=JudgmentConfig(max_units_per_request=4))
    assert [(d.first_unit, d.last_unit) for d in result.decisions] == [(0, 3), (4, 5)]
    assert [call["purpose"] for call in result.trace.calls] == [
        "segment_source",
        "segment_source",
        "classify_source",
    ]


def test_affective_scores_ride_along_once_per_event() -> None:
    def script(key, question, state):
        if isinstance(question, Score):
            index = {"valence": 4, "arousal": 0, "complexity": 2}[key]
            return ScoreAnswer(
                value=float(index),
                confidence=0.9,
                probabilities={str(i): (1.0 if i == index else 0.0) for i in range(5)},
            )
        if isinstance(question, Noul):
            return NoulAnswer(0.0)
        choice = "asserted_fact" if key.startswith("kind:") else question.keys[0]
        return ChoiceAnswer(
            choice=choice,
            probabilities={k: (1.0 if k == choice else 0.0) for k in question.keys},
            confidence=1.0,
        )

    result = run([source("The marker is cedar.")], MockJudge(script), include_affective=True)

    assert result.cognitive_state is not None
    assert result.cognitive_state.valence == 1.0
    assert result.cognitive_state.arousal == 0.0
    assert result.cognitive_state.complexity == 0.5
    assert result.affective_confidence == {"valence": 0.9, "arousal": 0.9, "complexity": 0.9}
