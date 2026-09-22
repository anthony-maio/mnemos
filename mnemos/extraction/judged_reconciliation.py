"""Reconcile source candidates to existing facts through typed judgments.

For each candidate and each observed target the judge answers one Choice:
unrelated, the same information, or a different claim on the same subject.
For a user revision it also picks which target is being corrected. Code
turns those answers into the same operations the prompt-driven reconciler
produces, keeps every authority rule, and adds one thing the exact-text
matcher could not: a reworded restatement of an existing fact counts as
support.

Nothing here writes a fact or generates a word. ``reason`` text for conflict
and unresolved outcomes comes from a fixed table, keyed by a code that is
stored alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import JudgmentConfig
from ..judgment import Choice, ChoiceAnswer, Description, Judge, JudgmentTrace
from ..types import EpisodicEvent, MemoryChunk
from ..utils.storage import normalize_memory_text
from .source_reconciliation import (
    ReconciliationDecision,
    StagedReconciliation,
    _is_user_revision,
    _validate_decisions,
    _validate_inputs,
)
from .source_selection import ExtractiveMemory

ReasonCode = Literal[
    "contradicts_target",
    "multiple_matching_targets",
    "ambiguous_revision_target",
    "referent_not_found",
    "no_target_for_correction",
    "low_confidence",
]

REASON_TEXT: dict[str, str] = {
    "contradicts_target": "The source states something different about the subject of an existing fact.",
    "multiple_matching_targets": "More than one existing fact carries the same information.",
    "ambiguous_revision_target": "The correction's target could not be identified among the existing facts.",
    "referent_not_found": "The correction refers to something that is not among the existing facts.",
    "no_target_for_correction": "The correction has no existing fact to apply to.",
    "low_confidence": "The judge was not confident enough to act on this relationship.",
}

RELATION_OPTIONS: tuple[tuple[str, Description], ...] = (
    ("unrelated", "about something else, or only loosely related to the existing fact"),
    (
        "same_information",
        {
            "what": (
                "states the same fact, preference, or rule as the existing fact, possibly "
                "reworded, shortened, or expanded"
            ),
            "not": "any added, removed, or changed value, condition, scope, timing, or negation",
        },
    ),
    (
        "different_on_same_subject",
        "about the same subject as the existing fact but changes, contradicts, narrows, "
        "widens, or withdraws it",
    ),
)


@dataclass
class JudgedReconciliation:
    staged: list[StagedReconciliation]
    trace: JudgmentTrace = field(default_factory=JudgmentTrace)
    relations: dict[tuple[int, str], ChoiceAnswer] = field(default_factory=dict)
    referents: dict[int, ChoiceAnswer] = field(default_factory=dict)


def _decision(
    index: int,
    operation: str,
    *,
    target: MemoryChunk | None = None,
    reason_code: str | None = None,
) -> tuple[ReconciliationDecision, str | None]:
    return (
        ReconciliationDecision(
            candidate_index=index,
            operation=operation,  # type: ignore[arg-type]
            target_fact_id=target.id if target is not None else None,
            expected_version=target.version if target is not None else None,
            reason=REASON_TEXT[reason_code] if reason_code else None,
        ),
        reason_code,
    )


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


async def reconcile_source_candidates_with_judge(
    candidates: list[ExtractiveMemory],
    events: list[EpisodicEvent],
    targets: list[MemoryChunk],
    judge: Judge,
    *,
    config: JudgmentConfig | None = None,
    max_candidates_per_request: int = 10,
) -> JudgedReconciliation:
    config = config or JudgmentConfig()
    candidates, events, targets = _validate_inputs(candidates, events, targets)
    trace = JudgmentTrace()
    result = JudgedReconciliation(staged=[], trace=trace)

    normalized_target_ids: dict[str, list[str]] = {}
    for target in targets:
        normalized_target_ids.setdefault(normalize_memory_text(target.content), []).append(
            target.id
        )
    ambiguous_target_ids = {
        target_id for ids in normalized_target_ids.values() if len(ids) > 1 for target_id in ids
    }
    by_id = {target.id: target for target in targets}
    exact_matches = {
        index: normalized_target_ids.get(normalize_memory_text(candidate.content), [])
        for index, candidate in enumerate(candidates)
    }

    # Ask only what a decision can turn on.
    if targets:
        for batch in _batches(list(enumerate(candidates)), max_candidates_per_request):
            questions: dict[str, Choice] = {}
            state = {
                "candidates": [
                    {
                        "id": f"c{index}",
                        "role": candidate.source_role,
                        "kind": candidate.kind,
                        "text": candidate.content.strip(),
                    }
                    for index, candidate in batch
                ],
                "existing_facts": [
                    {"id": f"t{position}", "text": target.content}
                    for position, target in enumerate(targets)
                ],
            }
            aliases = {f"t{position}": target.id for position, target in enumerate(targets)}
            for index, candidate in batch:
                if candidate.kind != "correction_or_rejection":
                    for position, target in enumerate(targets):
                        if target.id in exact_matches[index]:
                            continue  # exact text needs no judgment
                        questions[f"relation:c{index}:t{position}"] = Choice(
                            f"How does candidate c{index} relate to existing fact t{position}?",
                            RELATION_OPTIONS,
                        )
                if _is_user_revision(candidate):
                    referent_options: list[tuple[str, Description]] = [
                        (
                            "none",
                            "none of the existing facts is the one being corrected or replaced",
                        )
                    ]
                    referent_options.extend(
                        (f"t{position}", {"text": target.content})
                        for position, target in enumerate(targets)
                        if target.id not in ambiguous_target_ids
                    )
                    if len(referent_options) > 1:
                        questions[f"referent:c{index}"] = Choice(
                            f"Which existing fact does candidate c{index} correct, replace, or reject?",
                            tuple(referent_options),
                        )
            if not questions:
                continue
            judgment = await judge.ask(state, questions, purpose="reconcile_candidates")
            trace.record("reconcile_candidates", judgment, len(questions))
            for key, answer in judgment.answers.items():
                assert isinstance(answer, ChoiceAnswer)
                if key.startswith("relation:"):
                    _, candidate_key, alias = key.split(":")
                    result.relations[(int(candidate_key[1:]), aliases[alias])] = answer
                else:
                    result.referents[int(key.split(":c")[1])] = answer

    decisions: list[ReconciliationDecision] = []
    codes: dict[int, str | None] = {}
    approved: set[tuple[int, str]] = set()
    for index, candidate in enumerate(candidates):
        decision, code = _decide(
            index,
            candidate,
            targets,
            by_id=by_id,
            ambiguous=ambiguous_target_ids,
            exact=exact_matches[index],
            relations=result.relations,
            referent=result.referents.get(index),
            aliases={f"t{position}": target.id for position, target in enumerate(targets)},
            config=config,
            approved=approved,
        )
        decisions.append(decision)
        codes[index] = code
    staged = _validate_decisions(
        decisions, candidates, targets, approved_paraphrase_support=frozenset(approved)
    )
    for item in staged:
        item.reason_code = codes.get(item.candidate_index)
    result.staged = staged
    return result


def _decide(
    index: int,
    candidate: ExtractiveMemory,
    targets: list[MemoryChunk],
    *,
    by_id: dict[str, MemoryChunk],
    ambiguous: set[str],
    exact: list[str],
    relations: dict[tuple[int, str], ChoiceAnswer],
    referent: ChoiceAnswer | None,
    aliases: dict[str, str],
    config: JudgmentConfig,
    approved: set[tuple[int, str]],
) -> tuple[ReconciliationDecision, str | None]:
    is_user = candidate.source_role == "user"
    revision = _is_user_revision(candidate)
    margin = 0.1

    if candidate.kind == "correction_or_rejection":
        if not targets:
            return _decision(index, "unresolved", reason_code="no_target_for_correction")
        return _referent_decision(index, referent, by_id, aliases, config, margin)

    same: list[tuple[str, float]] = []
    different: list[tuple[str, float]] = []
    for target in targets:
        if target.id in exact:
            same.append((target.id, 1.0))
            continue
        answer = relations.get((index, target.id))
        if answer is None:
            continue
        p_same = answer.probabilities.get("same_information", 0.0)
        p_diff = answer.probabilities.get("different_on_same_subject", 0.0)
        if answer.choice == "same_information":
            same.append((target.id, p_same))
        elif answer.choice == "different_on_same_subject":
            different.append((target.id, p_diff))

    confident_same = [(tid, p) for tid, p in same if p >= config.support_min_probability]
    near_same = [
        (tid, p)
        for tid, p in same
        if config.support_min_probability - margin <= p < config.support_min_probability
    ]
    if is_user:
        if len(confident_same) == 1 and not revision:
            target_id, _ = confident_same[0]
            if target_id in ambiguous:
                return _decision(index, "unresolved", reason_code="multiple_matching_targets")
            if target_id not in exact:
                approved.add((index, target_id))
            return _decision(index, "support", target=by_id[target_id])
        if len(confident_same) > 1:
            return _decision(index, "unresolved", reason_code="multiple_matching_targets")
        if near_same and not confident_same:
            return _decision(index, "unresolved", reason_code="low_confidence")

    confident_diff = [(tid, p) for tid, p in different if p >= config.conflict_min_probability]
    near_diff = [
        (tid, p)
        for tid, p in different
        if config.conflict_min_probability - margin <= p < config.conflict_min_probability
    ]
    if revision:
        if confident_diff or referent is not None:
            return _referent_decision(index, referent, by_id, aliases, config, margin)
        if near_diff:
            return _decision(index, "unresolved", reason_code="low_confidence")
        return _decision(index, "add")
    if confident_diff:
        target_id, _ = max(confident_diff, key=lambda item: item[1])
        conflict_target: MemoryChunk | None = None if target_id in ambiguous else by_id[target_id]
        return _decision(
            index, "conflict", target=conflict_target, reason_code="contradicts_target"
        )
    if near_diff:
        return _decision(index, "unresolved", reason_code="low_confidence")
    if is_user and exact:
        # Exact duplicate of an ambiguous pair; the schema path refused add here too.
        return _decision(index, "unresolved", reason_code="multiple_matching_targets")
    return _decision(index, "add")


def _referent_decision(
    index: int,
    referent: ChoiceAnswer | None,
    by_id: dict[str, MemoryChunk],
    aliases: dict[str, str],
    config: JudgmentConfig,
    margin: float,
) -> tuple[ReconciliationDecision, str | None]:
    if referent is None or referent.choice == "none":
        return _decision(index, "unresolved", reason_code="referent_not_found")
    probability = referent.probabilities.get(referent.choice, 1.0)
    if probability < config.conflict_min_probability - margin:
        return _decision(index, "unresolved", reason_code="ambiguous_revision_target")
    if probability < config.conflict_min_probability:
        return _decision(index, "unresolved", reason_code="low_confidence")
    return _decision(index, "correct", target=by_id[aliases[referent.choice]])
