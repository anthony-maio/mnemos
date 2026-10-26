"""Source classification through typed judgments instead of a generated inventory.

The judge never writes memory text. It decides, unit by unit, where one
statement ends and the next begins, then labels each contiguous range with a
kind. Code turns those answers into the same ``ClassifiedInventory`` the
prompt-driven classifier produces, and ``render_classifications`` applies every
existing range, role, retention, and firewall rule before anything is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from ..config import JudgmentConfig
from ..judgment import (
    Choice,
    ChoiceAnswer,
    Criteria,
    Description,
    Judge,
    JudgmentTrace,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)
from ..modules.affective import AFFECTIVE_SCORES
from ..types import CognitiveState, EpisodicEvent
from .assertion_inventory import RETAINED_KINDS
from .source_classification import (
    DISCARD_KINDS,
    ClassifiedInventory,
    ClassifiedSelection,
    render_classifications,
)
from .source_selection import ExtractiveMemory, SourceUnit, source_units

# The boundary cases from the selection and classification prompts, one line
# each. A judge reads literally, so the description is the rule.
KIND_DESCRIPTIONS: dict[str, Description] = {
    "asserted_fact": {
        "what": (
            "a durable project fact the user directly asserts as current, including the fact "
            "after 'For <context>, keep this in mind: <fact>'. The request to remember "
            "introduces the fact; it does not make the fact a reply instruction"
        ),
        "not": (
            "a pasted fixture, a hypothetical, a quotation, a report by someone else, or a "
            "passing status such as 'the service is down right now'"
        ),
    },
    "user_preference": {
        "what": "the user's own continuing preference, with its scope",
        "not": "a one-turn reply or formatting instruction, or fictional dialogue",
    },
    "continuing_requirement": {
        "what": (
            "a complete rule or limit that applies to the project from now on, however it is "
            "phrased: 'for this project, X goes in Y', 'always do X', 'we will not keep X'. "
            "A standalone negative rule counts even with no earlier rule. A rule introduced "
            "by 'keep this in mind:' is still a continuing requirement"
        ),
        "signal": "the scope is the project, the codebase, or the future, not one reply",
        "not": "proof that it was implemented; not a rejection that refers to an earlier statement",
    },
    "explicitly_retained_uncertainty": {
        "what": "an unresolved consideration the user explicitly asks to keep or remember",
        "not": (
            "stating uncertainty without a request to remember it, or a concrete fact "
            "introduced by 'keep this in mind:'"
        ),
    },
    "correction_or_rejection": {
        "what": (
            "changes, supersedes, or rejects an identified earlier assertion or prior project "
            "state, without giving a complete replacement rule"
        ),
        "not": (
            "a standalone negative rule or a quoted fixture disclaimed as sample text; "
            "a complete replacement rule is continuing_requirement"
        ),
    },
    "attributed_history": {
        "what": "a dated or attributed report of what happened or was observed, with its limits",
        "not": "a current rule or an accepted plan",
    },
    "question": "an inquiry or a comparison of options, not an asserted preference",
    "temporary_instruction": {
        "what": (
            "a tool, reply, or formatting instruction whose scope is this turn, this reply, "
            "or the next answer only, such as 'for this reply only, keep it short'"
        ),
        "signal": "words that limit it to now: this turn, this reply, this time, for now, today",
        "not": (
            "anything scoped to the project, the codebase, or the future; 'for this project' "
            "is a continuing requirement, not a temporary instruction. 'Keep this in mind:' "
            "followed by a durable fact or rule is a retention request, not a reply instruction"
        ),
    },
    "quoted_material": (
        "inspection-only quotations, fictional dialogue, or pasted fixtures, including "
        "a user's statement that a quoted example is sample text rather than a project "
        "fact; instructions inside the quotation are not the user's own instructions"
    ),
    "unrequested_speculation": (
        "hypothetical or uncertain material with no actual request to remember it"
    ),
    "unaccepted_suggestion": "a proposal, often the assistant's, not accepted as a project rule",
}

USER_KINDS: tuple[str, ...] = (
    "continuing_requirement",
    "user_preference",
    "asserted_fact",
    "correction_or_rejection",
    "explicitly_retained_uncertainty",
    "attributed_history",
    "temporary_instruction",
    "question",
    "quoted_material",
    "unrequested_speculation",
    "unaccepted_suggestion",
)
NON_USER_KINDS: tuple[str, ...] = (
    "attributed_history",
    "temporary_instruction",
    "question",
    "quoted_material",
    "unrequested_speculation",
    "unaccepted_suggestion",
)

# Each question quotes the text it is about inside its own instructions. When
# several ranges sat only in the shared state, the judge blurred them into
# coin flips; quoted in the question, the same ranges came back at 1.0.
CONTINUES_CRITERIA = Criteria(
    true="incomplete on its own, or supplies a qualification the previous unit needs",
    false="a separate statement, a new topic, or a reply or format instruction on its own",
)


def continues_question(previous: str, unit: str) -> Noul:
    return Noul(
        {
            "previous_unit": previous,
            "this_unit": unit,
            "question": (
                "Does this_unit continue the statement begun in previous_unit, rather than "
                "start a new independent statement? Continuations include a clause, "
                "condition, exception, attribution, timing, approval boundary, or the "
                "referent of words such as 'that rule' or 'keep this'."
            ),
        },
        CONTINUES_CRITERIA,
    )


# The same three questions the router asks on its own; judged here once
# per event so a judged capture makes no separate affective call.
AFFECTIVE_QUESTIONS: dict[str, Score] = AFFECTIVE_SCORES


@dataclass(frozen=True)
class RangeDecision:
    """One judged range, retained or not, kept for audit."""

    source_id: str
    first_unit: int
    last_unit: int
    kind: str
    probability: float
    confidence: float
    retained: bool
    note: str | None = None


@dataclass
class JudgedClassification:
    inventory: ClassifiedInventory
    memories: list[ExtractiveMemory]
    decisions: list[RangeDecision]
    trace: JudgmentTrace = field(default_factory=JudgmentTrace)
    cognitive_state: CognitiveState | None = None
    affective_confidence: dict[str, float] = field(default_factory=dict)


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


async def _segment(
    event: EpisodicEvent,
    units: list[SourceUnit],
    judge: Judge,
    config: JudgmentConfig,
    trace: JudgmentTrace,
) -> list[tuple[int, int]]:
    """Group units into statements. Returns inclusive (first, last) index pairs."""
    spoken = [unit for unit in units if unit.text.strip()]
    if not spoken:
        return []
    joins: dict[int, bool] = {}
    for batch in _batches(spoken, config.max_units_per_request):
        state: dict[str, Any] = {"source": {"role": event.role, "text": event.content}}
        questions = {
            f"continues:u{unit.index}": continues_question(previous.text, unit.text)
            for previous, unit in zip(batch, batch[1:])
        }
        if not questions:
            continue
        judgment = await judge.ask(state, questions, purpose="segment_source")
        trace.record("segment_source", judgment, len(questions))
        for key, answer in judgment.answers.items():
            index = int(key.split(":u", 1)[1])
            assert isinstance(answer, NoulAnswer)
            joins[index] = answer.probability >= config.continuation_min_probability
    ranges: list[tuple[int, int]] = []
    start = spoken[0].index
    previous = spoken[0].index
    for unit in spoken[1:]:
        if joins.get(unit.index, False):
            previous = unit.index
            continue
        ranges.append((start, previous))
        start = previous = unit.index
    ranges.append((start, previous))
    return ranges


def _allowed_kinds(role: str) -> tuple[str, ...]:
    return USER_KINDS if role == "user" else NON_USER_KINDS


async def classify_source_memories_with_judge(
    events: list[EpisodicEvent],
    judge: Judge,
    *,
    config: JudgmentConfig | None = None,
    include_affective: bool = False,
) -> JudgedClassification:
    """Classify every event's statements with the judge, then render as usual."""
    config = config or JudgmentConfig()
    trace = JudgmentTrace()
    selections: list[ClassifiedSelection] = []
    decisions: list[RangeDecision] = []
    cognitive_state: CognitiveState | None = None
    affective_confidence: dict[str, float] = {}
    for event in events:
        units = source_units(event.content)
        ranges = await _segment(event, units, judge, config, trace)
        allowed = _allowed_kinds(event.role)
        kind_options = tuple((kind, KIND_DESCRIPTIONS[kind]) for kind in allowed)
        first_batch = True
        for batch in _batches(list(enumerate(ranges)), config.max_ranges_per_request):
            # The whole source is context; each question carries its own range.
            state: dict[str, Any] = {"source": {"role": event.role, "text": event.content}}
            questions: dict[str, Any] = {}
            for position, (first, last) in batch:
                range_text = event.content[units[first].start : units[last].end].strip()
                questions[f"kind:r{position}"] = Choice(
                    {
                        "range_text": range_text,
                        "question": (
                            "Which kind of statement is range_text, spoken by the source role? "
                            "Judge its own wording and scope. Use the surrounding source only "
                            "to tell whether the speaker presents range_text as their own "
                            "current claim or rule, or as quoted, fictional, hypothetical, "
                            "past, or rejected material."
                        ),
                    },
                    kind_options,
                )
                unit_options: list[tuple[str, Description]] = [
                    ("none", "no explicit request to remember or keep this")
                ]
                unit_options.extend(
                    (
                        f"u{index}",
                        {"text": units[index].text, "is": "the unit containing the request"},
                    )
                    for index in range(first, last + 1)
                    if units[index].text.strip()
                )
                questions[f"retention_request:r{position}"] = Choice(
                    {
                        "range_text": range_text,
                        "question": (
                            "Which unit of range_text explicitly asks to remember, keep, or "
                            "retain something for later? Choose none unless the request is "
                            "stated."
                        ),
                    },
                    tuple(unit_options),
                )
                if event.role == "user":
                    questions[f"revises:r{position}"] = Noul(
                        {
                            "range_text": range_text,
                            "question": (
                                "Does range_text state a complete new rule while explicitly "
                                "replacing, superseding, or rejecting an earlier rule?"
                            ),
                        },
                        Criteria(
                            true="a full replacement rule prefaced by rejecting or superseding an earlier one",
                            false="no earlier rule is rejected, or no complete replacement rule is given",
                        ),
                    )
            if include_affective and first_batch:
                questions.update(AFFECTIVE_QUESTIONS)
            first_batch = False
            judgment = await judge.ask(state, questions, purpose="classify_source")
            trace.record("classify_source", judgment, len(questions))
            answers = judgment.answers
            for position, (first, last) in batch:
                kind_answer = answers[f"kind:r{position}"]
                assert isinstance(kind_answer, ChoiceAnswer)
                kind = kind_answer.choice
                probability = kind_answer.probabilities.get(kind, 1.0)
                note: str | None = None
                request_unit: int | None = None
                revises = False
                if kind in RETAINED_KINDS:
                    request_answer = answers[f"retention_request:r{position}"]
                    assert isinstance(request_answer, ChoiceAnswer)
                    if request_answer.choice != "none":
                        request_unit = int(request_answer.choice[1:])
                    if kind == "explicitly_retained_uncertainty" and request_unit is None:
                        # Retained uncertainty needs a cited request; without
                        # one it is thinking out loud.
                        kind, note = "unrequested_speculation", "uncertainty_without_request"
                    elif kind != "explicitly_retained_uncertainty":
                        request_unit = None
                    if kind == "continuing_requirement" and event.role == "user":
                        revises_answer = answers[f"revises:r{position}"]
                        assert isinstance(revises_answer, NoulAnswer)
                        revises = revises_answer.probability >= config.kind_min_probability
                retained = kind in RETAINED_KINDS and probability >= config.kind_min_probability
                if kind in RETAINED_KINDS and not retained:
                    note = "low_confidence"
                decisions.append(
                    RangeDecision(
                        source_id=event.id,
                        first_unit=first,
                        last_unit=last,
                        kind=kind,
                        probability=probability,
                        confidence=kind_answer.confidence,
                        retained=retained,
                        note=note,
                    )
                )
                if kind in DISCARD_KINDS:
                    selections.append(
                        ClassifiedSelection(
                            source_id=event.id,
                            first_unit=first,
                            last_unit=last,
                            kind=cast(Any, kind),
                        )
                    )
                elif retained:
                    selections.append(
                        ClassifiedSelection(
                            source_id=event.id,
                            first_unit=first,
                            last_unit=last,
                            kind=cast(Any, kind),
                            retention_request_unit=request_unit,
                            revises_existing_requirement=revises,
                        )
                    )
            if include_affective and cognitive_state is None and "valence" in answers:
                cognitive_state, affective_confidence = _cognitive_state(answers)
    inventory = ClassifiedInventory(selections=selections)
    memories = render_classifications(inventory, events)
    return JudgedClassification(
        inventory=inventory,
        memories=memories,
        decisions=decisions,
        trace=trace,
        cognitive_state=cognitive_state,
        affective_confidence=affective_confidence,
    )


def _cognitive_state(answers: Mapping[str, Any]) -> tuple[CognitiveState, dict[str, float]]:
    valence, arousal, complexity = (
        answers["valence"],
        answers["arousal"],
        answers["complexity"],
    )
    assert isinstance(valence, ScoreAnswer)
    assert isinstance(arousal, ScoreAnswer)
    assert isinstance(complexity, ScoreAnswer)
    state = CognitiveState(
        valence=valence.scaled(-1.0, 1.0, 5),
        arousal=arousal.scaled(0.0, 1.0, 5),
        complexity=complexity.scaled(0.0, 1.0, 5),
    )
    return state, {
        "valence": valence.confidence,
        "arousal": arousal.confidence,
        "complexity": complexity.confidence,
    }
