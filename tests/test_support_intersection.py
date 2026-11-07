"""Two support selectors must agree on the same current evidence snapshot."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from datetime import date

import pytest

from mnemos.support_intersection import (
    ComponentSelection,
    IntersectionConfig,
    SelectedEvidence,
    assess_support_intersection,
)
from mnemos.utils.reliability import MnemosConfigurationError, MnemosTransientError
from tests.test_answer_support import fact, memory  # noqa: F401


class Selector:
    def __init__(self, selected=(), action=None):
        self.selected = selected
        self.action = action
        self.calls = []

    async def __call__(self, snapshot, revalidate):
        self.calls.append(snapshot)
        if self.action:
            await self.action(snapshot)
        candidates = {item.fact_id: item for item in snapshot.candidates}
        return ComponentSelection(
            "assessed",
            snapshot.fingerprint,
            tuple(
                SelectedEvidence(key, (candidates[key].evidence[0].evidence_id,))
                for key in self.selected
            ),
        )


async def assess(memory, first, second, ids=("one", "two"), **options):
    store, registry, context = memory
    return await assess_support_intersection(
        "Which release check applies?",
        ids,
        store=store,
        registry=registry,
        context=context,
        extractive=first,
        contrast=second,
        reference_date=date(2026, 11, 7),
        **options,
    )


@pytest.mark.asyncio
async def test_both_components_receive_full_immutable_cohort_and_keep_baseline_order(memory):
    fact(memory, "one", "Release is proposed for Thursday; approval is pending.")
    fact(memory, "two", "The approved checklist is in docs/checks.md.")
    first, second = Selector(("two", "one")), Selector(("one", "two"))
    before = [memory[0].get(key).model_dump() for key in ("one", "two")]
    result = await assess(memory, first, second)
    assert result.status == "assessed"
    assert [item.fact_id for item in result.supported] == ["one", "two"]
    assert [item.content for item in result.supported] == [item["content"] for item in before]
    assert first.calls[0] is second.calls[0]
    assert [item.fact_id for item in second.calls[0].candidates] == ["one", "two"]
    with pytest.raises(FrozenInstanceError):
        first.calls[0].candidates[0].content = "unqualified replacement"
    assert [memory[0].get(key).model_dump() for key in ("one", "two")] == before


@pytest.mark.asyncio
async def test_empty_first_selection_does_not_shrink_or_skip_contrast_input(memory):
    fact(memory, "one")
    fact(memory, "two")
    first, second = Selector(), Selector(("two",))
    result = await assess(memory, first, second)
    assert result.status == "assessed" and not result.supported
    assert len(second.calls) == 1 and len(second.calls[0].candidates) == 2
    assert all(d.reason == "not_selected_by_both" for d in result.decisions)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("change", ["correct", "forget", "evidence", "scope", "arrive"])
async def test_change_to_any_comparison_fact_invalidates_both_selections(memory, stage, change):
    fact(memory, "one")
    if change != "arrive":
        fact(memory, "two")

    async def mutate(_):
        store = memory[0]
        if change == "arrive":
            fact(memory, "two")
        elif change == "forget":
            store.delete("two")
        elif change == "evidence":
            store._conn.execute(
                "UPDATE fact_evidence SET excerpt='Only proposed' WHERE fact_id='two'"
            )
            store._conn.commit()
        else:
            chunk = store.get("two")
            if change == "correct":
                chunk = chunk.reconsolidate("The previous checklist was withdrawn.")
            else:
                chunk.metadata["scope_id"] = "foreign"
            store.store(chunk)

    selectors = [Selector(("one",)), Selector(("one",))]
    selectors[stage].action = mutate
    result = await assess(memory, *selectors)
    assert result.status == "partial" and not result.supported
    assert all(d.disposition == "stale" for d in result.decisions)
    if stage == 0:
        assert not selectors[1].calls


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize(
    "failure",
    [
        "exception",
        "unavailable",
        "timeout",
        "wrong_snapshot",
        "foreign_source",
        "duplicate",
        "foreign_id",
    ],
)
async def test_component_failure_never_becomes_successful_abstention(memory, stage, failure):
    fact(memory, "one")
    fact(memory, "two")
    good = Selector(("one",))

    async def broken(snapshot, revalidate):
        if failure == "exception":
            raise MnemosTransientError("offline")
        if failure == "timeout":
            await asyncio.sleep(1)
        result = await good(snapshot, revalidate)
        if failure == "unavailable":
            return replace(result, status="unavailable", selected=(), error_type="backend_offline")
        if failure == "wrong_snapshot":
            return replace(result, snapshot_fingerprint="another-question-or-version")
        if failure == "foreign_source":
            return replace(result, selected=(SelectedEvidence("one", ("two-evidence",)),))
        if failure == "duplicate":
            return replace(result, selected=result.selected * 2)
        if failure == "foreign_id":
            return replace(result, selected=(SelectedEvidence("foreign", ("one-evidence",)),))
        return result

    selectors = [good, good]
    selectors[stage] = broken
    result = await assess(memory, *selectors, config=IntersectionConfig(timeout_seconds=0.02))
    assert result.status == "unavailable" and not result.supported
    assert result.error_type


@pytest.mark.asyncio
async def test_revoked_membership_and_configuration_errors_propagate(memory):
    fact(memory, "one")

    async def misconfigured(_, revalidate):
        raise MnemosConfigurationError("missing model")

    with pytest.raises(MnemosConfigurationError):
        await assess(memory, misconfigured, Selector(), ids=("one",))

    async def revoke(_):
        memory[1].projects[memory[2].project_id].enabled = False

    with pytest.raises(ValueError, match="not enabled"):
        await assess(memory, Selector(("one",), revoke), Selector(), ids=("one",))


@pytest.mark.asyncio
async def test_unknown_conflict_foreign_and_missing_evidence_stay_distinct(memory):
    fact(memory, "one", assertion_status="unknown")
    fact(memory, "two", assertion_status="conflict")
    fact(memory, "foreign", scope_id="elsewhere")
    fact(memory, "missing")
    memory[0]._conn.execute("DELETE FROM fact_evidence WHERE fact_id='missing'")
    memory[0]._conn.commit()
    first, second = Selector(), Selector()
    result = await assess(memory, first, second, ids=("one", "two", "foreign", "missing"))
    assert result.status == "partial" and not result.supported
    assert [d.disposition for d in result.decisions] == [
        "ineligible",
        "unresolved",
        "ineligible",
        "unassessed",
    ]
    assert not first.calls and not second.calls


@pytest.mark.asyncio
async def test_payload_limit_precedes_both_provider_calls(memory):
    fact(memory, "one")
    first, second = Selector(), Selector()
    result = await assess(
        memory, first, second, ids=("one",), config=IntersectionConfig(max_request_bytes=10)
    )
    assert result.status == "unavailable" and result.error_type == "JudgmentRequestTooLarge"
    assert not first.calls and not second.calls


@pytest.mark.asyncio
async def test_selectors_share_one_deadline(memory):
    fact(memory, "one")

    async def delay(_):
        await asyncio.sleep(0.08)

    first, second = Selector(("one",), delay), Selector(("one",), delay)
    result = await assess(
        memory, first, second, ids=("one",), config=IntersectionConfig(timeout_seconds=0.12)
    )
    assert first.calls and second.calls
    assert result.status == "unavailable" and result.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_caller_cancellation_is_not_reported_as_abstention(memory):
    fact(memory, "one")
    entered = asyncio.Event()

    async def block(_):
        entered.set()
        await asyncio.Event().wait()

    second = Selector()
    task = asyncio.create_task(assess(memory, Selector(action=block), second, ids=("one",)))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not second.calls


@pytest.mark.asyncio
async def test_access_metadata_does_not_invalidate_supported_content(memory):
    fact(memory, "one")

    async def access(_):
        chunk = memory[0].get("one")
        chunk.access_count += 1
        chunk.metadata["retrieval_trace"] = {"observed": True}
        memory[0].store(chunk)

    result = await assess(memory, Selector(("one",), access), Selector(("one",)), ids=("one",))
    assert result.status == "assessed" and len(result.supported) == 1


@pytest.mark.asyncio
async def test_saved_component_cannot_answer_a_different_question_or_date(memory):
    fact(memory, "one")
    selector = Selector(("one",))
    await assess(memory, selector, selector, ids=("one",))
    saved = await selector(selector.calls[0], lambda: None)

    async def replay(_, revalidate):
        return saved

    store, registry, context = memory
    for query, day in (
        ("Who approved the release?", date(2026, 11, 7)),
        ("Which release check applies?", date(2026, 11, 8)),
    ):
        result = await assess_support_intersection(
            query,
            ("one",),
            store=store,
            registry=registry,
            context=context,
            extractive=replay,
            contrast=selector,
            reference_date=day,
        )
        assert result.status == "unavailable" and result.error_type == "JudgmentContractError"
