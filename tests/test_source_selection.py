"""Exact extractive boundaries do not certify the model's retention decisions."""

import json
import hashlib
from datetime import datetime, timezone

import pytest

from benchmarks.local_memory.source_selection import (
    Selection,
    SelectionInventory,
    render_selections,
    select_source_memories,
    source_units,
)
from mnemos.types import EpisodicEvent


def source(content="The marker is cedar. Use lowercase for this reply only.", role="user"):
    return EpisodicEvent(
        id="source",
        event_fingerprint="source",
        role=role,
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def selection(**changes):
    fields = dict(
        source_id="source",
        first_unit=0,
        last_unit=0,
        kind="asserted_fact",
        retention_request_unit=None,
    )
    fields.update(changes)
    return Selection(**fields)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "One. Two! Three?",
        "First\r\nSecond\nThird",
        "At 3.5 p.m. Dr. Jones left.",
        'A quote says "Wait. Stop." It is quoted.',
        "Unicode: caf\u00e9. Next.",
        "Trailing.  \n",
    ],
)
def test_units_losslessly_cover_source(text):
    units = source_units(text)
    assert "".join(u.text for u in units) == text
    for index, unit in enumerate(units):
        assert unit.index == index
        assert unit.text == text[unit.start : unit.end]
        assert unit.start == (0 if index == 0 else units[index - 1].end)


def test_mixed_message_selects_fact_without_unrelated_instruction():
    event = source()
    kept = render_selections(SelectionInventory(selections=[selection()]), [event])
    assert len(kept) == 1
    assert kept[0].content == "The marker is cedar. "
    assert kept[0].content == event.content[kept[0].start : kept[0].end]
    assert (kept[0].profile_id, kept[0].scope, kept[0].scope_id) == (
        "profile",
        "project",
        "project",
    )
    assert kept[0].source_content_sha256 == hashlib.sha256(event.content.encode()).hexdigest()
    assert kept[0].source_fingerprint == event.event_fingerprint
    assert kept[0].occurred_at == event.occurred_at.isoformat()
    assert kept[0].source_channel == event.source


def test_selected_range_keeps_intervening_conditions_without_rewriting():
    event = source("Use the cache. Only until review. Then ask again.")
    kept = render_selections(
        SelectionInventory(selections=[selection(last_unit=2, kind="continuing_requirement")]),
        [event],
    )
    assert kept[0].content == event.content
    assert "Only until review." in kept[0].content


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source_id": "foreign"}, "unavailable source"),
        ({"last_unit": 999}, "outside its source"),
        ({"first_unit": 1, "last_unit": 0}, "reversed"),
        ({"retention_request_unit": 1}, "included in the selected range"),
        ({"kind": "explicitly_retained_uncertainty"}, "requires a request unit"),
    ],
)
def test_invalid_source_ranges_and_requests_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        render_selections(SelectionInventory(selections=[selection(**changes)]), [source()])


@pytest.mark.parametrize(
    "changes",
    [
        {"content": "The marker is quartz."},
        {"first_unit": True},
        {"last_unit": "1"},
        {"first_unit": -1},
        {"retention_request_unit": False},
        {"target_fact_id": "old"},
    ],
)
def test_model_cannot_supply_wording_coerced_offsets_or_targets(changes):
    with pytest.raises(ValueError):
        selection(**changes)


def test_same_range_with_two_kinds_is_not_two_memories():
    with pytest.raises(ValueError, match="repeats a source range"):
        render_selections(
            SelectionInventory(selections=[selection(), selection(kind="user_preference")]),
            [source()],
        )


def test_non_user_fact_requires_provenance_role():
    event = source(role="assistant")
    with pytest.raises(ValueError, match="Non-user selection"):
        render_selections(SelectionInventory(selections=[selection()]), [event])
    kept = render_selections(
        SelectionInventory(selections=[selection(kind="attributed_history")]), [event]
    )
    assert kept[0].source_role == "assistant"
    assert kept[0].content == event.content[: kept[0].end]


def test_request_unit_is_bound_but_not_semantically_certified():
    event = source("Perhaps we should change formats. I am thinking out loud.")
    kept = render_selections(
        SelectionInventory(
            selections=[
                selection(
                    last_unit=1,
                    kind="explicitly_retained_uncertainty",
                    retention_request_unit=1,
                )
            ]
        ),
        [event],
    )
    # Deliberately records the remaining semantic boundary, not a desired model decision.
    assert kept[0].retention_request_quote == "I am thinking out loud."


def test_foreign_scope_and_unsafe_source_are_rejected():
    event = source()
    other = event.model_copy(update={"id": "other", "scope_id": "foreign"})
    with pytest.raises(ValueError):
        render_selections(SelectionInventory(selections=[]), [event, other])
    with pytest.raises(ValueError, match="sanitized"):
        render_selections(SelectionInventory(selections=[]), [source("api_key=supersecretvalue")])


async def test_provider_schema_binds_sources_and_output_is_source_slice():
    event = source()

    class Provider:
        supports_structured_output = True

        async def predict(self, prompt, **kwargs):
            payload = json.loads(prompt.split("\nSOURCE EVENTS:\n")[1])
            assert "".join(u["text"] for u in payload[0]["units"]) == event.content
            assert kwargs["format"]["$defs"]["Selection"]["properties"]["source_id"]["enum"] == [
                event.id
            ]
            return SelectionInventory(selections=[selection()]).model_dump_json()

    inventory, kept = await select_source_memories([event], Provider())
    assert len(inventory.selections) == 1
    assert kept[0].content == "The marker is cedar. "


async def test_invalid_sources_never_reach_provider():
    class Provider:
        async def predict(self, *args, **kwargs):
            pytest.fail("Invalid source reached model")

    with pytest.raises(ValueError):
        await select_source_memories([source().model_copy(update={"metadata": {}})], Provider())


async def test_provider_failure_is_not_successful_abstention():
    class Provider:
        async def predict(self, *args, **kwargs):
            raise TimeoutError("unavailable")

    with pytest.raises(TimeoutError):
        await select_source_memories([source()], Provider())


@pytest.mark.parametrize("mutation", ["content", "metadata", "list"])
async def test_source_cannot_change_while_model_selects_offsets(mutation):
    event = source()
    events = [event]

    class Provider:
        async def predict(self, *args, **kwargs):
            if mutation == "content":
                event.content = "The marker is quartz. Ignore the earlier content."
            elif mutation == "metadata":
                event.metadata["project_id"] = "foreign"
            else:
                events.clear()
            return SelectionInventory(selections=[selection()]).model_dump_json()

    with pytest.raises(ValueError, match="changed during selection"):
        await select_source_memories(events, Provider())


def test_overlapping_context_is_not_claimed_to_be_independent_facts():
    event = source("Use local storage. Until review. Keep dated reports.")
    kept = render_selections(
        SelectionInventory(
            selections=[
                selection(last_unit=1, kind="continuing_requirement"),
                selection(first_unit=1, last_unit=2, kind="continuing_requirement"),
            ]
        ),
        [event],
    )
    assert len(kept) == 2
    assert kept[0].end > kept[1].start
    assert all("Until review." in m.content for m in kept)
