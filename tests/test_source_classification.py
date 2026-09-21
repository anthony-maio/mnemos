"""Discard labels never bypass provenance checks or confer semantic certainty."""

from datetime import datetime, timezone
import json

import pytest

from benchmarks.local_memory.source_classification import (
    ClassifiedInventory,
    ClassifiedSelection,
    DISCARD_KINDS,
    classify_source_memories,
    render_classifications,
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


def item(**changes):
    fields = dict(source_id="source", first_unit=0, last_unit=0, kind="asserted_fact")
    fields.update(changes)
    return ClassifiedSelection(**fields)


def render(items, events=None):
    return render_classifications(ClassifiedInventory(selections=items), events or [source()])


def test_mixed_message_preserves_fact_and_auditable_discard():
    inventory = ClassifiedInventory(
        selections=[item(), item(first_unit=1, last_unit=1, kind="temporary_instruction")]
    )
    before = inventory.model_dump()
    kept = render_classifications(inventory, [source()])
    assert [m.content for m in kept] == ["The marker is cedar. "]
    assert inventory.model_dump() == before


@pytest.mark.parametrize("kind", sorted(DISCARD_KINDS))
def test_discarded_material_never_emits_a_memory(kind):
    assert render([item(kind=kind)], [source("Should we use cedar?", role="assistant")]) == []


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source_id": "foreign"}, "unavailable"),
        ({"last_unit": 99}, "outside"),
        ({"first_unit": 1, "last_unit": 0}, "reversed"),
        ({"retention_request_unit": 0}, "cannot cite"),
    ],
)
def test_discard_is_not_a_validation_bypass(changes, message):
    with pytest.raises(ValueError, match=message):
        render([item(kind="question", **changes)])


def test_conflicting_keep_discard_range_rejected():
    with pytest.raises(ValueError, match="repeats"):
        render([item(), item(kind="question")])


def test_overlapping_discard_does_not_cancel_durable_requirement():
    kept = render(
        [item(kind="continuing_requirement"), item(last_unit=1, kind="quoted_material")],
        [source('Keep the manuscript rule. The character says "Wait."')],
    )
    assert [m.content for m in kept] == ["Keep the manuscript rule. "]


def test_retained_nonuser_cannot_gain_direct_authority():
    with pytest.raises(ValueError, match="Non-user"):
        render([item()], [source(role="assistant")])


def test_uncertainty_still_requires_source_bound_request():
    with pytest.raises(ValueError, match="requires a request"):
        render([item(kind="explicitly_retained_uncertainty")])
    with pytest.raises(ValueError, match="included"):
        render([item(kind="explicitly_retained_uncertainty", retention_request_unit=1)])


def test_wrong_semantic_label_is_not_mechanically_detected():
    kept = render([item(kind="user_preference")], [source("Which color should we pick?")])
    assert kept[0].content == "Which color should we pick?"


@pytest.mark.asyncio
async def test_source_mutation_during_classification_rejected():
    event = source()

    class Provider:
        async def predict(self, prompt, **kwargs):
            event.metadata["user_profile_id"] = "other"
            return '{"selections": []}'

    with pytest.raises(ValueError, match="changed during"):
        await classify_source_memories([event], Provider())


@pytest.mark.asyncio
async def test_schema_binds_source_and_records_discard_without_emission():
    class Provider:
        supports_structured_output = True

        async def predict(self, prompt, **kwargs):
            schema = kwargs["format"]["$defs"]["ClassifiedSelection"]["properties"]
            assert schema["source_id"]["enum"] == ["source"]
            assert "question" in json.dumps(schema["kind"])
            return ClassifiedInventory(selections=[item(kind="question")]).model_dump_json()

    inventory, kept = await classify_source_memories([source("Which color?")], Provider())
    assert len(inventory.selections) == 1 and kept == []


@pytest.mark.asyncio
async def test_invalid_source_cannot_reach_model_even_for_discard():
    class Provider:
        async def predict(self, *args, **kwargs):
            pytest.fail("Invalid source reached model")

    event = source()
    event.metadata["project_id"] = "foreign"
    with pytest.raises(ValueError, match="contradicts"):
        await classify_source_memories([event], Provider())


@pytest.mark.asyncio
async def test_provider_failure_is_not_successful_discard():
    class Provider:
        async def predict(self, *args, **kwargs):
            raise TimeoutError("unavailable")

    with pytest.raises(TimeoutError):
        await classify_source_memories([source()], Provider())


def test_scope_partition_validation_applies_to_all_discarded_input():
    other = source()
    other.id = "other"
    other.scope_id = "foreign"
    other.metadata["project_id"] = "foreign"
    with pytest.raises(ValueError, match="scope partitions"):
        render([item(kind="question")], [source(), other])


@pytest.mark.parametrize("mutation", ["append", "remove", "replace"])
@pytest.mark.asyncio
async def test_event_list_change_during_provider_call_rejected(mutation):
    events = [source()]

    class Provider:
        async def predict(self, *args, **kwargs):
            if mutation == "append":
                events.append(source())
            elif mutation == "remove":
                events.clear()
            else:
                events[0] = source("A replacement event.")
            return '{"selections": []}'

    with pytest.raises(ValueError, match="changed during"):
        await classify_source_memories(events, Provider())


@pytest.mark.parametrize(
    "content",
    [
        "The test report from January 1 says restart passed; outages were not tested.",
        "Restart always works.",
    ],
)
def test_nonuser_history_preserves_provenance_without_certifying_claim(content):
    event = source(content, role="assistant")
    kept = render([item(kind="attributed_history")], [event])
    assert kept[0].content == content
    assert kept[0].source_role == "assistant"
    assert kept[0].occurred_at == event.occurred_at.isoformat()
    # Both pass structurally. The second lacks report qualifications; semantic
    # review must not mistake retained provenance for independent verification.
