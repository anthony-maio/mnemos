"""Exhaustive accounting exposes silent omission, not wrong semantic decisions."""

from datetime import datetime, timezone

import pytest

from benchmarks.local_memory.source_coverage import (
    COVERAGE_PROMPT,
    classify_all_source_units,
)
from mnemos.types import EpisodicEvent


def event(content="The marker is cedar. Keep reports concise."):
    return EpisodicEvent(
        id="source",
        event_fingerprint="source",
        role="user",
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def selection(first, last, kind="asserted_fact"):
    return dict(source_id="source", first_unit=first, last_unit=last, kind=kind)


class Provider:
    supports_structured_output = True

    def __init__(self, selections):
        self.selections = selections

    async def predict(self, prompt, **kwargs):
        import json

        assert prompt.startswith(COVERAGE_PROMPT)
        assert "not classify every source unit" not in prompt
        assert kwargs["format"]["$defs"]["ClassifiedSelection"]["properties"]["source_id"][
            "enum"
        ] == ["source"]
        return json.dumps({"selections": self.selections})


@pytest.mark.asyncio
async def test_missing_marker_is_failure_not_valid_policy_only_result():
    with pytest.raises(ValueError, match="unaccounted"):
        await classify_all_source_units(
            [event()], Provider([selection(1, 1, "continuing_requirement")])
        )


@pytest.mark.asyncio
async def test_independent_fact_and_policy_both_survive():
    inventory, kept = await classify_all_source_units(
        [event()], Provider([selection(0, 0), selection(1, 1, "continuing_requirement")])
    )
    assert len(inventory.selections) == 2
    assert [m.content for m in kept] == ["The marker is cedar. ", "Keep reports concise."]


@pytest.mark.asyncio
async def test_explicit_discard_covers_source_without_retention():
    inventory, kept = await classify_all_source_units(
        [event("Which color?")], Provider([selection(0, 0, "question")])
    )
    assert len(inventory.selections) == 1 and kept == []


@pytest.mark.asyncio
async def test_covered_but_wrong_discard_still_needs_semantic_review():
    _, kept = await classify_all_source_units(
        [event()], Provider([selection(0, 1, "quoted_material")])
    )
    assert kept == []  # Wrong meaning can satisfy structural coverage.


@pytest.mark.asyncio
async def test_blank_only_units_do_not_force_meaningless_selections():
    _, kept = await classify_all_source_units(
        [event("\nThe marker is cedar.")], Provider([selection(1, 1)])
    )
    assert kept[0].content == "The marker is cedar."


@pytest.mark.asyncio
async def test_full_coverage_does_not_bypass_authority():
    source = event()
    source.role = "assistant"
    with pytest.raises(ValueError, match="Non-user"):
        await classify_all_source_units([source], Provider([selection(0, 1)]))


@pytest.mark.asyncio
async def test_full_coverage_does_not_bypass_mutation_guard():
    source = event()

    class Mutating(Provider):
        async def predict(self, prompt, **kwargs):
            source.content = "Changed."
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="changed during"):
        await classify_all_source_units([source], Mutating([selection(0, 1)]))
