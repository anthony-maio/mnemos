"""A complete revised rule can be retained without fabricating an old target."""

from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from benchmarks.local_memory.source_reconciliation import (
    build_reconciliation_schema,
    reconcile_source_candidates,
)
from tests.test_source_reconciliation import Provider, candidate, decision, event, target


@pytest.mark.asyncio
@pytest.mark.parametrize("has_target", [False, True])
async def test_complete_revised_requirement_adds_or_corrects_observed_target(
    has_target: bool,
) -> None:
    source = event(
        "revised-rule", "Replace the earlier export rule: future exports must exclude account IDs."
    )
    requirement = replace(
        candidate(source, "continuing_requirement"), revises_existing_requirement=True
    )
    old = target(content="Future exports may include account IDs.")
    targets = [old] if has_target else []
    response: dict[str, object] = {
        "decisions": [
            (
                decision(0, "correct", target_fact_id=old.id, expected_version=old.version)
                if has_target
                else decision(0, "add")
            )
        ]
    }
    assert Draft202012Validator(build_reconciliation_schema([requirement], targets)).is_valid(
        response
    )
    result = await reconcile_source_candidates([requirement], [source], targets, Provider(response))
    assert result[0].content == source.content
    assert result[0].source_kind == "continuing_requirement"
    assert result[0].revises_existing_requirement is True
    assert result[0].operation == ("correct" if has_target else "add")
    assert result[0].target_fact_id == (old.id if has_target else None)


@pytest.mark.asyncio
async def test_unflagged_requirement_cannot_authorize_correction() -> None:
    source = event("rule", "Future exports must exclude account IDs.")
    response: dict[str, object] = {
        "decisions": [decision(0, "correct", target_fact_id="fact-1", expected_version=1)]
    }
    assert not Draft202012Validator(
        build_reconciliation_schema([candidate(source, "continuing_requirement")], [target()])
    ).is_valid(response)
    with pytest.raises(ValueError, match="Only user correction"):
        await reconcile_source_candidates(
            [candidate(source, "continuing_requirement")], [source], [target()], Provider(response)
        )


@pytest.mark.asyncio
async def test_revised_requirement_cannot_choose_ambiguous_target_or_conflict() -> None:
    source = event("rule", "Replace the old rule: future exports must exclude account IDs.")
    requirement = replace(
        candidate(source, "continuing_requirement"), revises_existing_requirement=True
    )
    targets = [target("first", "Exports include IDs."), target("second", "Exports include IDs.")]
    for operation, error in (
        ("correct", "ambiguous exact normalized"),
        ("conflict", "requires a target or unresolved"),
    ):
        response: dict[str, object] = {
            "decisions": [
                decision(
                    0, operation, target_fact_id="first", expected_version=1, reason="A revision."
                )
            ]
        }
        assert not Draft202012Validator(
            build_reconciliation_schema([requirement], targets)
        ).is_valid(response)
        with pytest.raises(ValueError, match=error):
            await reconcile_source_candidates([requirement], [source], targets, Provider(response))


@pytest.mark.asyncio
async def test_forged_revision_flag_on_assistant_or_fact_fails_before_provider() -> None:
    for role, kind in (("assistant", "attributed_history"), ("user", "asserted_fact")):
        source = event("source", "Exports include IDs.", role)
        forged = replace(candidate(source, kind), revises_existing_requirement=True)
        provider = Provider(
            {"decisions": [decision(0, "correct", target_fact_id="fact-1", expected_version=1)]}
        )
        with pytest.raises(ValueError):
            await reconcile_source_candidates([forged], [source], [target()], provider)
        assert provider.calls == []
