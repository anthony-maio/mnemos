"""Write-free source candidate to target reconciliation contract tests."""

from datetime import datetime, timezone
import hashlib
import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from benchmarks.local_memory.source_reconciliation import (
    ReconciliationResponse,
    StagedReconciliation,
    build_reconciliation_schema,
    reconcile_source_candidates,
)
from benchmarks.local_memory.source_selection import ExtractiveMemory, source_units
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider


def event(event_id: str, content: str, role: str = "user") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role=role,
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project-1",
        source="manual",
        metadata={"user_profile_id": "profile-1", "project_id": "project-1"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def candidate(
    source: EpisodicEvent,
    kind: str = "asserted_fact",
    retention_request_quote: str | None = None,
) -> ExtractiveMemory:
    unit = source_units(source.content)[0]
    return ExtractiveMemory(
        source_id=source.id,
        source_role=source.role,
        source_fingerprint=source.event_fingerprint,
        source_content_sha256=hashlib.sha256(source.content.encode()).hexdigest(),
        source_channel=source.source,
        occurred_at=source.occurred_at.isoformat(),
        profile_id="profile-1",
        scope=source.scope,
        scope_id=source.scope_id,
        start=unit.start,
        end=unit.end,
        content=unit.text,
        kind=kind,  # type: ignore[arg-type]
        retention_request_quote=retention_request_quote,
    )


def target(fact_id: str = "fact-1", content: str = "The marker is cedar.") -> MemoryChunk:
    return MemoryChunk(
        id=fact_id,
        content=content,
        version=1,
        metadata={
            "user_profile_id": "profile-1",
            "scope": "project",
            "scope_id": "project-1",
        },
    )


class Provider(LLMProvider):
    supports_structured_output = True

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def predict(self, prompt: str, **kwargs: object) -> str:
        self.calls.append((prompt, dict(kwargs)))
        return json.dumps(self.response)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        _ = prompt, labels
        return {}


def decision(index: int, operation: str, **extra: object) -> dict[str, object]:
    result: dict[str, object] = {"candidate_index": index, "operation": operation}
    result.update(extra)
    return result


@pytest.mark.asyncio
async def test_allowed_correction_is_staged_with_exact_source_content() -> None:
    source = event("correction-source", "The marker is cedar.")
    candidate_value = candidate(source, "correction_or_rejection")
    provider = Provider(
        {
            "decisions": [
                decision(
                    0,
                    "correct",
                    target_fact_id="fact-1",
                    expected_version=1,
                )
            ]
        }
    )

    staged = await reconcile_source_candidates([candidate_value], [source], [target()], provider)

    assert staged == [
        StagedReconciliation(
            candidate_index=0,
            source_id=source.id,
            source_role="user",
            source_kind="correction_or_rejection",
            content=source.content,
            operation="correct",
            target_fact_id="fact-1",
            expected_version=1,
            reason=None,
        )
    ]
    assert (
        "content"
        not in provider.calls[0][1]["format"]["$defs"]["ReconciliationDecision"]["properties"]
    )
    correction_branches = [
        branch
        for branch in provider.calls[0][1]["format"]["properties"]["decisions"]["items"]["oneOf"]
        if branch["properties"]["operation"]["enum"] == ["correct"]
    ]
    assert len(correction_branches) == 1
    assert correction_branches[0]["properties"]["target_fact_id"]["enum"] == ["fact-1"]
    assert correction_branches[0]["properties"]["expected_version"]["enum"] == [1]


@pytest.mark.asyncio
async def test_correction_rejects_ambiguous_exact_targets_but_keeps_unique_and_add() -> None:
    correction_source = event("ambiguous-correction", "Correction: the marker is quartz.")
    correction = candidate(correction_source, "correction_or_rejection")
    duplicate_a = target("fact-old-a", "The marker is cedar.")
    duplicate_b = target("fact-old-b", "  The marker is CEDAR. ")
    unique = target("fact-unrelated", "The unrelated fact is amber.")
    targets = [duplicate_a, duplicate_b, unique]

    schema = build_reconciliation_schema([correction], targets)
    correct_branches = [
        branch
        for branch in schema["properties"]["decisions"]["items"]["oneOf"]
        if branch["properties"]["operation"]["enum"] == ["correct"]
    ]
    assert [branch["properties"]["target_fact_id"]["enum"][0] for branch in correct_branches] == [
        "fact-unrelated"
    ]
    validator = Draft202012Validator(schema)
    assert not validator.is_valid(
        {
            "decisions": [
                decision(
                    0,
                    "correct",
                    target_fact_id="fact-old-a",
                    expected_version=1,
                )
            ]
        }
    )
    assert validator.is_valid(
        {
            "decisions": [
                decision(
                    0,
                    "correct",
                    target_fact_id="fact-unrelated",
                    expected_version=1,
                )
            ]
        }
    )

    ignored_schema = Provider(
        {
            "decisions": [
                decision(
                    0,
                    "correct",
                    target_fact_id="fact-old-a",
                    expected_version=1,
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="ambiguous exact normalized content"):
        await reconcile_source_candidates(
            [correction], [correction_source], targets, ignored_schema
        )

    add_source = event("independent-add", "The new fact is violet.")
    add_provider = Provider({"decisions": [decision(0, "add")]})
    staged = await reconcile_source_candidates(
        [candidate(add_source)], [add_source], targets, add_provider
    )
    assert staged[0].operation == "add"


@pytest.mark.asyncio
async def test_add_and_support_preserve_candidate_content_without_generation() -> None:
    first = event("new-source", "The review window is Tuesday.")
    second = event("support-source", "The marker is cedar.")
    candidates = [candidate(first), candidate(second)]
    provider = Provider(
        {
            "decisions": [
                decision(0, "add"),
                decision(1, "support", target_fact_id="fact-1", expected_version=1),
            ]
        }
    )

    staged = await reconcile_source_candidates(candidates, [first, second], [target()], provider)

    assert [item.operation for item in staged] == ["add", "support"]
    assert [item.content for item in staged] == [first.content, second.content]
    assert all("content" not in item for item in provider.response["decisions"])


@pytest.mark.asyncio
async def test_exact_existing_user_fact_cannot_be_added_again() -> None:
    source = event("repeat-source", "The marker is cedar.")
    repeated = candidate(source)
    existing = target(content="  The marker is CEDAR. ")
    schema = build_reconciliation_schema([repeated], [existing])
    validator = Draft202012Validator(schema)
    duplicate = {"decisions": [decision(0, "add")]}
    assert not validator.is_valid(duplicate)
    assert validator.is_valid(
        {"decisions": [decision(0, "support", target_fact_id=existing.id, expected_version=1)]}
    )
    provider = Provider(duplicate)
    provider.supports_structured_output = False
    with pytest.raises(ValueError, match="existing exact normalized"):
        await reconcile_source_candidates([repeated], [source], [existing], provider)


@pytest.mark.asyncio
async def test_support_cannot_choose_between_duplicate_identities() -> None:
    source = event("repeat-source", "The marker is cedar.")
    targets = [target("first"), target("second")]
    response = {"decisions": [decision(0, "support", target_fact_id="first", expected_version=1)]}
    schema = build_reconciliation_schema([candidate(source)], targets)
    assert not Draft202012Validator(schema).is_valid(response)
    with pytest.raises(ValueError, match="ambiguous exact normalized"):
        await reconcile_source_candidates(
            [candidate(source)], [source], targets, Provider(response)
        )


def test_support_schema_excludes_nonmatching_targets() -> None:
    source = event("repeat-source", "The marker is cedar.")
    schema = build_reconciliation_schema([candidate(source)], [target(content="Different fact.")])
    assert not Draft202012Validator(schema).is_valid(
        {"decisions": [decision(0, "support", target_fact_id="fact-1", expected_version=1)]}
    )


def test_schema_binds_candidate_indices_and_operation_authority() -> None:
    user_correction = candidate(
        event("user-correction", "The marker is cedar."), "correction_or_rejection"
    )
    assistant_history = candidate(
        event("assistant-history", "The report says the marker is cedar.", "assistant"),
        "attributed_history",
    )
    schema = build_reconciliation_schema([user_correction, assistant_history], [target()])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(
        {
            "decisions": [
                decision(0, "correct", target_fact_id="fact-1", expected_version=1),
                decision(1, "add"),
            ]
        }
    )
    assert not validator.is_valid(
        {
            "decisions": [
                decision(0, "add"),
                decision(1, "support", target_fact_id="fact-1", expected_version=1),
            ]
        }
    )

    zero_target = build_reconciliation_schema([user_correction], [])
    Draft202012Validator.check_schema(zero_target)
    assert not validator.is_valid(
        {"decisions": [decision(0, "correct", target_fact_id="fact-1", expected_version=2)]}
    )
    assert not validator.is_valid(
        {
            "decisions": [
                decision(
                    0,
                    "conflict",
                    target_fact_id="foreign-fact",
                    expected_version=1,
                    reason="Conflicting source.",
                )
            ]
        }
    )
    zero_operations = {
        branch["properties"]["operation"]["enum"][0]
        for branch in zero_target["properties"]["decisions"]["items"]["oneOf"]
    }
    assert zero_operations == {"conflict", "unresolved"}
    assert all(
        branch["properties"].get("target_fact_id") == {"type": "null"}
        for branch in zero_target["properties"]["decisions"]["items"]["oneOf"]
    )


@pytest.mark.asyncio
async def test_optional_source_request_ref_is_reconstructed_for_any_retained_kind() -> None:
    source = event("requirement", "Keep the marker cedar.")
    provider = Provider({"decisions": [decision(0, "add")]})
    staged = await reconcile_source_candidates(
        [candidate(source, "continuing_requirement", source.content)],
        [source],
        [],
        provider,
    )
    assert staged[0].source_kind == "continuing_requirement"


@pytest.mark.asyncio
async def test_unsanitized_source_is_rejected_before_provider() -> None:
    source = event("unsafe", "api_key=supersecretvalue")
    provider = Provider({"decisions": [decision(0, "add")]})
    with pytest.raises(ValueError, match="sanitized"):
        await reconcile_source_candidates([candidate(source)], [source], [], provider)
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["support", "correct"])
async def test_nonuser_history_cannot_support_or_correct(operation: str) -> None:
    source = event("assistant-source", "The report says the marker is cedar.", "assistant")
    provider = Provider(
        {"decisions": [decision(0, operation, target_fact_id="fact-1", expected_version=1)]}
    )
    with pytest.raises(ValueError, match="Only .* may"):
        await reconcile_source_candidates(
            [candidate(source, "attributed_history")], [source], [target()], provider
        )


@pytest.mark.asyncio
async def test_unresolved_correction_is_explicit_and_cannot_become_add() -> None:
    source = event("unresolved-source", "The marker changed, but no replacement is accepted.")
    correction = candidate(source, "correction_or_rejection")
    unresolved = Provider(
        {"decisions": [decision(0, "unresolved", reason="No current target was accepted.")]}
    )
    staged = await reconcile_source_candidates([correction], [source], [], unresolved)
    assert staged[0].operation == "unresolved"
    assert staged[0].content == source.content

    add = Provider({"decisions": [decision(0, "add")]})
    with pytest.raises(ValueError, match="cannot become an add"):
        await reconcile_source_candidates([correction], [source], [], add)


@pytest.mark.asyncio
async def test_support_requires_exact_normalized_content_and_current_version() -> None:
    source = event("support-source", "The marker is cedar.")
    same = target(content="  The marker is CEDAR. ")
    provider = Provider(
        {"decisions": [decision(0, "support", target_fact_id="fact-1", expected_version=1)]}
    )
    staged = await reconcile_source_candidates([candidate(source)], [source], [same], provider)
    assert staged[0].operation == "support"

    stale = Provider(
        {"decisions": [decision(0, "support", target_fact_id="fact-1", expected_version=2)]}
    )
    with pytest.raises(ValueError, match="target version is stale"):
        await reconcile_source_candidates([candidate(source)], [source], [same], stale)


@pytest.mark.asyncio
async def test_foreign_target_and_unbound_candidate_are_rejected_before_provider() -> None:
    source = event("source", "The marker is cedar.")
    foreign = target()
    foreign.metadata["scope_id"] = "other-project"
    provider = Provider({"decisions": [decision(0, "add")]})
    with pytest.raises(ValueError, match="foreign scope"):
        await reconcile_source_candidates([candidate(source)], [source], [foreign], provider)
    assert provider.calls == []

    forged = replace(candidate(source), content="The marker is quartz.")
    with pytest.raises(ValueError, match="original source slice"):
        await reconcile_source_candidates([forged], [source], [], provider)
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses,error",
    [
        ({"decisions": []}, "exactly one decision"),
        ({"decisions": [decision(0, "add"), decision(0, "add")]}, "exactly one decision"),
        ({"decisions": [decision(4, "add")]}, "unknown candidate"),
        (
            {"decisions": [decision(0, "support", target_fact_id="missing", expected_version=1)]},
            "unavailable target",
        ),
    ],
)
async def test_decisions_must_account_for_candidates_and_targets(
    responses: dict[str, object], error: str
) -> None:
    source = event("source", "The marker is cedar.")
    with pytest.raises(ValueError, match=error):
        await reconcile_source_candidates(
            [candidate(source)], [source], [target()], Provider(responses)
        )


@pytest.mark.asyncio
async def test_mutation_after_provider_call_is_rejected() -> None:
    source = event("source", "The marker is cedar.")
    candidate_value = candidate(source)

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            source.content = "Changed while waiting."
            return await super().predict(prompt, **kwargs)

    provider = MutatingProvider({"decisions": [decision(0, "add")]})
    with pytest.raises(ValueError, match="Source events changed"):
        await reconcile_source_candidates([candidate_value], [source], [], provider)


@pytest.mark.asyncio
async def test_target_snapshot_mutation_after_provider_call_is_rejected() -> None:
    source = event("source", "The marker is cedar.")
    observed = target()

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            observed.content = "Changed while waiting."
            return await super().predict(prompt, **kwargs)

    provider = MutatingProvider({"decisions": [decision(0, "add")]})
    with pytest.raises(ValueError, match="inputs changed"):
        await reconcile_source_candidates([candidate(source)], [source], [observed], provider)


@pytest.mark.asyncio
async def test_unstructured_provider_still_gets_typed_runtime_validation() -> None:
    source = event("assistant-source", "The report says the marker is cedar.", "assistant")
    provider = Provider({"decisions": [decision(0, "add")]})
    provider.supports_structured_output = False
    staged = await reconcile_source_candidates(
        [candidate(source, "attributed_history")], [source], [], provider
    )
    assert staged[0].source_role == "assistant"
    assert provider.calls[0][1] == {}


def test_response_model_rejects_generated_content_and_missing_reason() -> None:
    with pytest.raises(ValueError):
        ReconciliationResponse.model_validate(
            {
                "decisions": [
                    {
                        "candidate_index": 0,
                        "operation": "unresolved",
                        "reason": " ",
                        "content": "invented",
                    }
                ]
            }
        )
