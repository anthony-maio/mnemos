"""Request-scoped evidence citation checks for sleep extraction."""

from datetime import datetime, timezone
import json

import pytest
from jsonschema import Draft202012Validator

from mnemos.modules.sleep import SleepDaemon
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.llm import LLMProvider
from mnemos.utils.storage import InMemoryStore


def _event(event_id: str, content: str = "I use SQLite.") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role="user",
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project-1",
        metadata={"user_profile_id": "profile-1"},
        occurred_at=datetime.now(timezone.utc),
    )


def _fact(fact_id: str = "fact-1", content: str = "The user uses SQLite.") -> MemoryChunk:
    return MemoryChunk(
        id=fact_id,
        content=content,
        metadata={
            "user_profile_id": "profile-1",
            "scope": "project",
            "scope_id": "project-1",
        },
    )


class _Provider(LLMProvider):
    supports_structured_output = True

    def __init__(self, response: dict):
        self.response = response
        self.prompt = ""
        self.kwargs: dict[str, object] = {}

    async def predict(self, prompt: str, **kwargs: object) -> str:
        self.prompt = prompt
        self.kwargs = kwargs
        return json.dumps(self.response)

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        _ = prompt, labels
        return {}


@pytest.mark.asyncio
async def test_structured_schema_binds_evidence_to_request_source_ids() -> None:
    events = [_event("source-1"), _event("source-2", "I also use Postgres.")]
    provider = _Provider(
        {
            "decision": "facts",
            "facts": [
                {
                    "content": "The user uses SQLite.",
                    "evidence_ids": ["source-1"],
                    "operation": "add",
                }
            ],
        }
    )

    proposals = await SleepDaemon(InMemoryStore()).propose_facts(
        events, provider, existing_facts=[_fact()]
    )

    assert len(proposals) == 1
    schema = provider.kwargs["format"]
    assert isinstance(schema, dict)
    branches = schema["$defs"]["FactProposal"]["oneOf"]
    assert all(
        branch["properties"]["evidence_ids"]["items"]["enum"] == ["source-1", "source-2"]
        for branch in branches
    )
    add_branch = next(
        branch for branch in branches if branch["properties"]["operation"]["enum"] == ["add"]
    )
    validator = Draft202012Validator(add_branch)
    valid = {
        "content": "The user uses SQLite.",
        "evidence_ids": ["source-1"],
        "operation": "add",
    }
    assert validator.is_valid(valid)
    assert not validator.is_valid({**valid, "evidence_ids": ["fact-1"]})
    assert "evidence_ids must contain only one or more IDs from SOURCE EVENTS" in provider.prompt
    assert "existing fact IDs may appear only as target_fact_id" in provider.prompt

    for operation in ("support", "correct"):
        branch = next(
            branch
            for branch in branches
            if branch["properties"]["operation"]["enum"] == [operation]
        )
        branch_validator = Draft202012Validator(branch)
        target_fact = {
            **valid,
            "operation": operation,
            "target_fact_id": "fact-1",
        }
        if operation == "correct":
            target_fact["expected_version"] = 1
        assert branch_validator.is_valid(target_fact)
        assert not branch_validator.is_valid({**target_fact, "evidence_ids": ["fact-1"]})

        assert branch["properties"]["target_fact_id"]["enum"] == ["fact-1"]
        if operation == "correct":
            assert branch["properties"]["expected_version"]["enum"] == [1]

    isolated_provider = _Provider(
        {
            "decision": "facts",
            "facts": [
                {
                    "content": "The user uses SQLite.",
                    "evidence_ids": ["source-3"],
                    "operation": "add",
                }
            ],
        }
    )
    await SleepDaemon(InMemoryStore()).propose_facts([_event("source-3")], isolated_provider)
    isolated_schema = isolated_provider.kwargs["format"]
    assert [
        branch["properties"]["operation"]["enum"]
        for branch in isolated_schema["$defs"]["FactProposal"]["oneOf"]
    ] == [["add"], ["conflict"]]
    assert isolated_schema["$defs"]["FactProposal"]["oneOf"][0]["properties"]["evidence_ids"][
        "items"
    ]["enum"] == ["source-3"]
    assert schema["$defs"]["FactProposal"]["oneOf"][0]["properties"]["evidence_ids"]["items"][
        "enum"
    ] == ["source-1", "source-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "support", "correct"])
async def test_valid_operations_keep_source_and_target_ids_separate(operation: str) -> None:
    source = _event("source-current", "The user now uses SQLite.")
    existing = _fact()
    fact = {
        "content": existing.content if operation == "support" else source.content,
        "evidence_ids": [source.id],
        "operation": operation,
    }
    if operation in {"support", "correct"}:
        fact["target_fact_id"] = existing.id
    if operation == "correct":
        fact["expected_version"] = existing.version
    provider = _Provider({"decision": "facts", "facts": [fact]})

    proposals = await SleepDaemon(InMemoryStore()).propose_facts(
        [source], provider, existing_facts=[existing]
    )

    assert proposals[0].evidence_ids == [source.id]
    if operation in {"support", "correct"}:
        assert proposals[0].target_fact_id == existing.id
        assert existing.id not in proposals[0].evidence_ids


@pytest.mark.asyncio
async def test_schema_pairs_each_correction_target_with_current_version() -> None:
    source = _event("source-current", "The user now uses SQLite.")
    first = _fact("fact-a")
    second = _fact("fact-b", "The user uses Postgres.")
    second.version = 3
    provider = _Provider(
        {
            "decision": "facts",
            "facts": [
                {
                    "content": "The user now uses SQLite.",
                    "evidence_ids": [source.id],
                    "operation": "correct",
                    "target_fact_id": first.id,
                    "expected_version": first.version,
                }
            ],
        }
    )

    await SleepDaemon(InMemoryStore()).propose_facts(
        [source], provider, existing_facts=[first, second]
    )
    schema = provider.kwargs["format"]
    assert isinstance(schema, dict)
    branches = schema["$defs"]["FactProposal"]["oneOf"]
    support = next(
        branch for branch in branches if branch["properties"]["operation"]["enum"] == ["support"]
    )
    assert support["properties"]["target_fact_id"]["enum"] == ["fact-a", "fact-b"]
    correction = [
        branch for branch in branches if branch["properties"]["operation"]["enum"] == ["correct"]
    ]
    assert len(correction) == 2
    assert {
        (
            branch["properties"]["target_fact_id"]["enum"][0],
            branch["properties"]["expected_version"]["enum"][0],
        )
        for branch in correction
    } == {("fact-a", 1), ("fact-b", 3)}

    validator = Draft202012Validator(schema)
    valid = {
        "decision": "facts",
        "facts": [
            {
                "content": "The user now uses SQLite.",
                "evidence_ids": [source.id],
                "operation": "correct",
                "target_fact_id": "fact-b",
                "expected_version": 3,
            }
        ],
    }
    invalid_cross_pair = {
        **valid,
        "facts": [{**valid["facts"][0], "target_fact_id": "fact-a"}],
    }
    assert validator.is_valid(valid)
    assert not validator.is_valid(invalid_cross_pair)


@pytest.mark.asyncio
async def test_stale_correction_version_remains_runtime_rejected() -> None:
    source = _event("source-current", "The user now uses SQLite.")
    existing = _fact()
    provider = _Provider(
        {
            "decision": "facts",
            "facts": [
                {
                    "content": source.content,
                    "evidence_ids": [source.id],
                    "operation": "correct",
                    "target_fact_id": existing.id,
                    "expected_version": existing.version + 1,
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="current version"):
        await SleepDaemon(InMemoryStore()).propose_facts(
            [source], provider, existing_facts=[existing]
        )


@pytest.mark.asyncio
async def test_conflict_branch_is_available_without_existing_targets() -> None:
    source = _event("source-conflict", "The cache may use PostgreSQL.")
    provider = _Provider(
        {
            "decision": "facts",
            "facts": [
                {
                    "content": "The cache may use PostgreSQL.",
                    "evidence_ids": [source.id],
                    "operation": "conflict",
                }
            ],
        }
    )

    proposals = await SleepDaemon(InMemoryStore()).propose_facts([source], provider)
    assert proposals[0].operation == "conflict"
    schema = provider.kwargs["format"]
    assert isinstance(schema, dict)
    assert [
        branch["properties"]["operation"]["enum"]
        for branch in schema["$defs"]["FactProposal"]["oneOf"]
    ] == [["add"], ["conflict"]]


@pytest.mark.asyncio
async def test_existing_fact_id_in_evidence_is_rejected_at_runtime() -> None:
    source = _event("source-current")
    existing = _fact()
    provider = _Provider(
        {
            "decision": "facts",
            "facts": [
                {
                    "content": existing.content,
                    "evidence_ids": [existing.id],
                    "operation": "support",
                    "target_fact_id": existing.id,
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="unavailable evidence"):
        await SleepDaemon(InMemoryStore()).propose_facts(
            [source], provider, existing_facts=[existing]
        )


@pytest.mark.asyncio
async def test_no_durable_fact_and_empty_source_contracts_remain_strict() -> None:
    source = _event("source-current")
    provider = _Provider(
        {"decision": "no_durable_facts", "facts": [], "reason": "No supported assertion."}
    )
    assert await SleepDaemon(InMemoryStore()).propose_facts([source], provider) == []

    with pytest.raises(ValueError, match="requires source events"):
        await SleepDaemon(InMemoryStore()).propose_facts([], provider)
