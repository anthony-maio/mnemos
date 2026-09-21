"""Structured source-role authority binding remains mechanical, not semantic."""

from datetime import datetime, timezone
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from benchmarks.local_memory.source_classification import (
    ClassifiedInventory,
    classify_source_memories,
)
from benchmarks.local_memory.source_role_binding import (
    build_source_role_binding_schema,
    classify_source_memories_with_role_binding,
)
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider


def source(event_id: str, role: str, content: str = "The marker is cedar.") -> EpisodicEvent:
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role=role,
        content=content,
        normalized_content=content.lower(),
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile", "project_id": "project"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def selection(source_id: str, kind: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "first_unit": 0,
        "last_unit": 0,
        "kind": kind,
        "revises_existing_requirement": False,
    }


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


def branch_for(schema: dict[str, Any], source_id: str) -> dict[str, Any]:
    items = schema["properties"]["selections"]["items"]
    branches = items["oneOf"]
    return next(
        branch for branch in branches if branch["properties"]["source_id"]["enum"] == [source_id]
    )


def test_schema_binds_user_and_nonuser_kinds_per_source() -> None:
    events = [source("user", "user"), source("assistant", "assistant")]
    schema = build_source_role_binding_schema(events)
    user = branch_for(schema, "user")
    assistant = branch_for(schema, "assistant")
    user_kinds = set(user["properties"]["kind"]["enum"])
    assistant_kinds = set(assistant["properties"]["kind"]["enum"])
    assert user_kinds == {
        "asserted_fact",
        "attributed_history",
        "continuing_requirement",
        "correction_or_rejection",
        "explicitly_retained_uncertainty",
        "question",
        "quoted_material",
        "temporary_instruction",
        "unaccepted_suggestion",
        "unrequested_speculation",
        "user_preference",
    }
    assert assistant_kinds == {
        "attributed_history",
        "question",
        "quoted_material",
        "temporary_instruction",
        "unaccepted_suggestion",
        "unrequested_speculation",
    }

    user_validator = Draft202012Validator(user)
    assistant_validator = Draft202012Validator(assistant)
    assert user_validator.is_valid(selection("user", "user_preference"))
    assert assistant_validator.is_valid(selection("assistant", "attributed_history"))
    assert assistant_validator.is_valid(selection("assistant", "question"))
    assert not assistant_validator.is_valid(selection("assistant", "asserted_fact"))
    assert not assistant_validator.is_valid(selection("assistant", "user_preference"))
    missing_intent = selection("user", "continuing_requirement")
    del missing_intent["revises_existing_requirement"]
    assert not user_validator.is_valid(missing_intent)
    assert not assistant_validator.is_valid(
        {**selection("assistant", "attributed_history"), "revises_existing_requirement": True}
    )


def test_schema_rejects_unavailable_source_and_keeps_same_scope_roles_distinct() -> None:
    schema = build_source_role_binding_schema(
        [source("user", "user"), source("assistant", "assistant")]
    )
    validator = Draft202012Validator(schema)
    valid = {"selections": [selection("assistant", "attributed_history")]}
    assert validator.is_valid(valid)
    assert not validator.is_valid({"selections": [selection("missing", "attributed_history")]})
    assert not validator.is_valid({"selections": [selection("assistant", "asserted_fact")]})


@pytest.mark.asyncio
async def test_wrapper_forwards_unchanged_prompt_and_role_bound_schema() -> None:
    event = source("assistant", "assistant")
    provider = Provider({"selections": [selection("assistant", "attributed_history")]})
    inventory, memories = await classify_source_memories_with_role_binding([event], provider)
    assert inventory == ClassifiedInventory.model_validate(provider.response)
    assert memories[0].source_role == "assistant"
    prompt, kwargs = provider.calls[0]
    original_provider = Provider(provider.response)
    original_inventory, original_memories = await classify_source_memories(
        [event], original_provider
    )
    assert prompt == original_provider.calls[0][0]
    assert inventory == original_inventory
    assert memories == original_memories
    schema = kwargs["format"]
    assert isinstance(schema, dict)
    assert branch_for(schema, "assistant")["properties"]["kind"]["enum"] == [
        "attributed_history",
        "question",
        "quoted_material",
        "temporary_instruction",
        "unaccepted_suggestion",
        "unrequested_speculation",
    ]


@pytest.mark.asyncio
async def test_runtime_renderer_rejects_nonuser_direct_fact_if_decoder_ignores_schema() -> None:
    event = source("assistant", "assistant")
    provider = Provider({"selections": [selection("assistant", "asserted_fact")]})
    with pytest.raises(ValueError, match="Non-user"):
        await classify_source_memories_with_role_binding([event], provider)


@pytest.mark.asyncio
async def test_runtime_rejects_unavailable_source_even_if_decoder_ignores_schema() -> None:
    provider = Provider({"selections": [selection("missing", "question")]})
    with pytest.raises(ValueError, match="unavailable source"):
        await classify_source_memories_with_role_binding([source("user", "user")], provider)


@pytest.mark.asyncio
async def test_snapshot_guard_remains_active_through_wrapper() -> None:
    event = source("user", "user")

    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            self.calls.append((prompt, dict(kwargs)))
            event.content = "Changed while the classifier was running."
            return json.dumps({"selections": []})

    with pytest.raises(ValueError, match="changed during classification"):
        await classify_source_memories_with_role_binding([event], MutatingProvider({}))


@pytest.mark.asyncio
async def test_nonstructured_provider_gets_no_schema_but_final_guards_remain() -> None:
    event = source("assistant", "assistant")

    class UnstructuredProvider(Provider):
        supports_structured_output = False

    provider = UnstructuredProvider({"selections": [selection("assistant", "asserted_fact")]})
    with pytest.raises(ValueError, match="Non-user"):
        await classify_source_memories_with_role_binding([event], provider)
    assert provider.calls[0][1] == {}


def test_schema_builder_does_not_mutate_base_classification_schema() -> None:
    before = ClassifiedInventory.model_json_schema()
    build_source_role_binding_schema([source("user", "user")])
    after = ClassifiedInventory.model_json_schema()
    assert after == before


@pytest.mark.asyncio
async def test_provider_schema_mutation_does_not_change_builder_schema() -> None:
    class MutatingProvider(Provider):
        async def predict(self, prompt: str, **kwargs: object) -> str:
            schema = kwargs["format"]
            assert isinstance(schema, dict)
            schema["properties"]["selections"]["items"]["oneOf"].clear()
            self.calls.append((prompt, dict(kwargs)))
            return json.dumps({"selections": []})

    provider = MutatingProvider({})
    await classify_source_memories_with_role_binding([source("user", "user")], provider)
    schema = build_source_role_binding_schema([source("user", "user")])
    assert schema["properties"]["selections"]["items"]["oneOf"]
