"""Revision intent is valid only for a direct user continuing requirement."""

from datetime import datetime, timezone
import json

from jsonschema import Draft202012Validator
import pytest

from mnemos.extraction.source_role_binding import (
    USER_SOURCE_KINDS,
    build_source_role_binding_schema,
    classify_source_memories_with_role_binding,
)
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider


def event(role: str = "user") -> EpisodicEvent:
    return EpisodicEvent(
        id="observed-source",
        event_fingerprint="observed-fingerprint",
        role=role,
        content="Correction: the backup window is Friday, not Thursday.",
        normalized_content="correction: the backup window is friday, not thursday.",
        scope="project",
        scope_id="project",
        source="manual",
        metadata={"user_profile_id": "profile"},
        occurred_at=datetime(2026, 11, 7, tzinfo=timezone.utc),
    )


def selection(kind: str, flag: bool) -> dict[str, object]:
    return {
        "source_id": "observed-source",
        "first_unit": 0,
        "last_unit": 0,
        "kind": kind,
        "retention_request_unit": None,
        "revises_existing_requirement": flag,
    }


@pytest.mark.parametrize("kind", sorted(USER_SOURCE_KINDS - {"continuing_requirement"}))
def test_other_kinds_reject_revision_true_in_every_decoder_alternative(kind: str) -> None:
    schema = build_source_role_binding_schema([event()])
    Draft202012Validator.check_schema(schema)
    valid, invalid = selection(kind, False), selection(kind, True)
    assert Draft202012Validator(schema).is_valid({"selections": [valid]})
    assert not Draft202012Validator(schema).is_valid({"selections": [invalid]})
    alternatives = schema["properties"]["selections"]["items"]["oneOf"]
    assert not any(Draft202012Validator(branch).is_valid(invalid) for branch in alternatives)


@pytest.mark.parametrize("flag", [False, True])
def test_user_continuing_requirement_preserves_both_revision_decisions(flag: bool) -> None:
    schema = build_source_role_binding_schema([event()])
    item = selection("continuing_requirement", flag)
    assert Draft202012Validator(schema).is_valid({"selections": [item]})
    alternatives = schema["properties"]["selections"]["items"]["oneOf"]
    accepting = [branch for branch in alternatives if Draft202012Validator(branch).is_valid(item)]
    assert len(accepting) == 1
    for branch in accepting:
        validator = Draft202012Validator(branch)
        assert not validator.is_valid({k: v for k, v in item.items() if k != "source_id"})
        assert not validator.is_valid({**item, "source_id": "unobserved"})
        assert not validator.is_valid({**item, "first_unit": -1})
        assert not validator.is_valid({**item, "unexpected": True})


@pytest.mark.parametrize("role", ["assistant", "tool", "system"])
def test_nonuser_cannot_gain_revision_authority(role: str) -> None:
    validator = Draft202012Validator(build_source_role_binding_schema([event(role)]))
    assert validator.is_valid({"selections": [selection("attributed_history", False)]})
    assert not validator.is_valid({"selections": [selection("attributed_history", True)]})
    assert not validator.is_valid({"selections": [selection("continuing_requirement", True)]})


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_runtime_still_rejects_a_provider_ignoring_revision_constraints(
    structured: bool,
) -> None:
    class Provider(LLMProvider):
        supports_structured_output = structured

        async def predict(self, prompt: str, **kwargs: object) -> str:
            return json.dumps({"selections": [selection("correction_or_rejection", True)]})

        async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
            return {}

    with pytest.raises(ValueError, match="direct user continuing requirement"):
        await classify_source_memories_with_role_binding([event()], Provider())
