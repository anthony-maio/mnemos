"""Focused contract checks for standalone negative project requirements."""

from datetime import datetime, timezone

import pytest

from benchmarks.local_memory.source_classification import (
    ClassifiedInventory,
    ClassifiedSelection,
    classify_source_memories,
    render_classifications,
)
from benchmarks.local_memory.source_role_binding import build_source_role_binding_schema
from benchmarks.local_memory.source_selection import Kind
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import LLMProvider


def event(content: str, *, role: str = "user") -> EpisodicEvent:
    return EpisodicEvent(
        id="source",
        event_fingerprint="fingerprint-source",
        role=role,
        content=content,
        normalized_content=content.casefold(),
        scope="project",
        scope_id="project-1",
        source="manual",
        metadata={"user_profile_id": "profile-1", "project_id": "project-1"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def selection(kind: Kind) -> ClassifiedSelection:
    return ClassifiedSelection(
        source_id="source",
        first_unit=0,
        last_unit=0,
        kind=kind,
    )


class Provider(LLMProvider):
    supports_structured_output = True

    def __init__(self, response: ClassifiedInventory) -> None:
        self.response = response
        self.prompt = ""

    async def predict(self, prompt: str, **kwargs: object) -> str:
        self.prompt = prompt
        return self.response.model_dump_json()

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        raise AssertionError("No separate label classifier is used")


def test_standalone_negative_requirement_keeps_exact_user_negation() -> None:
    source = event("We will not keep the side panel in this project.")
    kept = render_classifications(
        ClassifiedInventory(selections=[selection("continuing_requirement")]), [source]
    )
    assert len(kept) == 1
    assert kept[0].content == source.content
    assert kept[0].kind == "continuing_requirement"


def test_nonuser_negative_report_cannot_gain_user_requirement_authority() -> None:
    source = event("We will not keep the side panel in this project.", role="assistant")
    schema = build_source_role_binding_schema([source])
    kind_enum = schema["properties"]["selections"]["items"]["oneOf"][0]["properties"]["kind"]
    assert "continuing_requirement" not in kind_enum["enum"]
    with pytest.raises(ValueError, match="Non-user"):
        render_classifications(
            ClassifiedInventory(selections=[selection("continuing_requirement")]), [source]
        )


@pytest.mark.asyncio
async def test_prompt_distinguishes_referential_correction_from_standalone_negative_rule() -> None:
    standalone = event("We will not keep the side panel in this project.")
    provider = Provider(ClassifiedInventory(selections=[selection("continuing_requirement")]))
    inventory, kept = await classify_source_memories([standalone], provider)
    assert inventory.selections[0].kind == "continuing_requirement"
    assert kept[0].content == standalone.content
    assert "standalone prospective negative rule" in provider.prompt
    assert "existing assertion or prior project state" in provider.prompt

    correction = event("Reject the earlier side-panel requirement; it is no longer accepted.")
    correction_provider = Provider(
        ClassifiedInventory(selections=[selection("correction_or_rejection")])
    )
    correction_inventory, _ = await classify_source_memories([correction], correction_provider)
    assert correction_inventory.selections[0].kind == "correction_or_rejection"
