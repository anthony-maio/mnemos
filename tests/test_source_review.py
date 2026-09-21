"""Review references cannot become an independent copy or grant of authority."""

import hashlib

import pytest

from mnemos.source_review import SourceReviewReference
from mnemos.types import EpisodicEvent, MemoryChunk


def source() -> EpisodicEvent:
    return EpisodicEvent(
        id="source",
        event_fingerprint="fingerprint",
        role="user",
        content="Keep the old policy pending review.",
        normalized_content="unused",
        scope="project",
        scope_id="project-a",
        metadata={"user_profile_id": "owner"},
    )


def reference(event: EpisodicEvent, **overrides: object) -> SourceReviewReference:
    data = dict(
        candidate_index=0,
        source_id=event.id,
        source_fingerprint=event.event_fingerprint,
        source_content_sha256=hashlib.sha256(event.content.encode()).hexdigest(),
        source_role=event.role,
        source_channel=event.source,
        occurred_at=event.occurred_at.isoformat(),
        user_profile_id="owner",
        scope=event.scope,
        scope_id=event.scope_id,
        start=0,
        end=len(event.content),
        source_kind="continuing_requirement",
        operation="unresolved",
    )
    data.update(overrides)
    return SourceReviewReference.model_validate(data)


def test_roundtrip_resolves_source_without_persisting_its_text() -> None:
    event = source()
    encoded = reference(event).model_dump_json()
    assert event.content not in encoded
    restored = SourceReviewReference.model_validate_json(encoded)
    assert restored.resolve_source(event) == event.content


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "other"),
        ("event_fingerprint", "other"),
        ("role", "assistant"),
        ("content", "A different rule with an equally plausible source."),
        ("source", "other"),
        ("scope", "workspace"),
        ("scope_id", "project-b"),
        ("metadata", {"user_profile_id": "another-owner"}),
        ("metadata", {"user_profile_id": "owner", "identity_status": "provisional_unassociated"}),
    ],
)
def test_changed_source_is_rejected(field: str, value: object) -> None:
    event = source()
    saved = reference(event)
    changed = event.model_copy(deep=True, update={field: value})
    with pytest.raises(ValueError, match="source"):
        saved.resolve_source(changed)


def test_target_must_still_exist_at_observed_version_and_partition() -> None:
    event = source()
    saved = reference(event, operation="conflict", target_fact_id="fact", expected_version=2)
    target = MemoryChunk(
        id="fact",
        content="Old policy",
        version=2,
        metadata={
            "user_profile_id": "owner",
            "scope": "project",
            "scope_id": "project-a",
        },
    )
    assert saved.resolve_source(event, target=target) == event.content
    for changed in (
        None,
        target.model_copy(update={"version": 3}),
        target.model_copy(update={"metadata": {**target.metadata, "scope_id": "other"}}),
    ):
        with pytest.raises(ValueError, match="target"):
            saved.resolve_source(event, target=changed)


@pytest.mark.parametrize(
    "changes",
    [
        {"start": 4, "end": 3},
        {"start": True},
        {"target_fact_id": "fact"},
        {"target_fact_id": "fact", "expected_version": 2},
        {"source_role": "assistant"},
        {"reason": "The model claims this is authoritative"},
        {"content": "A copied assertion"},
        {"scope_id": None},
        {"source_kind": "asserted_fact", "revises_existing_requirement": True},
    ],
)
def test_invalid_or_content_bearing_records_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        reference(source(), **changes)


def test_out_of_bounds_range_cannot_resolve() -> None:
    event = source()
    with pytest.raises(ValueError, match="range"):
        reference(event, end=len(event.content) + 1).resolve_source(event)
