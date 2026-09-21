"""Source-bound review references for the pending extraction integration.

These records contain neither copied source text nor model-written explanations.
They describe unresolved work, never permission to write or correct a fact.
Storage must still enforce its lease, suppression and atomic commit checks.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .memory_safety import MemoryWriteFirewall
from .types import EpisodicEvent, MemoryChunk


class SourceReviewReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    candidate_index: int = Field(ge=0, strict=True)
    source_id: str = Field(min_length=1)
    source_fingerprint: str = Field(min_length=1)
    source_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_role: str = Field(min_length=1)
    source_channel: str = Field(min_length=1)
    occurred_at: str = Field(min_length=1)
    user_profile_id: str = Field(min_length=1)
    scope: Literal["project", "workspace", "global"]
    scope_id: str | None
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    source_kind: Literal[
        "asserted_fact",
        "user_preference",
        "continuing_requirement",
        "explicitly_retained_uncertainty",
        "correction_or_rejection",
        "attributed_history",
    ]
    operation: Literal["conflict", "unresolved"]
    target_fact_id: str | None = Field(default=None, min_length=1)
    expected_version: int | None = Field(default=None, ge=1, strict=True)
    revises_existing_requirement: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_contract(self) -> SourceReviewReference:
        if self.end <= self.start:
            raise ValueError("Review source range must be nonempty and forward")
        if self.scope != "global" and not self.scope_id:
            raise ValueError("Project and workspace reviews require a scope identity")
        if (self.target_fact_id is None) != (self.expected_version is None):
            raise ValueError("Review target and version must be supplied together")
        if self.operation == "unresolved" and self.target_fact_id is not None:
            raise ValueError("Unresolved reviews cannot select a target")
        if self.source_role != "user" and self.source_kind != "attributed_history":
            raise ValueError("Non-user review cannot establish user authority")
        if self.revises_existing_requirement and (
            self.source_role != "user" or self.source_kind != "continuing_requirement"
        ):
            raise ValueError("Revision intent requires a direct user requirement")
        return self

    def resolve_source(
        self,
        event: EpisodicEvent,
        *,
        target: MemoryChunk | None = None,
        firewall: MemoryWriteFirewall | None = None,
    ) -> str:
        """Validate current provenance before exposing a pending review excerpt.

        This does not check suppression or authorize resolution. Callers must do
        both inside their scoped storage transaction before using the reference.
        """
        if (
            event.id != self.source_id
            or event.event_fingerprint != self.source_fingerprint
            or hashlib.sha256(event.content.encode()).hexdigest() != self.source_content_sha256
            or event.role != self.source_role
            or event.source != self.source_channel
            or event.occurred_at.isoformat() != self.occurred_at
            or event.metadata.get("user_profile_id") != self.user_profile_id
            or event.scope != self.scope
            or event.scope_id != self.scope_id
            or event.metadata.get("identity_status") == "provisional_unassociated"
        ):
            raise ValueError("Review source is unavailable, changed or outside its partition")
        if self.end > len(event.content):
            raise ValueError("Review source range exceeds current source")
        if self.target_fact_id is None:
            if target is not None:
                raise ValueError("Untargeted review cannot acquire a target implicitly")
        elif (
            target is None
            or target.id != self.target_fact_id
            or target.version != self.expected_version
            or target.metadata.get("user_profile_id") != self.user_profile_id
            or target.metadata.get("scope") != self.scope
            or target.metadata.get("scope_id") != self.scope_id
        ):
            raise ValueError("Review target is unavailable, stale or outside its partition")
        excerpt = event.content[self.start : self.end]
        decision = (firewall or MemoryWriteFirewall()).apply(excerpt)
        if not excerpt.strip() or not decision.allowed or decision.content != excerpt:
            raise ValueError("Review source excerpt violates write policy")
        return excerpt
