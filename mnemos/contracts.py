"""Versioned inputs for the local memory service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .scopes import ScopeContext


def _fact_proposal_json_schema(schema: dict[str, Any]) -> None:
    """Expose proposal cross-field requirements to structured-output providers."""
    schema["oneOf"] = [
        {
            "properties": {"operation": {"enum": ["add"]}},
            "required": ["operation"],
        },
        {
            "properties": {"operation": {"enum": ["support"]}},
            "required": ["operation", "target_fact_id"],
        },
        {
            "properties": {"operation": {"enum": ["correct"]}},
            "required": ["operation", "target_fact_id", "expected_version"],
        },
        {
            "properties": {"operation": {"enum": ["conflict"]}},
            "required": ["operation"],
        },
    ]


class CaptureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1, max_length=512)
    context: ScopeContext
    scope: Literal["project", "workspace", "global"] = "project"
    occurred_at: datetime
    role: Literal["user", "assistant", "tool", "system"]
    content: str = Field(min_length=1, max_length=100_000)
    source_kind: str = Field(min_length=1, max_length=100)
    source_ref: str | None = Field(default=None, max_length=4096)

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Event timestamp must include a timezone")
        return value


class FactProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_fact_proposal_json_schema)
    content: str = Field(min_length=1, max_length=10_000)
    evidence_ids: list[str] = Field(min_length=1)
    operation: Literal["add", "support", "correct", "conflict"]
    target_fact_id: str | None = None
    expected_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_correction_target(self) -> FactProposal:
        if self.operation in {"support", "correct"} and not self.target_fact_id:
            raise ValueError("Support and correction proposals need a target fact")
        if self.operation == "correct" and self.expected_version is None:
            raise ValueError("Corrections need an expected version")
        return self


class ExtractionResponse(BaseModel):
    """The complete, schema-checked response from sleep extraction."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["facts", "no_durable_facts"]
    facts: list[FactProposal]
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Reason must contain text when supplied")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> ExtractionResponse:
        if self.decision == "facts" and not self.facts:
            raise ValueError("Facts decision requires at least one proposal")
        if self.decision == "no_durable_facts":
            if self.facts:
                raise ValueError("No-facts decision cannot include proposals")
            if self.reason is None or not self.reason.strip():
                raise ValueError("No-facts decision requires a reason")
        return self


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=10_000)
    context: ScopeContext
    allowed_scopes: list[Literal["project", "workspace", "global"]] = Field(
        default_factory=lambda: cast(
            list[Literal["project", "workspace", "global"]], ["project", "global"]
        )
    )
    top_k: int = Field(default=5, ge=0, le=100)
    max_context_tokens: int = Field(default=1200, ge=0, le=32_000)
    profile: Literal["baseline", "biomimetic"] = "baseline"
