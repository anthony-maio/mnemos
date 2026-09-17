"""Versioned inputs for the local memory service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .scopes import ScopeContext


def _fact_proposal_json_schema(schema: dict[str, Any]) -> None:
    """Use complete alternatives for grammar converters that select union branches.

    A partial oneOf alongside shared properties is valid JSON Schema, but some
    constrained decoders compile only the alternatives. Each alternative must
    therefore carry the content, evidence, and operation requirements itself.
    Runtime validation still checks the full proposal and its source evidence.
    """
    base = deepcopy(schema)
    branches = []
    for operation in ("add", "support", "correct", "conflict"):
        branch = deepcopy(base)
        branch["properties"]["operation"] = {"type": "string", "enum": [operation]}
        if operation in {"support", "correct"}:
            branch["required"].append("target_fact_id")
            branch["properties"]["target_fact_id"] = {"type": "string", "minLength": 1}
        if operation == "correct":
            branch["required"].append("expected_version")
            branch["properties"]["expected_version"] = {"type": "integer", "minimum": 1}
        branches.append(branch)
    schema.clear()
    schema["oneOf"] = branches


class CaptureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identity_version: Literal[1, 2] = 1
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


class ForgetPreviewRequest(BaseModel):
    """Read-only, bounded plan for removing a fact and reviewed capture copies."""

    model_config = ConfigDict(extra="forbid")
    context: ScopeContext
    fact_id: str = Field(min_length=1, max_length=512, strict=True)
    expected_version: int = Field(ge=1, strict=True)
    limit: int = Field(default=100, ge=1, le=500, strict=True)

    @field_validator("fact_id", mode="before")
    @classmethod
    def normalize_fact_id(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Fact ID must be a string")
        value = value.strip()
        if not 1 <= len(value) <= 512:
            raise ValueError("Fact ID length is invalid")
        return value


class ForgetRelatedRequest(BaseModel):
    """Compare-and-set apply request for a previously reviewed forget preview."""

    model_config = ConfigDict(extra="forbid")
    context: ScopeContext
    fact_id: str = Field(min_length=1, max_length=512, strict=True)
    expected_version: int = Field(ge=1, strict=True)
    preview_digest: str = Field(min_length=64, max_length=64, strict=True)
    preview_limit: int = Field(default=100, ge=1, le=500, strict=True)
    selected_fact_ids: list[str] = Field(default_factory=list, max_length=500)
    selected_source_event_ids: list[str] = Field(default_factory=list, max_length=500)

    @field_validator("fact_id", mode="before")
    @classmethod
    def normalize_fact_id(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Fact ID must be a string")
        value = value.strip()
        if not 1 <= len(value) <= 512:
            raise ValueError("Fact ID length is invalid")
        return value

    @field_validator("preview_digest")
    @classmethod
    def validate_preview_digest(cls, value: str) -> str:
        if any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("Preview digest must be hexadecimal")
        return value.lower()

    @field_validator("selected_fact_ids", "selected_source_event_ids", mode="before")
    @classmethod
    def normalize_selected_ids(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("Selected IDs must be a list")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Selected IDs must be strings")
            item = item.strip()
            if not 1 <= len(item) <= 512:
                raise ValueError("Selected ID length is invalid")
            normalized.append(item)
        return normalized
