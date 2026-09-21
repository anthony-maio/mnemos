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


def _extraction_response_json_schema(schema: dict[str, Any]) -> None:
    """Keep each extraction decision complete for constrained decoders.

    Some structured-output grammar converters inspect only ``oneOf``
    alternatives and ignore sibling object properties.  Put the decision,
    facts, and reason constraints on each complete alternative so a decoder
    cannot emit a no-facts response without its required explanation.
    """

    original = deepcopy(schema)
    properties = original.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Extraction response schema has no properties")
    facts = properties.get("facts")
    reason = properties.get("reason")
    if not isinstance(facts, dict) or not isinstance(reason, dict):
        raise ValueError("Extraction response schema is missing facts or reason")

    nonempty_reason = {
        "anyOf": [
            {"pattern": r"^[\s\S]*\S[\s\S]*$", "type": "string"},
            {"type": "null"},
        ]
    }
    facts_branch = {
        "additionalProperties": False,
        "properties": {
            "decision": {"enum": ["facts"], "type": "string"},
            "facts": {**facts, "minItems": 1},
            "reason": nonempty_reason,
        },
        "required": ["decision", "facts"],
        "type": "object",
    }
    no_facts_branch = {
        "additionalProperties": False,
        "properties": {
            "decision": {"enum": ["no_durable_facts"], "type": "string"},
            "facts": {**facts, "maxItems": 0},
            "reason": {
                "pattern": r"^[\s\S]*\S[\s\S]*$",
                "type": "string",
            },
        },
        "required": ["decision", "facts", "reason"],
        "type": "object",
    }

    definitions = original.get("$defs")
    title = original.get("title")
    description = original.get("description")
    schema.clear()
    if isinstance(definitions, dict):
        schema["$defs"] = definitions
    if isinstance(description, str):
        schema["description"] = description
    if isinstance(title, str):
        schema["title"] = title
    schema["oneOf"] = [facts_branch, no_facts_branch]


class ReviewScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: ScopeContext
    scope: Literal["project", "workspace", "global"] = "project"


class ReviewItemRequest(ReviewScopeRequest):
    job_id: str = Field(min_length=1)
    candidate_index: int = Field(ge=0, strict=True)


class ReviewResolutionRequest(ReviewItemRequest):
    event_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    operation: Literal["add", "correct"]
    target_fact_id: str | None = Field(default=None, min_length=1)
    expected_version: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def validate_resolution(self) -> ReviewResolutionRequest:
        if not self.content.strip() or not self.event_id.strip():
            raise ValueError("Resolution requires explicit clarification and event identity")
        if self.operation == "add":
            if self.target_fact_id is not None or self.expected_version is not None:
                raise ValueError("Add resolution cannot name a correction target")
        elif self.target_fact_id is None or self.expected_version is None:
            raise ValueError("Correction resolution requires a target and expected version")
        return self


class CaptureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identity_version: Literal[1, 2] = 1
    # A provisional native hook event was durably acknowledged before a
    # transcript occurrence identifier was available.  Such an event is
    # retained as raw evidence only; the status is deliberately a single
    # typed field so it cannot disagree with a second capture-only flag.
    identity_status: Literal["normal", "provisional_unassociated"] = "normal"
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


FACT_CONTENT_LIMIT = 10_000


class FactProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_fact_proposal_json_schema)
    content: str = Field(min_length=1, max_length=FACT_CONTENT_LIMIT)
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

    model_config = ConfigDict(extra="forbid", json_schema_extra=_extraction_response_json_schema)
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
    # None applies the configured retrieval profile; hosts do not choose one.
    profile: Literal["baseline", "biomimetic"] | None = None
    view: Literal["memory", "captured_evidence"] = "memory"


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
