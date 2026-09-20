"""Durable maintenance job contracts used by the local storage runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


JobStatus = Literal["pending", "leased", "completed", "failed", "exhausted"]


class JobNotFoundError(KeyError):
    """The requested job is absent or unavailable in the caller's scope."""


class FactNotFoundError(KeyError):
    """The requested fact is absent or unavailable in the caller's scope."""


class DurableJob(BaseModel):
    """A retryable, lease-protected unit of local maintenance work."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = "pending"
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=5, ge=1)
    available_at: datetime = Field(default_factory=_utcnow)
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_until: datetime | None = None
    last_error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class FactEvidence(BaseModel):
    """Source-backed evidence retained alongside an accepted fact."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    fact_id: str
    source_event_id: str | None = None
    excerpt: str
    source_ref: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class FactRevision(BaseModel):
    """A content revision with an optimistic expected-version check."""

    revision_id: str = Field(default_factory=lambda: str(uuid4()))
    fact_id: str
    version: int = Field(ge=1)
    content: str
    reason: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
