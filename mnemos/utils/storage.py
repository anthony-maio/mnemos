"""
mnemos/utils/storage.py — Storage backends for the Mnemos memory system.

Storage backends handle persistence of MemoryChunks. The interface is
intentionally simple and synchronous — async wrappers are used at the
module level where needed.

Backends:
- MemoryStore (abstract): defines the interface
- InMemoryStore: dict-based, cosine similarity search (development default)
- SQLiteStore: persistent storage using Python's built-in sqlite3

The InMemoryStore is analogous to the hippocampus — fast, temporary,
in-process working memory. The SQLiteStore is analogous to neocortical
long-term storage — persistent and restart-safe.

All backends implement the same interface, so you can swap them transparently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import threading
from uuid import uuid4
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from ..types import (
    CognitiveState,
    EpisodicEvent,
    MemoryChunk,
    ReconsolidationJob,
    RetrievalFeedbackEvent,
)
from ..jobs import DurableJob, FactEvidence, FactRevision, JobStatus
from .embeddings import cosine_similarity


def normalize_memory_text(text: str) -> str:
    """Normalize text for exact-dedupe and canonical-key generation."""
    return " ".join(text.strip().lower().split())


def compute_canonical_key(
    content: str,
    scope: str,
    scope_id: str | None,
    user_profile_id: str | None = None,
) -> str:
    """Hash content, scope, and optional profile into a canonical identity.

    Canonical content deliberately preserves case.  Existing callers that omit
    a profile retain the legacy key shape and remain readable.
    """
    normalized = " ".join(content.strip().split())
    scope_token = scope.strip().lower()
    scope_id_token = "" if scope_id is None else scope_id.strip()
    if user_profile_id is None:
        # Preserve the legacy no-profile key format for existing databases.
        payload_text = f"{scope_token}|{scope_id_token.lower()}|{normalized.lower()}"
    else:
        # Profile identities and scope IDs are opaque and case-sensitive.  A
        # structured payload avoids delimiter collisions between identities.
        payload_text = json.dumps(
            [user_profile_id.strip(), scope_token, scope_id_token, normalized],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    payload = payload_text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_event_fingerprint(
    *,
    role: str,
    content: str,
    scope: str,
    scope_id: str | None,
    source: str,
    hook_event: str | None,
    occurred_at: datetime | None = None,
) -> str:
    normalized = normalize_memory_text(content)
    scope_token = scope.strip().lower()
    scope_id_token = "" if scope_id is None else scope_id.strip().lower()
    source_token = source.strip().lower()
    event_token = "" if hook_event is None else hook_event.strip().lower()
    occurrence_token = ""
    if source_token != "claude_hook" and occurred_at is not None:
        occurrence_token = occurred_at.isoformat()
    payload = (
        f"{role.strip().lower()}|{scope_token}|{scope_id_token}|"
        f"{source_token}|{event_token}|{occurrence_token}|{normalized}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _chunk_scope(chunk: MemoryChunk) -> tuple[str, str | None]:
    scope_raw = chunk.metadata.get("scope")
    scope = str(scope_raw).strip().lower() if isinstance(scope_raw, str) else "global"
    if scope not in {"project", "workspace", "global"}:
        scope = "global"
    if scope == "global":
        return scope, None
    scope_id_raw = chunk.metadata.get("scope_id")
    if isinstance(scope_id_raw, str) and scope_id_raw.strip():
        return scope, scope_id_raw.strip()
    return scope, "default"


def _source_identity_key(event: EpisodicEvent) -> str | None:
    """Build a case-sensitive identity for an externally supplied event ID."""
    metadata = event.metadata
    raw_id = metadata.get("source_event_id") or metadata.get("event_id")
    if raw_id is None or not str(raw_id):
        return None
    host = metadata.get("host") or metadata.get("source_host") or event.source
    profile = metadata.get("user_profile_id") or ""
    session = metadata.get("session_id") or ""
    return json.dumps(
        [str(host), str(profile), str(session), str(raw_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


class MemoryStore(ABC):
    """
    Abstract base class for memory storage backends.

    All methods are synchronous. For async contexts, run in a thread pool
    using asyncio.get_event_loop().run_in_executor().
    """

    @abstractmethod
    def store(self, chunk: MemoryChunk) -> None:
        """
        Persist a MemoryChunk (insert or overwrite by id).

        Args:
            chunk: The MemoryChunk to store.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """
        Retrieve the top-k most similar chunks by cosine similarity.

        Args:
            query_embedding: The query vector to search against.
            top_k: Maximum number of results to return.
            filter_fn: Optional predicate; only chunks where filter_fn(chunk) is True
                       are considered.

        Returns:
            List of MemoryChunks sorted by descending similarity, up to top_k.
        """
        ...

    @abstractmethod
    def update(self, chunk_id: str, chunk: MemoryChunk) -> bool:
        """
        Update an existing chunk by id.

        Args:
            chunk_id: The ID of the chunk to update.
            chunk: The new chunk data (replaces existing).

        Returns:
            True if a chunk with this id existed and was updated, False otherwise.
        """
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk by id.

        Args:
            chunk_id: The ID of the chunk to delete.

        Returns:
            True if the chunk existed and was deleted, False otherwise.
        """
        ...

    @abstractmethod
    def get_all(self) -> list[MemoryChunk]:
        """
        Return all stored chunks (unordered).

        Returns:
            List of all MemoryChunks in the store.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Return storage statistics.

        Returns:
            Dict with at minimum: 'total_chunks', 'total_bytes' (if applicable).
        """
        ...

    def get(self, chunk_id: str) -> MemoryChunk | None:
        """
        Retrieve a specific chunk by ID.

        Default implementation scans all chunks — backends may override for O(1) lookup.

        Args:
            chunk_id: The ID to look up.

        Returns:
            The MemoryChunk if found, None otherwise.
        """
        for chunk in self.get_all():
            if chunk.id == chunk_id:
                return chunk
        return None

    def list_chunks(
        self,
        *,
        limit: int = 100,
        after_id: str = "",
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        chunks = sorted(self.get_all(), key=lambda chunk: chunk.id)
        return [
            chunk
            for chunk in chunks
            if chunk.id > after_id and (filter_fn is None or filter_fn(chunk))
        ][: max(0, limit)]

    def list_legacy_chunks(self, *, limit: int = 100, after_id: str = "") -> list[MemoryChunk]:
        """List unassigned legacy facts for explicit, read-only review.

        A chunk is legacy only when neither its storage column nor either
        supported metadata spelling declares a profile.  The profile checks
        happen before pagination so profiled facts cannot consume the page.
        """

        def is_unassigned(chunk: MemoryChunk) -> bool:
            profile = chunk.metadata.get("user_profile_id")
            legacy_profile = chunk.metadata.get("profile_id")
            return all(
                value is None or not str(value).strip() for value in (profile, legacy_profile)
            )

        return self.list_chunks(
            limit=limit,
            after_id=after_id,
            filter_fn=is_unassigned,
        )

    def get_graph_edges(self, chunk_ids: list[str] | None = None) -> dict[str, dict[str, float]]:
        """Return persisted graph edges keyed by source chunk ID."""
        return {}

    def replace_graph_neighbors(self, chunk_id: str, neighbors: dict[str, float]) -> None:
        """Replace the outgoing graph-neighbor set for a chunk."""
        return None

    def set_graph_rebuild_cursor(self, cursor: str | None) -> None:
        raise NotImplementedError("This backend does not support durable graph rebuilding")

    def forget_fact(self, fact_id: str, *, expected_version: int) -> bool:
        raise NotImplementedError("This backend does not support durable forget suppression")

    def store_feedback_event(self, event: RetrievalFeedbackEvent) -> None:
        """Persist one retrieval-feedback event."""
        raise NotImplementedError("This storage backend does not support feedback events.")

    def list_feedback_events(
        self,
        *,
        event_type: str | None = None,
        user_profile_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> list[RetrievalFeedbackEvent]:
        """List stored retrieval-feedback events with optional filters."""
        return []

    def upsert_canonical(self, chunk: MemoryChunk) -> MemoryChunk:
        """Store or merge a chunk using its canonical key when available."""
        self.store(chunk)
        return chunk

    def append_episodic_event(self, event: EpisodicEvent) -> EpisodicEvent:
        """Persist one episodic event, returning the canonical stored record."""
        raise NotImplementedError("This storage backend does not support episodic events.")

    def get_episodic_event(self, event_id: str) -> EpisodicEvent | None:
        for event in self.list_episodic_events():
            if event.id == event_id:
                return event
        return None

    def capture_event_and_job(self, event: EpisodicEvent, job: DurableJob) -> EpisodicEvent:
        stored = self.append_episodic_event(event)
        self.enqueue_job(job)
        return stored

    def list_episodic_events(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        user_profile_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
        newest_first: bool = False,
        limit: int = 0,
    ) -> list[EpisodicEvent]:
        """List persisted episodic events with optional filters."""
        return []

    def enqueue_job(self, job: DurableJob) -> DurableJob:
        """Queue a durable maintenance job."""
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def get_job(self, job_id: str) -> DurableJob | None:
        return next((job for job in self.list_jobs() if job.id == job_id), None)

    def claim_jobs(
        self,
        *,
        owner: str,
        limit: int = 1,
        lease_seconds: int = 300,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> list[DurableJob]:
        """Claim available jobs, recovering leases that have expired."""
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def complete_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        result: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> bool:
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def fail_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        error: str,
        now: datetime | None = None,
    ) -> bool:
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def commit_fact_and_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        chunk: MemoryChunk | None = None,
        chunks: list[MemoryChunk] | None = None,
        evidence: list[FactEvidence] | None = None,
        revision: FactRevision | None = None,
        revisions: list[FactRevision] | None = None,
        processed_event_ids: list[str] | None = None,
        expected_version: int | None = None,
        expected_versions: dict[str, int] | None = None,
        result: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> bool:
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def retry_job(self, job_id: str, *, now: datetime | None = None) -> bool:
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 0,
        user_profile_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[DurableJob]:
        raise NotImplementedError("This storage backend does not support durable jobs.")

    def list_fact_evidence(self, fact_id: str) -> list[FactEvidence]:
        raise NotImplementedError("This storage backend does not support fact evidence.")

    def job_summary(
        self,
        *,
        now: datetime,
        user_profile_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        jobs = self.list_jobs()
        if user_profile_id is not None:
            jobs = [
                job
                for job in jobs
                if job.payload.get("context", {}).get("user_profile_id") == user_profile_id
                and job.payload.get("context", {}).get("project_id") == project_id
                and job.payload.get("context", {}).get("workspace_id") == workspace_id
            ]
        unfinished = [job for job in jobs if job.status != "completed"]
        pending = [job for job in jobs if job.status == "pending"]
        return {
            **{
                status: sum(job.status == status for job in jobs)
                for status in ("pending", "leased", "exhausted", "completed")
            },
            "retry_count": sum(max(0, job.attempt_count - 1) for job in jobs),
            "oldest_unfinished_seconds": max(
                ((now - job.created_at).total_seconds() for job in unfinished), default=0
            ),
            "oldest_pending_seconds": max(
                ((now - job.created_at).total_seconds() for job in pending), default=0
            ),
        }

    def list_fact_revisions(self, fact_id: str) -> list[FactRevision]:
        raise NotImplementedError("This storage backend does not support fact revisions.")

    def record_suppression(
        self,
        *,
        fact_id: str | None = None,
        canonical_key: str | None = None,
        source_event_id: str | None = None,
        reason: str,
    ) -> str:
        raise NotImplementedError("This storage backend does not support suppressions.")

    def is_suppressed(
        self,
        *,
        fact_id: str | None = None,
        canonical_key: str | None = None,
        source_event_id: str | None = None,
    ) -> bool:
        raise NotImplementedError("This storage backend does not support suppressions.")

    def update_episodic_event(
        self,
        event_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        """Update episodic event processing status."""
        return False

    def enqueue_reconsolidation_job(self, job: ReconsolidationJob) -> ReconsolidationJob:
        """Persist or refresh a reconsolidation job for a chunk."""
        raise NotImplementedError("This storage backend does not support reconsolidation jobs.")

    def list_reconsolidation_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 0,
    ) -> list[ReconsolidationJob]:
        """List durable reconsolidation jobs."""
        return []

    def update_reconsolidation_job(
        self,
        chunk_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        """Update a reconsolidation job's status or failure reason."""
        return False

    def delete_reconsolidation_job(self, chunk_id: str) -> bool:
        """Remove a reconsolidation job after successful processing."""
        return False

    def mark_retrieved(
        self,
        chunk_id: str,
        *,
        retrieved_at: datetime | None = None,
        reconsolidation_due_at: datetime | None = None,
    ) -> bool:
        """Persist retrieval-time maintenance fields such as last_retrieved_at."""
        chunk = self.get(chunk_id)
        if chunk is None:
            return False
        touched_at = retrieved_at or datetime.now(timezone.utc)
        chunk.access_count += 1
        chunk.last_retrieved_at = touched_at
        chunk.reconsolidation_due_at = reconsolidation_due_at
        return self.update(chunk_id, chunk)

    def mark_retrieved_many(
        self, chunk_ids: list[str], *, retrieved_at: datetime | None = None
    ) -> int:
        """Record one retrieval for each distinct selected fact."""
        return sum(
            self.mark_retrieved(chunk_id, retrieved_at=retrieved_at)
            for chunk_id in dict.fromkeys(chunk_ids)
        )

    def rebuild_indexes(
        self,
        *,
        rebuild_fts: bool = True,
        rebuild_vec: bool = True,
    ) -> dict[str, Any]:
        """Explicit maintenance hook for search/vector index rebuilds."""
        return {
            "rebuild_fts": rebuild_fts,
            "rebuild_vec": rebuild_vec,
            "performed": False,
        }

    def clear(self) -> None:
        """Remove all chunks from the store. Used for testing and consolidation."""
        for chunk in list(self.get_all()):
            self.delete(chunk.id)

    def touch(
        self,
        chunk_id: str,
        *,
        access_count: int | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """
        Persist an access-count/timestamp bump for a chunk.

        Default implementation falls back to get+update. Backends can override
        this with a more efficient payload/SQL-only mutation path.
        """
        chunk = self.get(chunk_id)
        if chunk is None:
            return False
        chunk.access_count = chunk.access_count + 1 if access_count is None else access_count
        chunk.updated_at = updated_at or datetime.now(timezone.utc)
        return self.update(chunk_id, chunk)


class InMemoryStore(MemoryStore):
    """
    In-memory storage backend using a dict and cosine similarity search.

    Thread-safe via a threading.Lock. Fast for development and testing.
    Data is lost when the process exits — pair with SQLiteStore for persistence.

    The cosine similarity search scans all chunks (O(N)) — suitable for up to
    ~100k chunks. For larger workloads, use a vector database backend.

    Args:
        name: Optional name for this store (used in stats output).
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._store: dict[str, MemoryChunk] = {}
        self._feedback_events: list[RetrievalFeedbackEvent] = []
        self._episodic_events: dict[str, EpisodicEvent] = {}
        self._reconsolidation_jobs: dict[str, ReconsolidationJob] = {}
        self._lock = threading.Lock()

    def store(self, chunk: MemoryChunk) -> None:
        """Store or overwrite a chunk by ID."""
        with self._lock:
            self._store[chunk.id] = chunk

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """
        Retrieve top-k chunks by cosine similarity.

        Chunks without embeddings are scored as 0.0 (unembedded content
        is not searchable but won't cause errors).
        """
        with self._lock:
            candidates = list(self._store.values())

        if filter_fn is not None:
            candidates = [c for c in candidates if filter_fn(c)]

        scored: list[tuple[float, MemoryChunk]] = []
        for chunk in candidates:
            if chunk.embedding is not None and len(chunk.embedding) == len(query_embedding):
                sim = cosine_similarity(query_embedding, chunk.embedding)
            else:
                sim = 0.0
            scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def update(self, chunk_id: str, chunk: MemoryChunk) -> bool:
        """Update an existing chunk; returns False if not found."""
        with self._lock:
            if chunk_id not in self._store:
                return False
            self._store[chunk_id] = chunk
            return True

    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk by ID; returns False if not found."""
        with self._lock:
            if chunk_id not in self._store:
                return False
            del self._store[chunk_id]
            return True

    def touch(
        self,
        chunk_id: str,
        *,
        access_count: int | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Update access_count and updated_at in-place without replacing the full chunk."""
        with self._lock:
            chunk = self._store.get(chunk_id)
            if chunk is None:
                return False
            chunk.access_count = chunk.access_count + 1 if access_count is None else access_count
            chunk.updated_at = updated_at or datetime.now(timezone.utc)
            return True

    def mark_retrieved(
        self,
        chunk_id: str,
        *,
        retrieved_at: datetime | None = None,
        reconsolidation_due_at: datetime | None = None,
    ) -> bool:
        with self._lock:
            chunk = self._store.get(chunk_id)
            if chunk is None:
                return False
            touched_at = retrieved_at or datetime.now(timezone.utc)
            chunk.access_count += 1
            chunk.last_retrieved_at = touched_at
            chunk.reconsolidation_due_at = reconsolidation_due_at
            return True

    def get(self, chunk_id: str) -> MemoryChunk | None:
        """O(1) lookup by ID."""
        with self._lock:
            return self._store.get(chunk_id)

    def get_all(self) -> list[MemoryChunk]:
        """Return a snapshot of all stored chunks."""
        with self._lock:
            return list(self._store.values())

    def get_stats(self) -> dict[str, Any]:
        """Return basic statistics about the store."""
        with self._lock:
            chunks = list(self._store.values())

        total = len(chunks)
        with_embedding = sum(1 for c in chunks if c.embedding is not None)
        avg_salience = sum(c.salience for c in chunks) / total if total > 0 else 0.0
        avg_access = sum(c.access_count for c in chunks) / total if total > 0 else 0.0

        return {
            "backend": "InMemoryStore",
            "name": self.name,
            "total_chunks": total,
            "chunks_with_embeddings": with_embedding,
            "average_salience": round(avg_salience, 4),
            "average_access_count": round(avg_access, 4),
        }

    def store_feedback_event(self, event: RetrievalFeedbackEvent) -> None:
        with self._lock:
            self._feedback_events.append(event)

    def list_feedback_events(
        self,
        *,
        event_type: str | None = None,
        user_profile_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> list[RetrievalFeedbackEvent]:
        with self._lock:
            events = list(self._feedback_events)

        if user_profile_id is not None:
            events = [event for event in events if event.user_profile_id == user_profile_id]
        if event_type is not None:
            events = [event for event in events if event.event_type == event_type]
        if scope is not None:
            events = [event for event in events if event.scope == scope]
        if scope_id is not None:
            events = [event for event in events if event.scope_id == scope_id]
        return events

    def upsert_canonical(self, chunk: MemoryChunk) -> MemoryChunk:
        scope, scope_id = _chunk_scope(chunk)
        profile = chunk.metadata.get("user_profile_id")
        profile_id = profile.strip() if isinstance(profile, str) and profile.strip() else None
        canonical_key = compute_canonical_key(chunk.content, scope, scope_id, profile_id)
        chunk.canonical_key = canonical_key
        if chunk.last_observed_at is None:
            chunk.last_observed_at = chunk.updated_at
        if chunk.observation_count <= 0:
            chunk.observation_count = 1

        with self._lock:
            existing = next(
                (
                    stored
                    for stored in self._store.values()
                    if stored.canonical_key == canonical_key
                ),
                None,
            )
            if existing is None:
                self._store[chunk.id] = chunk
                return chunk

            merged_duplicate_ids = list(existing.metadata.get("merged_duplicate_ids", []))
            if chunk.id != existing.id and chunk.id not in merged_duplicate_ids:
                merged_duplicate_ids.append(chunk.id)

            existing.metadata["merged_duplicate_ids"] = merged_duplicate_ids
            existing.metadata.update(chunk.metadata)
            existing.content = chunk.content
            existing.embedding = chunk.embedding or existing.embedding
            existing.salience = max(existing.salience, chunk.salience)
            existing.canonical_key = canonical_key
            existing.updated_at = max(existing.updated_at, chunk.updated_at)
            existing.created_at = min(existing.created_at, chunk.created_at)
            existing.last_observed_at = max(
                existing.last_observed_at or existing.updated_at,
                chunk.last_observed_at or chunk.updated_at,
            )
            if chunk.id != existing.id:
                existing.observation_count += max(chunk.observation_count, 1)
            else:
                existing.observation_count = max(
                    existing.observation_count, chunk.observation_count
                )
            if chunk.surprisal_score is not None:
                existing.surprisal_score = chunk.surprisal_score
            if chunk.surprisal_status != "unknown":
                existing.surprisal_status = chunk.surprisal_status
            if chunk.cognitive_state is not None:
                existing.cognitive_state = chunk.cognitive_state
            if chunk.cognitive_state_status != "unknown":
                existing.cognitive_state_status = chunk.cognitive_state_status
            if chunk.last_retrieved_at is not None:
                existing.last_retrieved_at = chunk.last_retrieved_at
            if chunk.reconsolidation_due_at is not None:
                existing.reconsolidation_due_at = chunk.reconsolidation_due_at
            self._store[existing.id] = existing
            return existing

    def append_episodic_event(self, event: EpisodicEvent) -> EpisodicEvent:
        with self._lock:
            source_identity = _source_identity_key(event)
            if source_identity is not None:
                existing_by_source = next(
                    (
                        stored
                        for stored in self._episodic_events.values()
                        if _source_identity_key(stored) == source_identity
                    ),
                    None,
                )
                if existing_by_source is not None:
                    if (
                        existing_by_source.content != event.content
                        or existing_by_source.role != event.role
                        or existing_by_source.scope != event.scope
                        or existing_by_source.scope_id != event.scope_id
                    ):
                        raise ValueError("Source event identity already has a different payload")
                    return existing_by_source
            existing = self._episodic_events.get(event.event_fingerprint)
            if existing is not None:
                return existing
            self._episodic_events[event.event_fingerprint] = event
            return event

    def get_episodic_event(self, event_id: str) -> EpisodicEvent | None:
        with self._lock:
            return next(
                (event for event in self._episodic_events.values() if event.id == event_id),
                None,
            )

    def list_episodic_events(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        user_profile_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
        newest_first: bool = False,
        limit: int = 0,
    ) -> list[EpisodicEvent]:
        with self._lock:
            events = list(self._episodic_events.values())

        if status is not None:
            events = [event for event in events if event.status == status]
        if scope is not None:
            events = [event for event in events if event.scope == scope]
        if scope_id is not None:
            events = [event for event in events if event.scope_id == scope_id]
        if user_profile_id is not None:
            events = [
                event
                for event in events
                if event.metadata.get("user_profile_id") == user_profile_id
            ]
        if session_id is not None:
            events = [event for event in events if event.metadata.get("session_id") == session_id]
        if before is not None:
            events = [event for event in events if event.occurred_at < before]
        events.sort(key=lambda event: (event.occurred_at, event.id), reverse=newest_first)
        return events[:limit] if limit > 0 else events

    def update_episodic_event(
        self,
        event_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            for fingerprint, event in self._episodic_events.items():
                if event.id != event_id:
                    continue
                self._episodic_events[fingerprint] = event.model_copy(
                    update={"status": status, "error": error}
                )
                return True
        return False

    def enqueue_reconsolidation_job(self, job: ReconsolidationJob) -> ReconsolidationJob:
        with self._lock:
            existing = self._reconsolidation_jobs.get(job.chunk_id)
            if existing is None:
                self._reconsolidation_jobs[job.chunk_id] = job
                return job
            updated = existing.model_copy(
                update={
                    "context": job.context,
                    "status": "pending",
                    "error": None,
                    "queued_at": job.queued_at,
                    "due_at": job.due_at,
                }
            )
            self._reconsolidation_jobs[job.chunk_id] = updated
            return updated

    def list_reconsolidation_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 0,
    ) -> list[ReconsolidationJob]:
        with self._lock:
            jobs = list(self._reconsolidation_jobs.values())
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        jobs.sort(key=lambda job: job.queued_at)
        return jobs[:limit] if limit > 0 else jobs

    def update_reconsolidation_job(
        self,
        chunk_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            job = self._reconsolidation_jobs.get(chunk_id)
            if job is None:
                return False
            self._reconsolidation_jobs[chunk_id] = job.model_copy(
                update={
                    "status": status,
                    "error": error,
                    "attempt_count": job.attempt_count + 1,
                    "last_attempt_at": datetime.now(timezone.utc),
                }
            )
            return True

    def delete_reconsolidation_job(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id not in self._reconsolidation_jobs:
                return False
            del self._reconsolidation_jobs[chunk_id]
            return True

    def clear(self) -> None:
        """Clear all chunks atomically."""
        with self._lock:
            self._store.clear()
            self._episodic_events.clear()
            self._reconsolidation_jobs.clear()


class SQLiteStore(MemoryStore):
    """
    Persistent storage backend using Python's built-in sqlite3 module.

    Embeddings are stored as JSON blobs (list of floats serialized to text).
    CognitiveState is stored as a JSON object. This is deliberately simple
    and portable — no external dependencies required.

    The database schema is automatically created on first use.
    Thread-safe via check_same_thread=False and a threading.Lock.

    Args:
        db_path: Path to the SQLite database file.
            Use ":memory:" for an in-memory SQLite database.
        name: Optional name for this store (used in stats output).
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            salience REAL NOT NULL DEFAULT 0.5,
            cognitive_state TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            canonical_key TEXT,
            user_profile_id TEXT,
            surprisal_score REAL,
            surprisal_status TEXT NOT NULL DEFAULT 'unknown',
            cognitive_state_status TEXT NOT NULL DEFAULT 'unknown',
            last_observed_at TEXT,
            observation_count INTEGER NOT NULL DEFAULT 1,
            last_retrieved_at TEXT,
            reconsolidation_due_at TEXT
        );
    """

    _CREATE_CHUNK_CANONICAL_INDEX_SQL = """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_chunks_canonical_key
        ON memory_chunks(canonical_key)
        WHERE canonical_key IS NOT NULL;
    """

    _CREATE_META_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS mnemos_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """

    _CREATE_EDGES_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS memory_edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL NOT NULL,
            edge_type TEXT NOT NULL DEFAULT 'SEMANTIC_SIMILAR',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (source_id, target_id, edge_type),
            FOREIGN KEY (source_id) REFERENCES memory_chunks(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
        );
    """

    _CREATE_EPISODIC_EVENTS_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS episodic_events (
            id TEXT PRIMARY KEY,
            event_fingerprint TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL,
            scope TEXT NOT NULL,
            scope_id TEXT,
            source TEXT NOT NULL,
            hook_event TEXT,
            source_event_id TEXT,
            source_identity_key TEXT,
            user_profile_id TEXT,
            session_id TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            created_at TEXT NOT NULL,
            occurred_at TEXT NOT NULL
        );
    """

    _CREATE_EPISODIC_EVENTS_STATUS_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_episodic_events_status
        ON episodic_events(status, scope, scope_id, created_at ASC);
    """

    _CREATE_RECONSOLIDATION_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS reconsolidation_jobs (
            id TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            context TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            queued_at TEXT NOT NULL,
            due_at TEXT NOT NULL,
            last_attempt_at TEXT,
            FOREIGN KEY (chunk_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
        );
    """

    _CREATE_RECONSOLIDATION_STATUS_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_reconsolidation_jobs_status
        ON reconsolidation_jobs(status, due_at ASC, queued_at ASC);
    """

    _CREATE_FEEDBACK_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS retrieval_feedback_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            query TEXT NOT NULL,
            scope TEXT NOT NULL,
            scope_id TEXT,
            chunk_ids TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
    """

    _CREATE_FEEDBACK_TYPE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_type
        ON retrieval_feedback_events(event_type, created_at DESC);
    """

    _CREATE_FEEDBACK_SCOPE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_scope
        ON retrieval_feedback_events(scope, scope_id, created_at DESC);
    """

    _CREATE_EDGES_SOURCE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_memory_edges_source
        ON memory_edges(edge_type, source_id);
    """

    _CREATE_EDGES_TARGET_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_memory_edges_target
        ON memory_edges(edge_type, target_id);
    """

    _CREATE_DURABLE_JOBS_SQL = """
        CREATE TABLE IF NOT EXISTS memory_jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 5,
            available_at TEXT NOT NULL,
            lease_owner TEXT,
            lease_token TEXT,
            lease_until TEXT,
            last_error TEXT,
            result TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """

    _CREATE_FACT_EVIDENCE_SQL = """
        CREATE TABLE IF NOT EXISTS fact_evidence (
            evidence_id TEXT PRIMARY KEY,
            fact_id TEXT NOT NULL,
            source_event_id TEXT,
            excerpt TEXT NOT NULL,
            source_ref TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (fact_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
        );
    """

    _CREATE_FACT_REVISIONS_SQL = """
        CREATE TABLE IF NOT EXISTS fact_revisions (
            revision_id TEXT PRIMARY KEY,
            fact_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (fact_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
        );
    """

    _CREATE_MEMORY_SUPPRESSIONS_SQL = """
        CREATE TABLE IF NOT EXISTS memory_suppressions (
            suppression_id TEXT PRIMARY KEY,
            fact_id TEXT,
            canonical_key TEXT,
            source_event_id TEXT,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """

    _CREATE_JOB_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_memory_jobs_available
        ON memory_jobs(status, available_at, lease_until, created_at);
    """

    _CREATE_EPISODIC_SCOPE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_episodic_events_scope_time
        ON episodic_events(user_profile_id, session_id, scope, scope_id, occurred_at);
    """

    _CREATE_FTS_SQL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
        USING fts5(content, content='memory_chunks', content_rowid='rowid');
    """

    _CREATE_FTS_INSERT_TRIGGER_SQL = """
        CREATE TRIGGER IF NOT EXISTS memory_chunks_ai
        AFTER INSERT ON memory_chunks
        BEGIN
            INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
    """

    _CREATE_FTS_DELETE_TRIGGER_SQL = """
        CREATE TRIGGER IF NOT EXISTS memory_chunks_ad
        AFTER DELETE ON memory_chunks
        BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
        END;
    """

    _CREATE_FTS_UPDATE_TRIGGER_SQL = """
        CREATE TRIGGER IF NOT EXISTS memory_chunks_au
        AFTER UPDATE OF content ON memory_chunks
        BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
            INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
    """

    _SCHEMA_VERSION = "7"
    _CHUNK_SELECT_COLUMNS = """
        id, content, embedding, metadata, salience, cognitive_state,
        created_at, updated_at, access_count, version, canonical_key, user_profile_id,
        surprisal_score, surprisal_status, cognitive_state_status, last_observed_at,
        observation_count, last_retrieved_at, reconsolidation_due_at
    """
    _GRAPH_EDGE_TYPE = "SEMANTIC_SIMILAR"
    _SQLITE_VEC_DIM_KEY = "sqlite_vec_dim"

    def __init__(self, db_path: str = "mnemos_memory.db", name: str = "sqlite") -> None:
        self.db_path = db_path
        self.name = name
        self._lock = threading.RLock()
        self._transaction_active = False
        self._sqlite_vec_enabled = False
        self._sqlite_vec_dim: int | None = None
        self._sqlite_vec_module: Any | None = None
        self.migration_backup: str | None = None
        if str(db_path) != ":memory:":
            from ..service import _protect_private_path

            database_path = Path(db_path)
            if not database_path.exists():
                fd = os.open(database_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.close(fd)
            _protect_private_path(database_path)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        try:
            self._backup_before_migration()
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=FULL;")
            if str(db_path) != ":memory:":
                for suffix in ("-wal", "-shm"):
                    auxiliary = Path(str(db_path) + suffix)
                    if auxiliary.exists():
                        _protect_private_path(auxiliary)
            self._load_sqlite_vec_extension()
            self._initialize_schema()
        except Exception:
            self._conn.close()
            raise

    def _backup_before_migration(self) -> None:
        tables = {
            row[0]
            for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        existing = self._meta_value("schema_version") if "mnemos_meta" in tables else None
        if existing is not None and int(existing) > int(self._SCHEMA_VERSION):
            raise ValueError("Database schema is newer than this Mnemos version")
        if not tables or existing == self._SCHEMA_VERSION or str(self.db_path) == ":memory:":
            return
        from ..service import _protect_private_path

        backup_path = Path(
            str(self.db_path) + f".pre-schema-{self._SCHEMA_VERSION}-{uuid4().hex}.sqlite"
        )
        fd = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(fd)
        _protect_private_path(backup_path)
        destination = sqlite3.connect(str(backup_path))
        try:
            self._conn.backup(destination)
            # Reading the schema marker verifies that the snapshot is readable
            # without requiring optional virtual-table extensions on this connection.
            destination.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        finally:
            destination.close()
        self.migration_backup = str(backup_path.resolve())

    def bind_embedding_identity(self, identity: dict[str, Any]) -> None:
        """Reject same-dimension model swaps instead of silently mixing vector spaces."""
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        with self._lock:
            existing = self._meta_value("embedding_identity")
            if existing is not None and existing != encoded:
                raise ValueError(
                    "Embedding identity changed; build and activate a reindex generation"
                )
            if existing is None:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM memory_chunks WHERE embedding IS NOT NULL"
                ).fetchone()[0]
                if count:
                    raise ValueError(
                        "Existing embeddings have no verified identity; explicit reindex is required"
                    )
                self._conn.execute(
                    "INSERT INTO mnemos_meta(key,value) VALUES ('embedding_identity',?)", (encoded,)
                )
                self._conn.commit()

    def bind_graph_contract(self, contract: dict[str, Any]) -> None:
        """Invalidate incompatible derived edges and resume rebuilding in bounded batches."""
        with self._lock:
            encoded = json.dumps(
                {**contract, "embedding_identity": self._meta_value("embedding_identity")},
                sort_keys=True,
                separators=(",", ":"),
            )
            if self._meta_value("graph_contract") == encoded:
                return
            self._conn.execute("DELETE FROM memory_edges")
            self._conn.executemany(
                "INSERT INTO mnemos_meta(key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [("graph_contract", encoded), ("graph_rebuild_after", "")],
            )
            self._conn.commit()

    def graph_rebuild_cursor(self) -> str | None:
        with self._lock:
            return self._meta_value("graph_rebuild_after")

    def set_graph_rebuild_cursor(self, cursor: str | None) -> None:
        with self._lock:
            if cursor is None:
                self._conn.execute("DELETE FROM mnemos_meta WHERE key='graph_rebuild_after'")
            else:
                self._conn.execute(
                    "INSERT INTO mnemos_meta(key,value) VALUES ('graph_rebuild_after',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (cursor,),
                )
            self._conn.commit()

    def _load_sqlite_vec_extension(self) -> None:
        try:
            import sqlite_vec
        except ImportError:
            return

        enable_extension = getattr(self._conn, "enable_load_extension", None)
        if not callable(enable_extension):
            return

        try:
            enable_extension(True)
            sqlite_vec.load(self._conn)
            enable_extension(False)
        except Exception:
            try:
                enable_extension(False)
            except Exception:
                pass
            return

        self._sqlite_vec_enabled = True
        self._sqlite_vec_module = sqlite_vec

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            str(row[1]) for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _ensure_chunk_columns(self) -> None:
        self._ensure_column("memory_chunks", "canonical_key", "TEXT")
        self._ensure_column("memory_chunks", "surprisal_score", "REAL")
        self._ensure_column(
            "memory_chunks",
            "surprisal_status",
            "TEXT NOT NULL DEFAULT 'unknown'",
        )
        self._ensure_column(
            "memory_chunks",
            "cognitive_state_status",
            "TEXT NOT NULL DEFAULT 'unknown'",
        )
        self._ensure_column("memory_chunks", "last_observed_at", "TEXT")
        self._ensure_column(
            "memory_chunks",
            "observation_count",
            "INTEGER NOT NULL DEFAULT 1",
        )
        self._ensure_column("memory_chunks", "last_retrieved_at", "TEXT")
        self._ensure_column("memory_chunks", "reconsolidation_due_at", "TEXT")

    def _apply_migrations(self) -> None:
        """Apply additive numbered migrations without rewriting existing data."""
        self._ensure_column("memory_chunks", "user_profile_id", "TEXT")
        self._ensure_column("retrieval_feedback_events", "user_profile_id", "TEXT")
        self._ensure_column("episodic_events", "source_event_id", "TEXT")
        self._ensure_column("episodic_events", "source_identity_key", "TEXT")
        self._ensure_column("episodic_events", "user_profile_id", "TEXT")
        self._ensure_column("episodic_events", "session_id", "TEXT")
        self._conn.execute("""UPDATE episodic_events
               SET source_identity_key = source || '|' || COALESCE(user_profile_id, '') || '|'
                   || COALESCE(session_id, '') || '|' || source_event_id
               WHERE source_event_id IS NOT NULL AND source_identity_key IS NULL""")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodic_source_event "
            "ON episodic_events(source_identity_key) WHERE source_identity_key IS NOT NULL"
        )
        self._conn.execute(self._CREATE_DURABLE_JOBS_SQL)
        self._ensure_column("memory_jobs", "lease_token", "TEXT")
        self._conn.execute(self._CREATE_FACT_EVIDENCE_SQL)
        self._conn.execute(self._CREATE_FACT_REVISIONS_SQL)
        self._conn.execute(self._CREATE_MEMORY_SUPPRESSIONS_SQL)
        self._conn.execute(self._CREATE_JOB_INDEX_SQL)
        self._conn.execute(self._CREATE_EPISODIC_SCOPE_INDEX_SQL)
        self._conn.execute(
            "INSERT INTO mnemos_meta(key, value) VALUES ('migration_4', 'applied') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        self._conn.execute("""CREATE INDEX IF NOT EXISTS idx_memory_chunks_scope
            ON memory_chunks(user_profile_id, json_extract(metadata, '$.scope'),
                             json_extract(metadata, '$.scope_id'))""")
        self._conn.execute(
            "INSERT OR IGNORE INTO mnemos_meta(key, value) VALUES ('migration_5', 'applied')"
        )

    def _initialize_schema(self) -> None:
        if self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mnemos_meta'"
        ).fetchone():
            existing = self._meta_value("schema_version")
            if existing is not None and int(existing) > int(self._SCHEMA_VERSION):
                self._conn.close()
                raise ValueError("Database schema is newer than this Mnemos version")
        self._conn.execute(self._CREATE_TABLE_SQL)
        self._ensure_chunk_columns()
        self._conn.execute(self._CREATE_CHUNK_CANONICAL_INDEX_SQL)
        self._conn.execute(self._CREATE_META_TABLE_SQL)
        self._conn.execute(self._CREATE_EDGES_TABLE_SQL)
        self._conn.execute(self._CREATE_EDGES_SOURCE_INDEX_SQL)
        self._conn.execute(self._CREATE_EDGES_TARGET_INDEX_SQL)
        self._conn.execute(self._CREATE_FEEDBACK_TABLE_SQL)
        self._conn.execute(self._CREATE_FEEDBACK_TYPE_INDEX_SQL)
        self._conn.execute(self._CREATE_FEEDBACK_SCOPE_INDEX_SQL)
        self._conn.execute(self._CREATE_EPISODIC_EVENTS_TABLE_SQL)
        self._conn.execute(self._CREATE_EPISODIC_EVENTS_STATUS_INDEX_SQL)
        self._conn.execute(self._CREATE_RECONSOLIDATION_TABLE_SQL)
        self._conn.execute(self._CREATE_RECONSOLIDATION_STATUS_INDEX_SQL)
        self._conn.execute(self._CREATE_FTS_SQL)
        self._conn.execute(self._CREATE_FTS_INSERT_TRIGGER_SQL)
        self._conn.execute(self._CREATE_FTS_DELETE_TRIGGER_SQL)
        self._conn.execute("DROP TRIGGER IF EXISTS memory_chunks_au")
        self._conn.execute(self._CREATE_FTS_UPDATE_TRIGGER_SQL)
        self._apply_migrations()
        # A regular table gives scoped exact cosine queries inexpensive primary
        # key lookups. vec0 point reads repeatedly decode internal vector blocks.
        # This is derived data; the JSON vectors remain the portable authority.
        vector_values_existed = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_vector_values'"
        ).fetchone()
        self._conn.execute("""CREATE TABLE IF NOT EXISTS memory_vector_values (
            rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL, dimension INTEGER NOT NULL
        )""")
        self._conn.execute("""CREATE TRIGGER IF NOT EXISTS memory_vector_values_delete
            AFTER DELETE ON memory_chunks BEGIN
                DELETE FROM memory_vector_values WHERE rowid=OLD.rowid;
            END""")
        if not vector_values_existed:
            self._rebuild_vector_values()
        self._conn.execute(
            """
            INSERT INTO mnemos_meta(key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (self._SCHEMA_VERSION,),
        )
        if self._fts_needs_rebuild():
            self._conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild');")
        if self._sqlite_vec_enabled:
            existing_dim = self._meta_value(self._SQLITE_VEC_DIM_KEY)
            if existing_dim not in (None, ""):
                self._sqlite_vec_dim = int(str(existing_dim))
            elif self._vec_table_exists():
                sample = self._conn.execute(
                    "SELECT rowid FROM memory_vec ORDER BY rowid LIMIT 1"
                ).fetchone()
                if sample is not None:
                    self._sqlite_vec_dim = None
            else:
                first_embedding = self._conn.execute(
                    "SELECT embedding FROM memory_chunks WHERE embedding IS NOT NULL LIMIT 1"
                ).fetchone()
                if first_embedding is not None and first_embedding[0]:
                    embedding = json.loads(first_embedding[0])
                    if isinstance(embedding, list) and embedding:
                        self._sqlite_vec_dim = len(embedding)
            if self._sqlite_vec_dim is not None:
                self._ensure_vec_table(self._sqlite_vec_dim)
        self._conn.commit()

    def _fts_needs_rebuild(self) -> bool:
        chunk_count = self._conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
        if chunk_count == 0:
            return False
        try:
            fts_count = self._conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        except sqlite3.DatabaseError:
            return True
        return int(fts_count) != int(chunk_count)

    def _meta_value(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM mnemos_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _vec_table_exists(self) -> bool:
        return bool(
            self._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'memory_vec'"
            ).fetchone()[0]
        )

    def _ensure_vec_table(self, dim: int) -> None:
        if not self._sqlite_vec_enabled:
            return
        if dim <= 0:
            raise ValueError("sqlite-vec dimension must be positive.")
        if self._sqlite_vec_dim is not None and self._sqlite_vec_dim != dim:
            raise ValueError(
                f"SQLite vector dimension mismatch: existing={self._sqlite_vec_dim}, attempted={dim}"
            )

        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{dim}])"
        )
        self._conn.execute(
            """
            INSERT INTO mnemos_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (self._SQLITE_VEC_DIM_KEY, str(dim)),
        )
        self._sqlite_vec_dim = dim

    def _serialize_vec(self, embedding: list[float]) -> Any:
        if not self._sqlite_vec_enabled or self._sqlite_vec_module is None:
            raise RuntimeError("sqlite-vec is not enabled")
        return self._sqlite_vec_module.serialize_float32(embedding)

    def _rebuild_vec_index(self) -> None:
        self._rebuild_vector_values()
        if (
            not self._sqlite_vec_enabled
            or self._sqlite_vec_dim is None
            or not self._vec_table_exists()
        ):
            return

        self._conn.execute("DELETE FROM memory_vec")
        rows = self._conn.execute(
            "SELECT rowid, embedding FROM memory_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        payload = []
        for rowid, embedding_json in rows:
            if not embedding_json:
                continue
            embedding = json.loads(embedding_json)
            if not isinstance(embedding, list) or len(embedding) != self._sqlite_vec_dim:
                continue
            payload.append((int(rowid), self._serialize_vec([float(x) for x in embedding])))
        if payload:
            self._conn.executemany(
                "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                payload,
            )

    def _rebuild_vector_values(self) -> None:
        """Rebuild the portable binary cache inside the caller's transaction."""
        self._conn.execute("DELETE FROM memory_vector_values")
        rows = self._conn.execute(
            "SELECT rowid, embedding FROM memory_chunks WHERE embedding IS NOT NULL"
        )
        while batch := rows.fetchmany(100):
            payload = []
            for rowid, raw in batch:
                embedding = json.loads(raw)
                if isinstance(embedding, list) and embedding:
                    payload.append(
                        (int(rowid), struct.pack(f"<{len(embedding)}f", *embedding), len(embedding))
                    )
            self._conn.executemany(
                "INSERT INTO memory_vector_values(rowid, embedding, dimension) VALUES (?, ?, ?)",
                payload,
            )

    def _sync_vec_row(self, chunk_id: str, embedding: list[float] | None) -> None:
        row = self._conn.execute(
            "SELECT rowid FROM memory_chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return
        rowid = int(row[0])

        self._conn.execute("DELETE FROM memory_vector_values WHERE rowid=?", (rowid,))
        if embedding:
            self._conn.execute(
                "INSERT INTO memory_vector_values(rowid, embedding, dimension) VALUES (?, ?, ?)",
                (rowid, struct.pack(f"<{len(embedding)}f", *embedding), len(embedding)),
            )
        if not self._sqlite_vec_enabled:
            return

        if self._vec_table_exists():
            self._conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (rowid,))

        if embedding is None:
            return

        dim = len(embedding)
        self._ensure_vec_table(dim)
        self._conn.execute(
            "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, self._serialize_vec(embedding)),
        )

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if value in (None, ""):
            return None
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _chunk_to_row(self, chunk: MemoryChunk) -> tuple[Any, ...]:
        """Serialize a MemoryChunk to a database row tuple."""
        embedding_json = json.dumps(chunk.embedding) if chunk.embedding is not None else None
        cognitive_state_json = (
            chunk.cognitive_state.model_dump_json() if chunk.cognitive_state is not None else None
        )
        return (
            chunk.id,
            chunk.content,
            embedding_json,
            json.dumps(chunk.metadata),
            chunk.salience,
            cognitive_state_json,
            chunk.created_at.isoformat(),
            chunk.updated_at.isoformat(),
            chunk.access_count,
            chunk.version,
            chunk.canonical_key,
            (
                str(chunk.metadata.get("user_profile_id"))
                if chunk.metadata.get("user_profile_id") not in (None, "")
                else None
            ),
            chunk.surprisal_score,
            chunk.surprisal_status,
            chunk.cognitive_state_status,
            chunk.last_observed_at.isoformat() if chunk.last_observed_at is not None else None,
            chunk.observation_count,
            chunk.last_retrieved_at.isoformat() if chunk.last_retrieved_at is not None else None,
            (
                chunk.reconsolidation_due_at.isoformat()
                if chunk.reconsolidation_due_at is not None
                else None
            ),
        )

    def _row_to_chunk(self, row: tuple[Any, ...]) -> MemoryChunk:
        """Deserialize a database row into a MemoryChunk."""
        if len(row) == 19:
            (
                id_,
                content,
                embedding_json,
                metadata_json,
                salience,
                cognitive_state_json,
                created_at,
                updated_at,
                access_count,
                version,
                canonical_key,
                user_profile_id,
                surprisal_score,
                surprisal_status,
                cognitive_state_status,
                last_observed_at,
                observation_count,
                last_retrieved_at,
                reconsolidation_due_at,
            ) = row
        else:
            (
                id_,
                content,
                embedding_json,
                metadata_json,
                salience,
                cognitive_state_json,
                created_at,
                updated_at,
                access_count,
                version,
                canonical_key,
                surprisal_score,
                surprisal_status,
                cognitive_state_status,
                last_observed_at,
                observation_count,
                last_retrieved_at,
                reconsolidation_due_at,
            ) = row
            user_profile_id = None

        embedding = json.loads(embedding_json) if embedding_json else None
        metadata = json.loads(metadata_json) if metadata_json else {}
        if user_profile_id not in (None, ""):
            metadata.setdefault("user_profile_id", str(user_profile_id))
        cognitive_state = (
            CognitiveState.model_validate_json(cognitive_state_json)
            if cognitive_state_json
            else None
        )

        return MemoryChunk(
            id=id_,
            content=content,
            embedding=embedding,
            metadata=metadata,
            salience=salience,
            cognitive_state=cognitive_state,
            created_at=cast(datetime, self._parse_dt(str(created_at))),
            updated_at=cast(datetime, self._parse_dt(str(updated_at))),
            access_count=access_count,
            version=version,
            canonical_key=(str(canonical_key) if canonical_key not in (None, "") else None),
            surprisal_score=(float(surprisal_score) if surprisal_score not in (None, "") else None),
            surprisal_status=str(surprisal_status or "unknown"),
            cognitive_state_status=str(cognitive_state_status or "unknown"),
            last_observed_at=self._parse_dt(cast(str | None, last_observed_at)),
            observation_count=int(observation_count or 0),
            last_retrieved_at=self._parse_dt(cast(str | None, last_retrieved_at)),
            reconsolidation_due_at=self._parse_dt(cast(str | None, reconsolidation_due_at)),
        )

    def _feedback_event_to_row(self, event: RetrievalFeedbackEvent) -> tuple[Any, ...]:
        return (
            event.id,
            event.event_type,
            event.query,
            event.scope,
            event.scope_id,
            json.dumps(event.chunk_ids),
            event.notes,
            event.created_at.isoformat(),
            event.user_profile_id,
        )

    def _row_to_feedback_event(self, row: tuple[Any, ...]) -> RetrievalFeedbackEvent:
        (
            event_id,
            event_type,
            query,
            scope,
            scope_id,
            chunk_ids_json,
            notes,
            created_at,
            user_profile_id,
        ) = row

        chunk_ids = json.loads(chunk_ids_json) if chunk_ids_json else []
        created_dt = datetime.fromisoformat(created_at)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        return RetrievalFeedbackEvent(
            id=str(event_id),
            event_type=str(event_type),
            query=str(query),
            scope=str(scope),
            scope_id=(str(scope_id) if scope_id is not None else None),
            chunk_ids=[str(chunk_id) for chunk_id in chunk_ids],
            notes=str(notes),
            created_at=created_dt,
            user_profile_id=user_profile_id,
        )

    def _episodic_event_to_row(self, event: EpisodicEvent) -> tuple[Any, ...]:
        source_event_id = event.metadata.get("source_event_id") or event.metadata.get("event_id")
        profile_id = event.metadata.get("user_profile_id")
        session_id = event.metadata.get("session_id")
        source_identity_key = _source_identity_key(event)
        return (
            event.id,
            event.event_fingerprint,
            event.role,
            event.content,
            event.normalized_content,
            event.scope,
            event.scope_id,
            event.source,
            event.hook_event,
            source_event_id if isinstance(source_event_id, str) else None,
            source_identity_key,
            profile_id if isinstance(profile_id, str) else None,
            session_id if isinstance(session_id, str) else None,
            json.dumps(event.metadata),
            event.status,
            event.error,
            event.created_at.isoformat(),
            event.occurred_at.isoformat(),
        )

    def _row_to_episodic_event(self, row: tuple[Any, ...]) -> EpisodicEvent:
        (
            event_id,
            event_fingerprint,
            role,
            content,
            normalized_content,
            scope,
            scope_id,
            source,
            hook_event,
            source_event_id,
            source_identity_key,
            profile_id,
            session_id,
            metadata_json,
            status,
            error,
            created_at,
            occurred_at,
        ) = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        if source_event_id is not None:
            metadata.setdefault("source_event_id", str(source_event_id))
        if profile_id is not None:
            metadata.setdefault("user_profile_id", str(profile_id))
        if session_id is not None:
            metadata.setdefault("session_id", str(session_id))
        return EpisodicEvent(
            id=str(event_id),
            event_fingerprint=str(event_fingerprint),
            role=str(role),
            content=str(content),
            normalized_content=str(normalized_content),
            scope=str(scope),
            scope_id=(str(scope_id) if scope_id is not None else None),
            source=str(source),
            hook_event=(str(hook_event) if hook_event is not None else None),
            metadata=metadata,
            status=str(status),
            error=(str(error) if error is not None else None),
            created_at=cast(datetime, self._parse_dt(str(created_at))),
            occurred_at=cast(datetime, self._parse_dt(str(occurred_at))),
        )

    def _reconsolidation_job_to_row(self, job: ReconsolidationJob) -> tuple[Any, ...]:
        return (
            job.id,
            job.chunk_id,
            job.context,
            job.status,
            job.error,
            job.attempt_count,
            job.queued_at.isoformat(),
            job.due_at.isoformat(),
            job.last_attempt_at.isoformat() if job.last_attempt_at is not None else None,
        )

    def _row_to_reconsolidation_job(self, row: tuple[Any, ...]) -> ReconsolidationJob:
        (
            job_id,
            chunk_id,
            context,
            status,
            error,
            attempt_count,
            queued_at,
            due_at,
            last_attempt_at,
        ) = row
        return ReconsolidationJob(
            id=str(job_id),
            chunk_id=str(chunk_id),
            context=str(context),
            status=str(status),
            error=(str(error) if error is not None else None),
            attempt_count=int(attempt_count or 0),
            queued_at=cast(datetime, self._parse_dt(str(queued_at))),
            due_at=cast(datetime, self._parse_dt(str(due_at))),
            last_attempt_at=self._parse_dt(cast(str | None, last_attempt_at)),
        )

    def store(self, chunk: MemoryChunk) -> None:
        """Insert or replace a chunk by ID."""
        row = self._chunk_to_row(chunk)
        with self._lock:
            prior_dimension = self._sqlite_vec_dim
            self._conn.execute("SAVEPOINT mnemos_store_write")
            try:
                self._conn.execute(
                    """
                    INSERT INTO memory_chunks
                    (id, content, embedding, metadata, salience, cognitive_state,
                     created_at, updated_at, access_count, version, canonical_key,
                     user_profile_id, surprisal_score, surprisal_status, cognitive_state_status,
                     last_observed_at, observation_count, last_retrieved_at, reconsolidation_due_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content=excluded.content,
                        embedding=excluded.embedding,
                        metadata=excluded.metadata,
                        salience=excluded.salience,
                        cognitive_state=excluded.cognitive_state,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        access_count=excluded.access_count,
                        version=excluded.version,
                        canonical_key=excluded.canonical_key,
                        user_profile_id=excluded.user_profile_id,
                        surprisal_score=excluded.surprisal_score,
                        surprisal_status=excluded.surprisal_status,
                        cognitive_state_status=excluded.cognitive_state_status,
                        last_observed_at=excluded.last_observed_at,
                        observation_count=excluded.observation_count,
                        last_retrieved_at=excluded.last_retrieved_at,
                        reconsolidation_due_at=excluded.reconsolidation_due_at
                    """,
                    row,
                )
                self._sync_vec_row(chunk.id, chunk.embedding)
                self._conn.execute("RELEASE SAVEPOINT mnemos_store_write")
            except Exception:
                self._conn.execute("ROLLBACK TO SAVEPOINT mnemos_store_write")
                self._conn.execute("RELEASE SAVEPOINT mnemos_store_write")
                self._sqlite_vec_dim = prior_dimension
                raise

    def upsert_canonical(self, chunk: MemoryChunk) -> MemoryChunk:
        scope, scope_id = _chunk_scope(chunk)
        profile = chunk.metadata.get("user_profile_id")
        profile_id = profile.strip() if isinstance(profile, str) and profile.strip() else None
        canonical_key = compute_canonical_key(chunk.content, scope, scope_id, profile_id)
        chunk.canonical_key = canonical_key
        if chunk.last_observed_at is None:
            chunk.last_observed_at = chunk.updated_at
        if chunk.observation_count <= 0:
            chunk.observation_count = 1

        with self._lock:
            existing_row = self._conn.execute(
                f"SELECT {self._CHUNK_SELECT_COLUMNS} FROM memory_chunks WHERE canonical_key = ?",
                (canonical_key,),
            ).fetchone()

        if existing_row is None:
            self.store(chunk)
            return chunk

        existing = self._row_to_chunk(existing_row)
        merged_duplicate_ids = list(existing.metadata.get("merged_duplicate_ids", []))
        if chunk.id != existing.id and chunk.id not in merged_duplicate_ids:
            merged_duplicate_ids.append(chunk.id)

        merged_metadata = dict(existing.metadata)
        merged_metadata.update(chunk.metadata)
        merged_metadata["merged_duplicate_ids"] = merged_duplicate_ids

        merged = existing.model_copy(
            update={
                "content": chunk.content,
                "embedding": chunk.embedding or existing.embedding,
                "metadata": merged_metadata,
                "salience": max(existing.salience, chunk.salience),
                "updated_at": max(existing.updated_at, chunk.updated_at),
                "created_at": min(existing.created_at, chunk.created_at),
                "canonical_key": canonical_key,
                "last_observed_at": max(
                    existing.last_observed_at or existing.updated_at,
                    chunk.last_observed_at or chunk.updated_at,
                ),
                "observation_count": (
                    existing.observation_count + max(chunk.observation_count, 1)
                    if chunk.id != existing.id
                    else max(existing.observation_count, chunk.observation_count)
                ),
                "surprisal_score": (
                    chunk.surprisal_score
                    if chunk.surprisal_score is not None
                    else existing.surprisal_score
                ),
                "surprisal_status": (
                    chunk.surprisal_status
                    if chunk.surprisal_status != "unknown"
                    else existing.surprisal_status
                ),
                "cognitive_state": (
                    chunk.cognitive_state
                    if chunk.cognitive_state is not None
                    else existing.cognitive_state
                ),
                "cognitive_state_status": (
                    chunk.cognitive_state_status
                    if chunk.cognitive_state_status != "unknown"
                    else existing.cognitive_state_status
                ),
                "last_retrieved_at": chunk.last_retrieved_at or existing.last_retrieved_at,
                "reconsolidation_due_at": (
                    chunk.reconsolidation_due_at
                    if chunk.reconsolidation_due_at is not None
                    else existing.reconsolidation_due_at
                ),
            }
        )
        self.store(merged)
        return merged

    @staticmethod
    def _scope_sql(filter_fn: Callable[[MemoryChunk], bool] | None) -> tuple[str, list[Any], bool]:
        """Compile engine-owned scope predicates before loading or ranking candidates."""
        if filter_fn is None:
            return "1", [], True
        spec = getattr(filter_fn, "_mnemos_scope_filter", None)
        if not isinstance(spec, dict):
            return "1", [], False
        clauses: list[str] = []
        bindings: list[Any] = []
        scoped = "user_profile_id" in spec
        profile_clause = "user_profile_id = ?" if scoped else "user_profile_id IS NULL"
        if scoped:
            bindings.append(spec["user_profile_id"])
        for scope in spec.get("allowed_scopes", ()):
            if scope == "global":
                clauses.append("json_extract(metadata, '$.scope') = 'global'")
            elif scope in {"project", "workspace"}:
                if scoped and scope == "project" and spec.get("member_workspace_id") is not None:
                    clauses.append(
                        "(json_extract(metadata, '$.scope') = 'project' AND "
                        "json_extract(metadata, '$.workspace_id') = ?)"
                    )
                    bindings.append(spec["member_workspace_id"])
                    continue
                if spec.get("all_profile_scopes") is True:
                    clauses.append("json_extract(metadata, '$.scope') = ?")
                    bindings.append(scope)
                    continue
                identity = spec.get(scope + "_id") if scoped else spec.get("scope_id")
                if identity is not None:
                    # Legacy embedded callers historically used an implicit default id.
                    identity_expr = "json_extract(metadata, '$.scope_id')"
                    if not scoped:
                        identity_expr = (
                            "COALESCE(NULLIF(trim(" + identity_expr + "), ''), 'default')"
                        )
                    clauses.append(
                        "(json_extract(metadata, '$.scope') = ? AND " + identity_expr + " = ?)"
                    )
                    bindings.extend([scope, identity])
        return profile_clause + " AND (" + (" OR ".join(clauses) or "0") + ")", bindings, True

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """Exact cosine top-k, with scope predicates applied in SQLite first.

        Native sqlite-vec scalar distance avoids the old L2 vec0 ranking mismatch.
        Without the extension, only authorized rows reach the Python fallback.
        Arbitrary caller predicates are an explicitly reported Python fallback.
        """
        if top_k <= 0 or not query_embedding:
            return []
        where, bindings, compiled = self._scope_sql(filter_fn)
        if compiled and self._sqlite_vec_enabled:
            with self._lock:
                distance = """COALESCE((SELECT 1.0 - vec_distance_cosine(embedding, ?)
                    FROM memory_vector_values WHERE rowid=memory_chunks.rowid
                    AND dimension=?), 0.0)"""
                vector_bindings = [self._serialize_vec(query_embedding), len(query_embedding)]
                rows = self._conn.execute(
                    f"SELECT {self._CHUNK_SELECT_COLUMNS}, {distance} AS similarity "
                    f"FROM memory_chunks WHERE {where} ORDER BY similarity DESC, id LIMIT ?",
                    [*vector_bindings, *bindings, top_k],
                ).fetchall()
            self.last_retrieval_backend = "sqlite_vec_scoped_exact_cosine"
            return [self._row_to_chunk(row[:-1]) for row in rows]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._CHUNK_SELECT_COLUMNS} FROM memory_chunks WHERE {where}", bindings
            ).fetchall()
        candidates = [self._row_to_chunk(row) for row in rows]
        if filter_fn is not None:
            candidates = [chunk for chunk in candidates if filter_fn(chunk)]
        scored = [
            (
                (
                    cosine_similarity(query_embedding, chunk.embedding)
                    if chunk.embedding and len(chunk.embedding) == len(query_embedding)
                    else 0.0
                ),
                chunk,
            )
            for chunk in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        self.last_retrieval_backend = (
            "python_scoped_exact_cosine" if compiled else "python_custom_predicate_exact_cosine"
        )
        return [chunk for _, chunk in scored[:top_k]]

    def retrieve_lexical(
        self,
        query: str,
        top_k: int = 20,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """Bounded FTS5 candidates; user text is never interpreted as FTS syntax."""
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)[:64]
        if not tokens or top_k <= 0:
            return []
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        where, bindings, compiled = self._scope_sql(filter_fn)
        if not compiled:
            # An arbitrary predicate cannot safely be applied after a bounded SQL limit.
            return []
        columns = ", ".join(
            f"memory_chunks.{column.strip()}" for column in self._CHUNK_SELECT_COLUMNS.split(",")
        )
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {columns} FROM memory_fts "
                "JOIN memory_chunks ON memory_chunks.rowid=memory_fts.rowid "
                f"WHERE {where} AND memory_fts MATCH ? "
                "ORDER BY bm25(memory_fts), memory_chunks.id LIMIT ?",
                [*bindings, match, top_k],
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def update(self, chunk_id: str, chunk: MemoryChunk) -> bool:
        """Update existing chunk; returns False if not found."""
        with self._lock:
            cursor = self._conn.execute("SELECT id FROM memory_chunks WHERE id = ?", (chunk_id,))
            if cursor.fetchone() is None:
                return False
        if chunk.id != chunk_id:
            chunk = chunk.model_copy(update={"id": chunk_id})
        self.store(chunk)
        return True

    def delete(self, chunk_id: str) -> bool:
        """Delete chunk by ID; returns False if not found."""
        with self._lock:
            if self._sqlite_vec_enabled and self._vec_table_exists():
                row = self._conn.execute(
                    "SELECT rowid FROM memory_chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                if row is not None:
                    self._conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (int(row[0]),))
            cursor = self._conn.execute("DELETE FROM memory_chunks WHERE id = ?", (chunk_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _job_to_row(job: DurableJob) -> tuple[Any, ...]:
        return (
            job.id,
            job.kind,
            json.dumps(job.payload),
            job.status,
            job.attempt_count,
            job.max_attempts,
            job.available_at.isoformat(),
            job.lease_owner,
            job.lease_token,
            job.lease_until.isoformat() if job.lease_until is not None else None,
            job.last_error,
            json.dumps(job.result) if job.result is not None else None,
            job.created_at.isoformat(),
            job.updated_at.isoformat(),
        )

    def _row_to_job(self, row: tuple[Any, ...]) -> DurableJob:
        (
            job_id,
            kind,
            payload,
            status,
            attempt_count,
            max_attempts,
            available_at,
            lease_owner,
            lease_token,
            lease_until,
            last_error,
            result,
            created_at,
            updated_at,
        ) = row
        return DurableJob(
            id=str(job_id),
            kind=str(kind),
            payload=json.loads(payload) if payload else {},
            status=cast(JobStatus, str(status)),
            attempt_count=int(attempt_count),
            max_attempts=int(max_attempts),
            available_at=cast(datetime, self._parse_dt(str(available_at))),
            lease_owner=str(lease_owner) if lease_owner is not None else None,
            lease_token=str(lease_token) if lease_token is not None else None,
            lease_until=self._parse_dt(cast(str | None, lease_until)),
            last_error=str(last_error) if last_error is not None else None,
            result=json.loads(result) if result else None,
            created_at=cast(datetime, self._parse_dt(str(created_at))),
            updated_at=cast(datetime, self._parse_dt(str(updated_at))),
        )

    def enqueue_job(self, job: DurableJob) -> DurableJob:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_jobs
                (id, kind, payload, status, attempt_count, max_attempts, available_at,
                 lease_owner, lease_token, lease_until, last_error, result, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                self._job_to_row(job),
            )
            self._conn.commit()
            row = self._conn.execute(
                """SELECT id, kind, payload, status, attempt_count, max_attempts, available_at,
                        lease_owner, lease_token, lease_until, last_error, result, created_at, updated_at
                 FROM memory_jobs WHERE id = ?""",
                (job.id,),
            ).fetchone()
        return self._row_to_job(cast(tuple[Any, ...], row))

    def claim_jobs(
        self,
        *,
        owner: str,
        limit: int = 1,
        lease_seconds: int = 300,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> list[DurableJob]:
        claimed_at = now or datetime.now(timezone.utc)
        lease_until = claimed_at + timedelta(seconds=max(1, lease_seconds))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """UPDATE memory_jobs SET
                       status=CASE WHEN attempt_count >= max_attempts THEN 'exhausted' ELSE 'pending' END,
                       lease_owner=NULL, lease_token=NULL, lease_until=NULL,
                       updated_at=? WHERE status='leased' AND lease_until <= ?""",
                    (claimed_at.isoformat(), claimed_at.isoformat()),
                )
                rows = self._conn.execute(
                    """SELECT id FROM memory_jobs
                       WHERE status='pending' AND attempt_count < max_attempts AND available_at <= ?
                         AND (? IS NULL OR id=?)
                       ORDER BY available_at ASC, created_at ASC LIMIT ?""",
                    (claimed_at.isoformat(), job_id, job_id, max(0, limit)),
                ).fetchall()
                ids = [str(row[0]) for row in rows]
                lease_tokens: dict[str, str] = {}
                for job_id in ids:
                    token = str(uuid4())
                    lease_tokens[job_id] = token
                    self._conn.execute(
                        """UPDATE memory_jobs SET status='leased', lease_owner=?, lease_token=?,
                           lease_until=?, attempt_count=attempt_count + 1, updated_at=? WHERE id=?""",
                        (owner, token, lease_until.isoformat(), claimed_at.isoformat(), job_id),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            claimed_rows = self._conn.execute(
                f"""SELECT id, kind, payload, status, attempt_count, max_attempts, available_at,
                    lease_owner, lease_token, lease_until, last_error, result, created_at, updated_at
                    FROM memory_jobs WHERE id IN ({placeholders})""",
                tuple(ids),
            ).fetchall()
        by_id = {str(row[0]): self._row_to_job(row) for row in claimed_rows}
        return [by_id[job_id] for job_id in ids]

    def complete_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        result: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> bool:
        completed_at = now or datetime.now(timezone.utc)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    """UPDATE memory_jobs SET status='completed', result=?, lease_owner=NULL,
                       lease_token=NULL, lease_until=NULL, updated_at=?
                       WHERE id=? AND status='leased' AND lease_owner=? AND lease_token=?
                         AND lease_until > ?""",
                    (
                        json.dumps(result) if result is not None else None,
                        completed_at.isoformat(),
                        job_id,
                        owner,
                        lease_token,
                        completed_at.isoformat(),
                    ),
                )
                self._conn.commit()
                return cursor.rowcount > 0
            except Exception:
                self._conn.rollback()
                raise

    def fail_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        error: str,
        now: datetime | None = None,
    ) -> bool:
        failed_at = now or datetime.now(timezone.utc)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """SELECT attempt_count, max_attempts FROM memory_jobs
                     WHERE id=? AND status='leased' AND lease_owner=? AND lease_token=?
                       AND lease_until > ?""",
                    (job_id, owner, lease_token, failed_at.isoformat()),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                attempts = int(row[0])
                max_attempts = int(row[1])
                status = "exhausted" if attempts >= max_attempts else "pending"
                backoff = min(3600, 2 ** min(attempts, 10))
                available_at = failed_at + timedelta(seconds=backoff)
                self._conn.execute(
                    """UPDATE memory_jobs SET status=?, available_at=?,
                       lease_owner=NULL, lease_token=NULL, lease_until=NULL,
                       last_error=?, updated_at=? WHERE id=?""",
                    (status, available_at.isoformat(), error, failed_at.isoformat(), job_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def retry_job(self, job_id: str, *, now: datetime | None = None) -> bool:
        retried_at = now or datetime.now(timezone.utc)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    """UPDATE memory_jobs SET status='pending', attempt_count=0,
                       available_at=?, lease_owner=NULL, lease_token=NULL, lease_until=NULL,
                       last_error=NULL, result=NULL, updated_at=?
                       WHERE id=? AND status='exhausted'""",
                    (retried_at.isoformat(), retried_at.isoformat(), job_id),
                )
                self._conn.commit()
                return cursor.rowcount > 0
            except Exception:
                self._conn.rollback()
                raise

    def commit_fact_and_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_token: str | None = None,
        chunk: MemoryChunk | None = None,
        chunks: list[MemoryChunk] | None = None,
        evidence: list[FactEvidence] | None = None,
        revision: FactRevision | None = None,
        revisions: list[FactRevision] | None = None,
        processed_event_ids: list[str] | None = None,
        expected_version: int | None = None,
        expected_versions: dict[str, int] | None = None,
        result: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> bool:
        committed_at = now or datetime.now(timezone.utc)
        fact_chunks = list(chunks or [])
        if chunk is not None:
            fact_chunks.insert(0, chunk)
        result_payload = dict(result or {})
        if not fact_chunks:
            decision = result_payload.get("decision")
            if not processed_event_ids or decision not in {
                "no_durable_facts",
                "surprisal_filtered",
                "suppressed",
            }:
                raise ValueError(
                    "A no-fact commit requires processed events and an explicit decision"
                )
        all_revisions = list(revisions or [])
        if revision is not None:
            all_revisions.insert(0, revision)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._transaction_active = True
            try:
                lease_row = self._conn.execute(
                    """SELECT kind, payload FROM memory_jobs
                       WHERE id=? AND status='leased' AND lease_owner=? AND lease_token=?
                         AND lease_until > ?""",
                    (job_id, owner, lease_token, committed_at.isoformat()),
                ).fetchone()
                if lease_row is None:
                    self._conn.rollback()
                    return False

                if expected_version is not None:
                    current = self._conn.execute(
                        "SELECT version FROM memory_chunks WHERE id=?", (fact_chunks[0].id,)
                    ).fetchone()
                    if current is None or int(current[0]) != expected_version:
                        raise RuntimeError("Fact version changed before atomic commit")

                for target_id, version in (expected_versions or {}).items():
                    current = self._conn.execute(
                        "SELECT version FROM memory_chunks WHERE id=?", (target_id,)
                    ).fetchone()
                    if current is None or int(current[0]) != version:
                        raise RuntimeError("Fact version changed before atomic commit")

                accepted_by_input: dict[str, str] = {}
                accepted_chunks: list[MemoryChunk] = []
                suppressed_ids: set[str] = set()
                for index, candidate in enumerate(fact_chunks):
                    if lease_row[0] == "extract":
                        canonical = compute_canonical_key(
                            candidate.content,
                            candidate.metadata.get("scope", "unassigned"),
                            candidate.metadata.get("scope_id"),
                            candidate.metadata.get("user_profile_id"),
                        )
                        source_id = json.loads(lease_row[1]).get("event_id")
                        if self.is_suppressed(canonical_key=canonical, source_event_id=source_id):
                            suppressed_ids.add(candidate.id)
                            continue
                    prior = self.get(candidate.id)
                    accepted = self.upsert_canonical(candidate)
                    if (
                        (expected_version is not None and index == 0)
                        or candidate.id in (expected_versions or {})
                    ) and accepted.id != candidate.id:
                        raise RuntimeError("Expected-version correction collided with another fact")
                    accepted_chunks.append(accepted)
                    accepted_by_input[candidate.id] = accepted.id
                    accepted_by_input[accepted.id] = accepted.id
                    if prior is not None and (
                        prior.content != accepted.content or prior.embedding != accepted.embedding
                    ):
                        self._conn.execute(
                            "DELETE FROM memory_edges WHERE source_id=? OR target_id=?",
                            (accepted.id, accepted.id),
                        )

                if suppressed_ids and not accepted_chunks:
                    result_payload["decision"] = "suppressed"
                for item in evidence or []:
                    if item.fact_id in suppressed_ids:
                        continue
                    accepted_fact_id = accepted_by_input.get(item.fact_id, item.fact_id)
                    if item.source_event_id is not None:
                        duplicate_evidence = self._conn.execute(
                            """SELECT 1 FROM fact_evidence
                               WHERE fact_id=? AND source_event_id=? LIMIT 1""",
                            (accepted_fact_id, item.source_event_id),
                        ).fetchone()
                        if duplicate_evidence is not None:
                            continue
                    self._conn.execute(
                        """INSERT INTO fact_evidence
                           (evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            item.evidence_id,
                            accepted_fact_id,
                            item.source_event_id,
                            item.excerpt,
                            item.source_ref,
                            item.created_at.isoformat(),
                        ),
                    )
                for fact_revision in all_revisions:
                    if fact_revision.fact_id in suppressed_ids:
                        continue
                    self._conn.execute(
                        """INSERT INTO fact_revisions
                           (revision_id, fact_id, version, content, reason, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            fact_revision.revision_id,
                            accepted_by_input.get(fact_revision.fact_id, fact_revision.fact_id),
                            fact_revision.version,
                            fact_revision.content,
                            fact_revision.reason,
                            fact_revision.created_at.isoformat(),
                        ),
                    )
                for event_id in processed_event_ids or []:
                    updated_event = self._conn.execute(
                        """UPDATE episodic_events SET status='consolidated', error=NULL WHERE id=?""",
                        (event_id,),
                    )
                    if updated_event.rowcount != 1:
                        raise ValueError(
                            f"Cannot mark missing episodic event as processed: {event_id}"
                        )
                result_payload.setdefault("fact_ids", [item.id for item in accepted_chunks])
                updated = self._conn.execute(
                    """UPDATE memory_jobs SET status='completed', result=?, lease_owner=NULL,
                       lease_token=NULL, lease_until=NULL, updated_at=?
                       WHERE id=? AND status='leased' AND lease_owner=? AND lease_token=?
                         AND lease_until > ?""",
                    (
                        json.dumps(result_payload),
                        committed_at.isoformat(),
                        job_id,
                        owner,
                        lease_token,
                        committed_at.isoformat(),
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("Job lease was lost while committing fact")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._transaction_active = False
            return True

    def capture_event_and_job(self, event: EpisodicEvent, job: DurableJob) -> EpisodicEvent:
        """Persist an ingest event and its processing job in one transaction."""
        source_identity_key = _source_identity_key(event)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._transaction_active = True
            try:
                existing_row = self._conn.execute(
                    """SELECT id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                              source, hook_event, source_event_id, source_identity_key,
                              user_profile_id, session_id, metadata, status, error, created_at, occurred_at
                       FROM episodic_events
                       WHERE event_fingerprint=?
                          OR (? IS NOT NULL AND source_identity_key=?)""",
                    (event.event_fingerprint, source_identity_key, source_identity_key),
                ).fetchone()
                if existing_row is None:
                    self._conn.execute(
                        """INSERT INTO episodic_events
                           (id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                            source, hook_event, source_event_id, source_identity_key, user_profile_id,
                            session_id, metadata, status, error, created_at, occurred_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        self._episodic_event_to_row(event),
                    )
                    stored_event = event
                else:
                    stored_event = self._row_to_episodic_event(existing_row)
                    if source_identity_key is not None and existing_row[10] == source_identity_key:
                        if (
                            stored_event.content != event.content
                            or stored_event.role != event.role
                            or stored_event.scope != event.scope
                            or stored_event.scope_id != event.scope_id
                        ):
                            raise ValueError(
                                "Source event identity already has a different payload"
                            )

                existing_job = self._conn.execute(
                    "SELECT kind, payload FROM memory_jobs WHERE id=?", (job.id,)
                ).fetchone()
                if existing_job is None:
                    self._conn.execute(
                        """INSERT INTO memory_jobs
                           (id, kind, payload, status, attempt_count, max_attempts, available_at,
                            lease_owner, lease_token, lease_until, last_error, result, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        self._job_to_row(job),
                    )
                elif (
                    str(existing_job[0]) != job.kind
                    or json.loads(existing_job[1] or "{}") != job.payload
                ):
                    raise ValueError("Job identity already has a different payload")
                self._conn.commit()
                return stored_event
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._transaction_active = False

    def get_episodic_event(self, event_id: str) -> EpisodicEvent | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                          source, hook_event, source_event_id, source_identity_key,
                          user_profile_id, session_id, metadata, status, error, created_at, occurred_at
                   FROM episodic_events WHERE id=?""",
                (event_id,),
            ).fetchone()
        return self._row_to_episodic_event(row) if row is not None else None

    def get_job(self, job_id: str) -> DurableJob | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, kind, payload, status, attempt_count, max_attempts, available_at, "
                "lease_owner, lease_token, lease_until, last_error, result, created_at, updated_at "
                "FROM memory_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 0,
        user_profile_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[DurableJob]:
        query = """SELECT id, kind, payload, status, attempt_count, max_attempts, available_at,
                   lease_owner, lease_token, lease_until, last_error, result, created_at, updated_at
                   FROM memory_jobs"""
        params: list[Any] = []
        clauses = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if user_profile_id is not None:
            clauses.extend(
                [
                    "json_extract(payload, '$.context.user_profile_id')=?",
                    "json_extract(payload, '$.context.project_id') IS ?",
                    "json_extract(payload, '$.context.workspace_id') IS ?",
                ]
            )
            params.extend([user_profile_id, project_id, workspace_id])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY available_at ASC, created_at ASC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_job(row) for row in rows]

    def job_summary(
        self,
        *,
        now: datetime,
        user_profile_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        where, parameters = "", []
        if user_profile_id is not None:
            where = " WHERE json_extract(payload, '$.context.user_profile_id')=? AND json_extract(payload, '$.context.project_id') IS ? AND json_extract(payload, '$.context.workspace_id') IS ?"
            parameters = [user_profile_id, project_id, workspace_id]
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*), MIN(created_at), "
                "COALESCE(SUM(MAX(attempt_count-1, 0)), 0) FROM memory_jobs"
                + where
                + " GROUP BY status",
                parameters,
            ).fetchall()
        result: dict[str, Any] = {
            status: 0 for status in ("pending", "leased", "exhausted", "completed")
        }
        result.update(retry_count=0, oldest_unfinished_seconds=0, oldest_pending_seconds=0)
        for status, count, oldest, retries in rows:
            result[status] = count
            result["retry_count"] += retries
            age = max(0, (now - datetime.fromisoformat(oldest)).total_seconds())
            if status != "completed":
                result["oldest_unfinished_seconds"] = max(result["oldest_unfinished_seconds"], age)
            if status == "pending":
                result["oldest_pending_seconds"] = age
        return result

    def list_fact_evidence(self, fact_id: str) -> list[FactEvidence]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT evidence_id, fact_id, source_event_id, excerpt, source_ref, created_at
                 FROM fact_evidence WHERE fact_id=? ORDER BY created_at ASC""",
                (fact_id,),
            ).fetchall()
        return [
            FactEvidence(
                evidence_id=str(row[0]),
                fact_id=str(row[1]),
                source_event_id=str(row[2]) if row[2] is not None else None,
                excerpt=str(row[3]),
                source_ref=str(row[4]) if row[4] is not None else None,
                created_at=cast(datetime, self._parse_dt(str(row[5]))),
            )
            for row in rows
        ]

    def forget_fact(self, fact_id: str, *, expected_version: int) -> bool:
        """Erase live derivations and atomically leave content-free replay suppressions."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._transaction_active = True
            try:
                chunk = self.get(fact_id)
                if chunk is None:
                    self._conn.rollback()
                    return False
                if chunk.version != expected_version:
                    raise ValueError("Forget version conflict")
                evidence = self.list_fact_evidence(fact_id)
                event_ids = {item.source_event_id for item in evidence if item.source_event_id}
                event_ids.update(
                    row[0]
                    for row in self._conn.execute(
                        "SELECT json_extract(payload, '$.event_id') FROM memory_jobs "
                        "WHERE json_extract(payload, '$.target_fact_id')=?",
                        (fact_id,),
                    ).fetchall()
                    if row[0]
                )
                event_ids.update(
                    row[0]
                    for row in self._conn.execute(
                        "SELECT evidence.value FROM memory_jobs, json_each(payload, '$.evidence_ids') AS evidence "
                        "WHERE json_extract(payload, '$.target_fact_id')=?",
                        (fact_id,),
                    ).fetchall()
                    if row[0]
                )
                keys = {
                    chunk.canonical_key
                    or compute_canonical_key(
                        chunk.content,
                        chunk.metadata.get("scope", "unassigned"),
                        chunk.metadata.get("scope_id"),
                        chunk.metadata.get("user_profile_id"),
                    )
                }
                for revision in self.list_fact_revisions(fact_id):
                    keys.add(
                        compute_canonical_key(
                            revision.content,
                            chunk.metadata.get("scope", "unassigned"),
                            chunk.metadata.get("scope_id"),
                            chunk.metadata.get("user_profile_id"),
                        )
                    )
                stamp = datetime.now(timezone.utc).isoformat()
                for key in keys:
                    self._conn.execute(
                        "INSERT INTO memory_suppressions VALUES (?, ?, ?, NULL, 'explicit_forget', ?)",
                        (str(uuid4()), fact_id, key, stamp),
                    )
                for event_id in event_ids:
                    self._conn.execute(
                        "INSERT INTO memory_suppressions VALUES (?, ?, NULL, ?, 'explicit_forget', ?)",
                        (str(uuid4()), fact_id, event_id, stamp),
                    )
                    self._conn.execute("DELETE FROM episodic_events WHERE id=?", (event_id,))
                    # A shared support excerpt may contain the forgotten assertion too.
                    self._conn.execute(
                        "DELETE FROM fact_evidence WHERE source_event_id=?", (event_id,)
                    )
                    self._conn.execute(
                        "UPDATE memory_jobs SET status='completed', result=?, last_error=NULL, "
                        "lease_owner=NULL, lease_token=NULL, lease_until=NULL, updated_at=? "
                        "WHERE json_extract(payload, '$.event_id')=?",
                        (json.dumps({"decision": "forgotten"}), stamp, event_id),
                    )
                self._conn.execute(
                    "UPDATE memory_jobs SET status='completed', result=?, last_error=NULL, "
                    "lease_owner=NULL, lease_token=NULL, lease_until=NULL, updated_at=?, "
                    "payload=json_remove(payload, '$.content') "
                    "WHERE json_extract(payload, '$.target_fact_id')=?",
                    (json.dumps({"decision": "forgotten"}), stamp, fact_id),
                )
                self._conn.execute(
                    "DELETE FROM retrieval_feedback_events WHERE EXISTS "
                    "(SELECT 1 FROM json_each(chunk_ids) WHERE value=?)",
                    (fact_id,),
                )
                if self._sqlite_vec_enabled and self._vec_table_exists():
                    self._conn.execute(
                        "DELETE FROM memory_vec WHERE rowid IN "
                        "(SELECT rowid FROM memory_chunks WHERE id=?)",
                        (fact_id,),
                    )
                self._conn.execute("DELETE FROM memory_chunks WHERE id=?", (fact_id,))
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._transaction_active = False

    def prune_processed_evidence(self, *, before: datetime, limit: int = 100) -> int:
        """Expire processed raw events; retained support excerpts remain inspectable."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM episodic_events WHERE id IN (SELECT id FROM episodic_events "
                "WHERE status='consolidated' AND created_at < ? "
                "AND NOT EXISTS (SELECT 1 FROM memory_jobs WHERE "
                "json_extract(payload, '$.event_id')=episodic_events.id "
                "AND status != 'completed') ORDER BY created_at LIMIT ?)",
                (before.isoformat(), max(0, min(limit, 1000))),
            )
            self._conn.commit()
            return cursor.rowcount

    def record_suppression(
        self,
        *,
        fact_id: str | None = None,
        canonical_key: str | None = None,
        source_event_id: str | None = None,
        reason: str,
    ) -> str:
        suppression_id = str(uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO memory_suppressions
                   (suppression_id, fact_id, canonical_key, source_event_id, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    suppression_id,
                    fact_id,
                    canonical_key,
                    source_event_id,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        return suppression_id

    def is_suppressed(
        self,
        *,
        fact_id: str | None = None,
        canonical_key: str | None = None,
        source_event_id: str | None = None,
    ) -> bool:
        selectors = [
            ("fact_id", fact_id),
            ("canonical_key", canonical_key),
            ("source_event_id", source_event_id),
        ]
        selectors = [(column, value) for column, value in selectors if value is not None]
        if not selectors:
            return False
        where = " OR ".join(f"{column} = ?" for column, _ in selectors)
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM memory_suppressions WHERE {where} LIMIT 1",
                tuple(value for _, value in selectors),
            ).fetchone()
        return row is not None

    def list_fact_revisions(self, fact_id: str) -> list[FactRevision]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT revision_id, fact_id, version, content, reason, created_at
                 FROM fact_revisions WHERE fact_id=? ORDER BY version ASC""",
                (fact_id,),
            ).fetchall()
        return [
            FactRevision(
                revision_id=str(row[0]),
                fact_id=str(row[1]),
                version=int(row[2]),
                content=str(row[3]),
                reason=str(row[4]) if row[4] is not None else None,
                created_at=cast(datetime, self._parse_dt(str(row[5]))),
            )
            for row in rows
        ]

    def mark_retrieved_many(
        self, chunk_ids: list[str], *, retrieved_at: datetime | None = None
    ) -> int:
        identities = list(dict.fromkeys(chunk_ids))
        if not identities:
            return 0
        touched_at = retrieved_at or datetime.now(timezone.utc)
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "UPDATE memory_chunks SET access_count=access_count+1, "
                    "last_retrieved_at=?, reconsolidation_due_at=NULL "
                    "WHERE id IN (" + ",".join("?" for _ in identities) + ")",
                    [touched_at.isoformat(), *identities],
                )
                if not self._transaction_active:
                    self._conn.commit()
                return cursor.rowcount
            except Exception:
                if not self._transaction_active:
                    self._conn.rollback()
                raise

    def mark_retrieved(
        self,
        chunk_id: str,
        *,
        retrieved_at: datetime | None = None,
        reconsolidation_due_at: datetime | None = None,
    ) -> bool:
        touched_at = retrieved_at or datetime.now(timezone.utc)
        due_at_iso = (
            reconsolidation_due_at.isoformat() if reconsolidation_due_at is not None else None
        )
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE memory_chunks
                SET access_count = access_count + 1,
                    last_retrieved_at = ?,
                    reconsolidation_due_at = ?
                WHERE id = ?
                """,
                (touched_at.isoformat(), due_at_iso, chunk_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def touch(
        self,
        chunk_id: str,
        *,
        access_count: int | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Persist access_count/updated_at with a narrow UPDATE statement."""
        touched_at = updated_at or datetime.now(timezone.utc)
        with self._lock:
            if access_count is None:
                cursor = self._conn.execute(
                    """
                    UPDATE memory_chunks
                    SET updated_at = ?, access_count = access_count + 1
                    WHERE id = ?
                    """,
                    (touched_at.isoformat(), chunk_id),
                )
            else:
                cursor = self._conn.execute(
                    """
                    UPDATE memory_chunks
                    SET updated_at = ?, access_count = ?
                    WHERE id = ?
                    """,
                    (touched_at.isoformat(), access_count, chunk_id),
                )
            self._conn.commit()
            return cursor.rowcount > 0

    def get(self, chunk_id: str) -> MemoryChunk | None:
        """O(1) lookup by primary key."""
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT {self._CHUNK_SELECT_COLUMNS} FROM memory_chunks WHERE id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_all(self) -> list[MemoryChunk]:
        """Load all chunks from the database."""
        with self._lock:
            cursor = self._conn.execute(f"SELECT {self._CHUNK_SELECT_COLUMNS} FROM memory_chunks")
            rows = cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def store_feedback_event(self, event: RetrievalFeedbackEvent) -> None:
        row = self._feedback_event_to_row(event)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO retrieval_feedback_events
                (id, event_type, query, scope, scope_id, chunk_ids, notes, created_at, user_profile_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    event_type=excluded.event_type,
                    query=excluded.query,
                    scope=excluded.scope,
                    scope_id=excluded.scope_id,
                    chunk_ids=excluded.chunk_ids,
                    notes=excluded.notes,
                    created_at=excluded.created_at,
                    user_profile_id=excluded.user_profile_id
                """,
                row,
            )
            self._conn.commit()

    def list_feedback_events(
        self,
        *,
        event_type: str | None = None,
        user_profile_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> list[RetrievalFeedbackEvent]:
        query = """
            SELECT id, event_type, query, scope, scope_id, chunk_ids, notes, created_at, user_profile_id
            FROM retrieval_feedback_events
        """
        clauses: list[str] = []
        params: list[Any] = []

        if user_profile_id is not None:
            clauses.append("user_profile_id = ?")
            params.append(user_profile_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"

        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_feedback_event(row) for row in rows]

    def append_episodic_event(self, event: EpisodicEvent) -> EpisodicEvent:
        source_event_id = event.metadata.get("source_event_id") or event.metadata.get("event_id")
        source_identity_key = _source_identity_key(event)
        with self._lock:
            existing_row = self._conn.execute(
                """
                SELECT id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                       source, hook_event, source_event_id, source_identity_key,
                       user_profile_id, session_id,
                       metadata, status, error, created_at, occurred_at
                FROM episodic_events
                WHERE event_fingerprint = ?
                   OR (? IS NOT NULL AND source_identity_key = ?)
                """,
                (event.event_fingerprint, source_identity_key, source_identity_key),
            ).fetchone()
            if existing_row is not None:
                if source_identity_key is not None and existing_row[10] == source_identity_key:
                    if (
                        str(existing_row[3]) != event.content
                        or str(existing_row[2]) != event.role
                        or str(existing_row[5]) != event.scope
                        or existing_row[6] != event.scope_id
                    ):
                        raise ValueError("Source event identity already has a different payload")
                return self._row_to_episodic_event(existing_row)

            self._conn.execute(
                """
                INSERT INTO episodic_events
                (id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                 source, hook_event, source_event_id, source_identity_key, user_profile_id,
                 session_id, metadata, status, error, created_at, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._episodic_event_to_row(event),
            )
            if not self._transaction_active:
                self._conn.commit()
            return event

    def list_episodic_events(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        user_profile_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
        newest_first: bool = False,
        limit: int = 0,
    ) -> list[EpisodicEvent]:
        query = """
            SELECT id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                   source, hook_event, source_event_id, source_identity_key,
                   user_profile_id, session_id,
                   metadata, status, error, created_at, occurred_at
            FROM episodic_events
        """
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if user_profile_id is not None:
            clauses.append("user_profile_id = ?")
            params.append(user_profile_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if before is not None:
            clauses.append("occurred_at < ?")
            params.append(before.isoformat())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += (
            " ORDER BY occurred_at "
            + ("DESC" if newest_first else "ASC")
            + ", id "
            + ("DESC" if newest_first else "ASC")
        )
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_episodic_event(row) for row in rows]

    def update_episodic_event(
        self,
        event_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE episodic_events
                SET status = ?, error = ?
                WHERE id = ?
                """,
                (status, error, event_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def enqueue_reconsolidation_job(self, job: ReconsolidationJob) -> ReconsolidationJob:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reconsolidation_jobs
                (id, chunk_id, context, status, error, attempt_count, queued_at, due_at, last_attempt_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    context=excluded.context,
                    status='pending',
                    error=NULL,
                    queued_at=excluded.queued_at,
                    due_at=excluded.due_at
                """,
                self._reconsolidation_job_to_row(job),
            )
            self._conn.commit()
            row = self._conn.execute(
                """
                SELECT id, chunk_id, context, status, error, attempt_count, queued_at, due_at,
                       last_attempt_at
                FROM reconsolidation_jobs
                WHERE chunk_id = ?
                """,
                (job.chunk_id,),
            ).fetchone()
        return self._row_to_reconsolidation_job(cast(tuple[Any, ...], row))

    def list_reconsolidation_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 0,
    ) -> list[ReconsolidationJob]:
        query = """
            SELECT id, chunk_id, context, status, error, attempt_count, queued_at, due_at,
                   last_attempt_at
            FROM reconsolidation_jobs
        """
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY due_at ASC, queued_at ASC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_reconsolidation_job(row) for row in rows]

    def update_reconsolidation_job(
        self,
        chunk_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        attempted_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE reconsolidation_jobs
                SET status = ?,
                    error = ?,
                    attempt_count = attempt_count + 1,
                    last_attempt_at = ?
                WHERE chunk_id = ?
                """,
                (status, error, attempted_at, chunk_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_reconsolidation_job(self, chunk_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM reconsolidation_jobs WHERE chunk_id = ?",
                (chunk_id,),
            )
            self._conn.execute(
                "UPDATE memory_chunks SET reconsolidation_due_at = NULL WHERE id = ?",
                (chunk_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def list_chunks(
        self,
        *,
        limit: int = 100,
        after_id: str = "",
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        where, bindings, compiled = self._scope_sql(filter_fn)
        if not compiled:
            return super().list_chunks(limit=limit, after_id=after_id, filter_fn=filter_fn)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._CHUNK_SELECT_COLUMNS} FROM memory_chunks WHERE {where} "
                "AND id > ? ORDER BY id LIMIT ?",
                [*bindings, after_id, max(0, limit)],
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def list_legacy_chunks(self, *, limit: int = 100, after_id: str = "") -> list[MemoryChunk]:
        """List unassigned legacy facts with filtering before pagination."""
        page_size = max(0, limit)
        if page_size == 0:
            return []
        query = f"""
            SELECT {self._CHUNK_SELECT_COLUMNS}
            FROM memory_chunks
            WHERE id > ?
              AND (user_profile_id IS NULL OR TRIM(user_profile_id) = '')
              AND (
                    json_extract(metadata, '$.user_profile_id') IS NULL
                    OR TRIM(CAST(json_extract(metadata, '$.user_profile_id') AS TEXT)) = ''
                  )
              AND (
                    json_extract(metadata, '$.profile_id') IS NULL
                    OR TRIM(CAST(json_extract(metadata, '$.profile_id') AS TEXT)) = ''
                  )
            ORDER BY id
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(query, (after_id, page_size)).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_graph_neighbors(
        self, source_id: str, *, filter_fn: Callable[[MemoryChunk], bool], limit: int = 32
    ) -> list[tuple[MemoryChunk, float]]:
        where, bindings, compiled = self._scope_sql(filter_fn)
        if not compiled:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._CHUNK_SELECT_COLUMNS}, (SELECT MAX(weight) FROM memory_edges "
                "WHERE (source_id=? AND target_id=memory_chunks.id) "
                "OR (target_id=? AND source_id=memory_chunks.id)) AS edge_weight "
                f"FROM memory_chunks WHERE {where} AND id IN "
                "(SELECT target_id FROM memory_edges WHERE source_id=? UNION "
                "SELECT source_id FROM memory_edges WHERE target_id=?) "
                "ORDER BY edge_weight DESC, id LIMIT ?",
                [source_id, source_id, *bindings, source_id, source_id, max(0, limit)],
            ).fetchall()
        return [(self._row_to_chunk(row[:-1]), float(row[-1])) for row in rows]

    def get_graph_edges(self, chunk_ids: list[str] | None = None) -> dict[str, dict[str, float]]:
        query = """
            SELECT source_id, target_id, weight
            FROM memory_edges
            WHERE edge_type = ?
        """
        params: list[Any] = [self._GRAPH_EDGE_TYPE]
        if chunk_ids:
            placeholders = ", ".join("?" for _ in chunk_ids)
            query += f" AND source_id IN ({placeholders})" f" AND target_id IN ({placeholders})"
            params.extend(chunk_ids)
            params.extend(chunk_ids)

        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()

        edge_map: dict[str, dict[str, float]] = {}
        for source_id, target_id, weight in rows:
            if source_id == target_id:
                continue
            edge_map.setdefault(str(source_id), {})[str(target_id)] = float(weight)
        return edge_map

    def replace_graph_neighbors(self, chunk_id: str, neighbors: dict[str, float]) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_edges WHERE source_id = ? AND edge_type = ?",
                (chunk_id, self._GRAPH_EDGE_TYPE),
            )

            if neighbors:
                target_ids = [target_id for target_id in neighbors if target_id != chunk_id]
                existing_targets: set[str] = set()
                if target_ids:
                    placeholders = ", ".join("?" for _ in target_ids)
                    existing_targets = {
                        str(row[0])
                        for row in self._conn.execute(
                            f"SELECT id FROM memory_chunks WHERE id IN ({placeholders})",
                            tuple(target_ids),
                        ).fetchall()
                    }

                now = datetime.now(timezone.utc).isoformat()
                rows = [
                    (chunk_id, target_id, float(weight), self._GRAPH_EDGE_TYPE, now)
                    for target_id, weight in neighbors.items()
                    if target_id != chunk_id and target_id in existing_targets
                ]
                if rows:
                    self._conn.executemany(
                        """
                        INSERT INTO memory_edges(source_id, target_id, weight, edge_type, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                            weight=excluded.weight,
                            updated_at=excluded.updated_at
                        """,
                        rows,
                    )

            self._conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """Return statistics from the database."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
            avg_salience = (
                self._conn.execute("SELECT AVG(salience) FROM memory_chunks").fetchone()[0] or 0.0
            )
            avg_access = (
                self._conn.execute("SELECT AVG(access_count) FROM memory_chunks").fetchone()[0]
                or 0.0
            )
            with_embedding = self._conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            directed_edges = self._conn.execute(
                "SELECT COUNT(*) FROM memory_edges WHERE edge_type = ?",
                (self._GRAPH_EDGE_TYPE,),
            ).fetchone()[0]
            episodic_pending = self._conn.execute(
                "SELECT COUNT(*) FROM episodic_events WHERE status = 'pending'"
            ).fetchone()[0]
            reconsolidation_pending = self._conn.execute(
                "SELECT COUNT(*) FROM reconsolidation_jobs WHERE status = 'pending'"
            ).fetchone()[0]
            schema_version_raw = self._conn.execute(
                "SELECT value FROM mnemos_meta WHERE key = 'schema_version'"
            ).fetchone()
            fts_exists = (
                self._conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
                ).fetchone()[0]
                > 0
            )

        return {
            "backend": "SQLiteStore",
            "name": self.name,
            "db_path": self.db_path,
            "migration_backup": self.migration_backup,
            "total_chunks": total,
            "chunks_with_embeddings": with_embedding,
            "average_salience": round(avg_salience, 4),
            "average_access_count": round(avg_access, 4),
            "related_edges": int(directed_edges // 2),
            "semantic_similar_edges": int(directed_edges // 2),
            "pending_episodic_events": int(episodic_pending),
            "pending_reconsolidation_jobs": int(reconsolidation_pending),
            "fts_enabled": bool(fts_exists),
            "schema_version": int(schema_version_raw[0]) if schema_version_raw else None,
            "sqlite_vec_enabled": self._sqlite_vec_enabled,
            "sqlite_vec_indexed": self._vec_table_exists(),
            "sqlite_vec_dim": self._sqlite_vec_dim,
        }

    def rebuild_indexes(
        self,
        *,
        rebuild_fts: bool = True,
        rebuild_vec: bool = True,
    ) -> dict[str, Any]:
        performed = False
        with self._lock:
            if rebuild_fts:
                self._conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild');")
                performed = True
            if rebuild_vec and self._sqlite_vec_enabled:
                if self._vec_table_exists():
                    self._conn.execute("DROP TABLE memory_vec")
                    self._conn.execute(
                        "DELETE FROM mnemos_meta WHERE key = ?",
                        (self._SQLITE_VEC_DIM_KEY,),
                    )
                    self._sqlite_vec_dim = None
                first_embedding = self._conn.execute(
                    "SELECT embedding FROM memory_chunks WHERE embedding IS NOT NULL LIMIT 1"
                ).fetchone()
                if first_embedding is not None and first_embedding[0]:
                    embedding = json.loads(first_embedding[0])
                    if isinstance(embedding, list) and embedding:
                        self._ensure_vec_table(len(embedding))
                        self._rebuild_vec_index()
                        performed = True
            self._conn.commit()
        return {
            "rebuild_fts": rebuild_fts,
            "rebuild_vec": rebuild_vec,
            "performed": performed,
        }

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        """Ensure connection is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
