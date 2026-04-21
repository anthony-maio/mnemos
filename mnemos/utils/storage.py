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
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast

from ..types import (
    CognitiveState,
    EpisodicEvent,
    MemoryChunk,
    ReconsolidationJob,
    RetrievalFeedbackEvent,
)
from .embeddings import cosine_similarity


def normalize_memory_text(text: str) -> str:
    """Normalize text for exact-dedupe and canonical-key generation."""
    return " ".join(text.strip().lower().split())


def compute_canonical_key(content: str, scope: str, scope_id: str | None) -> str:
    normalized = normalize_memory_text(content)
    scope_token = scope.strip().lower()
    scope_id_token = "" if scope_id is None else scope_id.strip().lower()
    payload = f"{scope_token}|{scope_id_token}|{normalized}".encode("utf-8")
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

    def get_graph_edges(self, chunk_ids: list[str] | None = None) -> dict[str, dict[str, float]]:
        """Return persisted graph edges keyed by source chunk ID."""
        return {}

    def replace_graph_neighbors(self, chunk_id: str, neighbors: dict[str, float]) -> None:
        """Replace the outgoing graph-neighbor set for a chunk."""
        return None

    def store_feedback_event(self, event: RetrievalFeedbackEvent) -> None:
        """Persist one retrieval-feedback event."""
        raise NotImplementedError("This storage backend does not support feedback events.")

    def list_feedback_events(
        self,
        *,
        event_type: str | None = None,
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

    def list_episodic_events(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        limit: int = 0,
    ) -> list[EpisodicEvent]:
        """List persisted episodic events with optional filters."""
        return []

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
        chunk.updated_at = touched_at
        chunk.last_retrieved_at = touched_at
        chunk.reconsolidation_due_at = reconsolidation_due_at
        return self.update(chunk_id, chunk)

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
            chunk.updated_at = touched_at
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
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> list[RetrievalFeedbackEvent]:
        with self._lock:
            events = list(self._feedback_events)

        if event_type is not None:
            events = [event for event in events if event.event_type == event_type]
        if scope is not None:
            events = [event for event in events if event.scope == scope]
        if scope_id is not None:
            events = [event for event in events if event.scope_id == scope_id]
        return events

    def upsert_canonical(self, chunk: MemoryChunk) -> MemoryChunk:
        canonical_key = chunk.canonical_key
        if not canonical_key:
            scope, scope_id = _chunk_scope(chunk)
            canonical_key = compute_canonical_key(chunk.content, scope, scope_id)
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
            existing = self._episodic_events.get(event.event_fingerprint)
            if existing is not None:
                return existing
            self._episodic_events[event.event_fingerprint] = event
            return event

    def list_episodic_events(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
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
        events.sort(key=lambda event: event.created_at)
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
        AFTER UPDATE ON memory_chunks
        BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
            INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
    """

    _SCHEMA_VERSION = "3"
    _GRAPH_EDGE_TYPE = "SEMANTIC_SIMILAR"
    _SQLITE_VEC_DIM_KEY = "sqlite_vec_dim"

    def __init__(self, db_path: str = "mnemos_memory.db", name: str = "sqlite") -> None:
        self.db_path = db_path
        self.name = name
        self._lock = threading.Lock()
        self._sqlite_vec_enabled = False
        self._sqlite_vec_dim: int | None = None
        self._sqlite_vec_module: Any | None = None
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrent access
        self._conn.execute("PRAGMA synchronous=NORMAL;")  # Lower fsync cost for high-read local use
        self._load_sqlite_vec_extension()
        self._initialize_schema()

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

    def _initialize_schema(self) -> None:
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
        self._conn.execute(self._CREATE_FTS_UPDATE_TRIGGER_SQL)
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

    def _sync_vec_row(self, chunk_id: str, embedding: list[float] | None) -> None:
        if not self._sqlite_vec_enabled:
            return

        row = self._conn.execute(
            "SELECT rowid FROM memory_chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return
        rowid = int(row[0])

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

        embedding = json.loads(embedding_json) if embedding_json else None
        metadata = json.loads(metadata_json) if metadata_json else {}
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
        )

    def _episodic_event_to_row(self, event: EpisodicEvent) -> tuple[Any, ...]:
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
            metadata_json,
            status,
            error,
            created_at,
            occurred_at,
        ) = row
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
            metadata=json.loads(metadata_json) if metadata_json else {},
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
            self._conn.execute(
                """
                INSERT INTO memory_chunks
                (id, content, embedding, metadata, salience, cognitive_state,
                 created_at, updated_at, access_count, version, canonical_key,
                 surprisal_score, surprisal_status, cognitive_state_status,
                 last_observed_at, observation_count, last_retrieved_at, reconsolidation_due_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            self._conn.commit()

    def upsert_canonical(self, chunk: MemoryChunk) -> MemoryChunk:
        canonical_key = chunk.canonical_key
        if not canonical_key:
            scope, scope_id = _chunk_scope(chunk)
            canonical_key = compute_canonical_key(chunk.content, scope, scope_id)
            chunk.canonical_key = canonical_key
        if chunk.last_observed_at is None:
            chunk.last_observed_at = chunk.updated_at
        if chunk.observation_count <= 0:
            chunk.observation_count = 1

        with self._lock:
            existing_row = self._conn.execute(
                "SELECT * FROM memory_chunks WHERE canonical_key = ?",
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

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """Retrieve top-k chunks by cosine similarity (loads all, scores in Python)."""
        if (
            filter_fn is None
            and top_k > 0
            and self._sqlite_vec_enabled
            and self._sqlite_vec_dim == len(query_embedding)
            and self._vec_table_exists()
        ):
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT rowid, distance
                    FROM memory_vec
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                    """,
                    (self._serialize_vec(query_embedding), top_k),
                ).fetchall()
                ordered_rows = []
                for rowid, _distance in rows:
                    row = self._conn.execute(
                        "SELECT * FROM memory_chunks WHERE rowid = ?",
                        (int(rowid),),
                    ).fetchone()
                    if row is not None:
                        ordered_rows.append(row)
            return [self._row_to_chunk(row) for row in ordered_rows]

        all_chunks = self.get_all()

        if filter_fn is not None:
            all_chunks = [c for c in all_chunks if filter_fn(c)]

        scored: list[tuple[float, MemoryChunk]] = []
        for chunk in all_chunks:
            if chunk.embedding is not None and len(chunk.embedding) == len(query_embedding):
                sim = cosine_similarity(query_embedding, chunk.embedding)
            else:
                sim = 0.0
            scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

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
                SET updated_at = ?,
                    access_count = access_count + 1,
                    last_retrieved_at = ?,
                    reconsolidation_due_at = ?
                WHERE id = ?
                """,
                (touched_at.isoformat(), touched_at.isoformat(), due_at_iso, chunk_id),
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
            cursor = self._conn.execute("SELECT * FROM memory_chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_all(self) -> list[MemoryChunk]:
        """Load all chunks from the database."""
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM memory_chunks")
            rows = cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def store_feedback_event(self, event: RetrievalFeedbackEvent) -> None:
        row = self._feedback_event_to_row(event)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO retrieval_feedback_events
                (id, event_type, query, scope, scope_id, chunk_ids, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    event_type=excluded.event_type,
                    query=excluded.query,
                    scope=excluded.scope,
                    scope_id=excluded.scope_id,
                    chunk_ids=excluded.chunk_ids,
                    notes=excluded.notes,
                    created_at=excluded.created_at
                """,
                row,
            )
            self._conn.commit()

    def list_feedback_events(
        self,
        *,
        event_type: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> list[RetrievalFeedbackEvent]:
        query = """
            SELECT id, event_type, query, scope, scope_id, chunk_ids, notes, created_at
            FROM retrieval_feedback_events
        """
        clauses: list[str] = []
        params: list[Any] = []

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
        with self._lock:
            existing_row = self._conn.execute(
                """
                SELECT id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                       source, hook_event, metadata, status, error, created_at, occurred_at
                FROM episodic_events
                WHERE event_fingerprint = ?
                """,
                (event.event_fingerprint,),
            ).fetchone()
            if existing_row is not None:
                return self._row_to_episodic_event(existing_row)

            self._conn.execute(
                """
                INSERT INTO episodic_events
                (id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                 source, hook_event, metadata, status, error, created_at, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._episodic_event_to_row(event),
            )
            self._conn.commit()
            return event

    def list_episodic_events(
        self,
        *,
        status: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        limit: int = 0,
    ) -> list[EpisodicEvent]:
        query = """
            SELECT id, event_fingerprint, role, content, normalized_content, scope, scope_id,
                   source, hook_event, metadata, status, error, created_at, occurred_at
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
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC"
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
