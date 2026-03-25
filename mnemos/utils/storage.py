"""
mnemos/utils/storage.py — Storage backends for the Mnemos memory system.

Storage backends handle persistence of MemoryChunks. The interface is
intentionally simple and synchronous — async wrappers are used at the
module level where needed.

Backends:
- MemoryStore (abstract): defines the interface
- InMemoryStore: dict-based, cosine similarity search (development default)
- SQLiteStore: persistent storage using Python's built-in sqlite3
- Neo4jStore: persistent property-graph storage using Neo4j
- QdrantStore: scalable vector retrieval with Qdrant

The InMemoryStore is analogous to the hippocampus — fast, temporary,
in-process working memory. The SQLiteStore/QdrantStore are analogous to
neocortical long-term storage — persistent and restart-safe.

All backends implement the same interface, so you can swap them transparently.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any, TypeVar, cast

from ..types import CognitiveState, MemoryChunk, RetrievalFeedbackEvent
from ..observability import log_event
from .embeddings import cosine_similarity
from .reliability import RetryPolicy, call_with_retry, is_retryable_qdrant_exception

T = TypeVar("T")


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

    def clear(self) -> None:
        """Clear all chunks atomically."""
        with self._lock:
            self._store.clear()


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
            version INTEGER NOT NULL DEFAULT 1
        );
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
            edge_type TEXT NOT NULL DEFAULT 'RELATED_TO',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (source_id, target_id, edge_type),
            FOREIGN KEY (source_id) REFERENCES memory_chunks(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES memory_chunks(id) ON DELETE CASCADE
        );
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

    _SCHEMA_VERSION = "2"
    _GRAPH_EDGE_TYPE = "RELATED_TO"
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

    def _initialize_schema(self) -> None:
        self._conn.execute(self._CREATE_TABLE_SQL)
        self._conn.execute(self._CREATE_META_TABLE_SQL)
        self._conn.execute(self._CREATE_EDGES_TABLE_SQL)
        self._conn.execute(self._CREATE_EDGES_SOURCE_INDEX_SQL)
        self._conn.execute(self._CREATE_EDGES_TARGET_INDEX_SQL)
        self._conn.execute(self._CREATE_FEEDBACK_TABLE_SQL)
        self._conn.execute(self._CREATE_FEEDBACK_TYPE_INDEX_SQL)
        self._conn.execute(self._CREATE_FEEDBACK_SCOPE_INDEX_SQL)
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
        # Keep the external-content FTS table consistent for existing databases.
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
                self._rebuild_vec_index()
        self._conn.commit()

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
        ) = row

        embedding = json.loads(embedding_json) if embedding_json else None
        metadata = json.loads(metadata_json) if metadata_json else {}
        cognitive_state = (
            CognitiveState.model_validate_json(cognitive_state_json)
            if cognitive_state_json
            else None
        )

        def _parse_dt(s: str) -> datetime:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        return MemoryChunk(
            id=id_,
            content=content,
            embedding=embedding,
            metadata=metadata,
            salience=salience,
            cognitive_state=cognitive_state,
            created_at=_parse_dt(created_at),
            updated_at=_parse_dt(updated_at),
            access_count=access_count,
            version=version,
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

    def store(self, chunk: MemoryChunk) -> None:
        """Insert or replace a chunk by ID."""
        row = self._chunk_to_row(chunk)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_chunks
                (id, content, embedding, metadata, salience, cognitive_state,
                 created_at, updated_at, access_count, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    embedding=excluded.embedding,
                    metadata=excluded.metadata,
                    salience=excluded.salience,
                    cognitive_state=excluded.cognitive_state,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    access_count=excluded.access_count,
                    version=excluded.version
                """,
                row,
            )
            self._sync_vec_row(chunk.id, chunk.embedding)
            self._conn.commit()

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
            "fts_enabled": bool(fts_exists),
            "schema_version": int(schema_version_raw[0]) if schema_version_raw else None,
            "sqlite_vec_enabled": self._sqlite_vec_enabled,
            "sqlite_vec_indexed": self._vec_table_exists(),
            "sqlite_vec_dim": self._sqlite_vec_dim,
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


class Neo4jStore(MemoryStore):
    """
    Persistent storage backend using Neo4j as a property graph store.

    This backend stores each MemoryChunk as a node with scalar properties plus
    JSON-serialized metadata/cognitive state. Retrieval currently preserves the
    generic MemoryStore contract by loading candidate chunks and scoring them
    in Python, mirroring SQLite semantics while enabling Neo4j-backed persistence.

    Requires optional dependency:
        pip install "mnemos-memory[neo4j]"
    """

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        label: str = "MnemosMemoryChunk",
        name: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "Neo4jStore requires the 'neo4j' package. "
                "Install with: pip install 'mnemos-memory[neo4j]'"
            ) from e

        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.label = label
        self.name = name
        self._lock = threading.RLock()
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        verify_connectivity = getattr(self._driver, "verify_connectivity", None)
        if callable(verify_connectivity):
            verify_connectivity()
        self._ensure_schema()

    @property
    def _escaped_label(self) -> str:
        return self.label.replace("`", "")

    @property
    def _constraint_name(self) -> str:
        if self._escaped_label == "MnemosMemoryChunk":
            return "mnemos_memory_chunk_id"
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", self._escaped_label).strip("_").lower()
        if not sanitized:
            sanitized = "chunks"
        return f"mnemos_memory_chunk_id_{sanitized}"

    def _run(self, query: str, **params: Any) -> Any:
        class _BufferedResult:
            def __init__(self, records: list[Any], counters: Any) -> None:
                self._records = records
                self._counters = counters

            def __iter__(self) -> Iterator[Any]:
                return iter(self._records)

            def single(self) -> Any:
                return self._records[0] if self._records else None

            def consume(self) -> Any:
                return self._counters

        with self._lock:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, **params)
                records = list(result)
                counters = result.consume()
                return _BufferedResult(records, counters)

    def _ensure_schema(self) -> None:
        check_query = """
            SHOW CONSTRAINTS
            YIELD name, labelsOrTypes, properties
            WHERE name = $constraint_name
               OR (
                   size(labelsOrTypes) = 1
                   AND labelsOrTypes[0] = $label
                   AND size(properties) = 1
                   AND properties[0] = 'id'
               )
            RETURN count(*) AS total
        """
        existing = (
            self._run(
                check_query,
                constraint_name=self._constraint_name,
                label=self._escaped_label,
            ).single()
            or {}
        )
        if int(existing.get("total", 0) or 0) > 0:
            return

        query = (
            f"CREATE CONSTRAINT {self._constraint_name} IF NOT EXISTS "
            f"FOR (chunk:{self._escaped_label}) REQUIRE chunk.id IS UNIQUE"
        )
        self._run(query)

    def _graph_edge_type(self) -> str:
        return "RELATED_TO"

    def _chunk_to_params(self, chunk: MemoryChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "metadata_json": json.dumps(chunk.metadata),
            "salience": chunk.salience,
            "cognitive_state_json": (
                chunk.cognitive_state.model_dump_json()
                if chunk.cognitive_state is not None
                else None
            ),
            "created_at": chunk.created_at.isoformat(),
            "updated_at": chunk.updated_at.isoformat(),
            "access_count": chunk.access_count,
            "version": chunk.version,
        }

    def _record_to_chunk(self, record: Any) -> MemoryChunk:
        metadata_json = record.get("metadata_json", "{}")
        cognitive_state_json = record.get("cognitive_state_json")
        created_at_raw = record.get("created_at")
        updated_at_raw = record.get("updated_at")

        metadata = json.loads(metadata_json) if metadata_json else {}
        cognitive_state = (
            CognitiveState.model_validate_json(cognitive_state_json)
            if isinstance(cognitive_state_json, str) and cognitive_state_json
            else None
        )

        def _parse_dt(value: str | None) -> datetime:
            if not value:
                return datetime.now(timezone.utc)
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        embedding_raw = record.get("embedding")
        embedding = [float(x) for x in embedding_raw] if isinstance(embedding_raw, list) else None

        return MemoryChunk(
            id=str(record["id"]),
            content=str(record.get("content", "")),
            embedding=embedding,
            metadata=metadata if isinstance(metadata, dict) else {},
            salience=float(record.get("salience", 0.5)),
            cognitive_state=cognitive_state,
            created_at=_parse_dt(created_at_raw),
            updated_at=_parse_dt(updated_at_raw),
            access_count=int(record.get("access_count", 0)),
            version=int(record.get("version", 1)),
        )

    def store(self, chunk: MemoryChunk) -> None:
        params = self._chunk_to_params(chunk)
        query = f"""
            MERGE (chunk:{self._escaped_label} {{id: $id}})
            SET chunk.content = $content,
                chunk.embedding = $embedding,
                chunk.metadata_json = $metadata_json,
                chunk.salience = $salience,
                chunk.cognitive_state_json = $cognitive_state_json,
                chunk.created_at = $created_at,
                chunk.updated_at = $updated_at,
                chunk.access_count = $access_count,
                chunk.version = $version
        """
        self._run(query, **params).consume()

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        if top_k <= 0:
            return []

        all_chunks = self.get_all()
        if filter_fn is not None:
            all_chunks = [chunk for chunk in all_chunks if filter_fn(chunk)]

        scored: list[tuple[float, MemoryChunk]] = []
        for chunk in all_chunks:
            if chunk.embedding is not None and len(chunk.embedding) == len(query_embedding):
                sim = cosine_similarity(query_embedding, chunk.embedding)
            else:
                sim = 0.0
            scored.append((sim, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def update(self, chunk_id: str, chunk: MemoryChunk) -> bool:
        if self.get(chunk_id) is None:
            return False
        if chunk.id != chunk_id:
            chunk = chunk.model_copy(update={"id": chunk_id})
        self.store(chunk)
        return True

    def delete(self, chunk_id: str) -> bool:
        if self.get(chunk_id) is None:
            return False
        query = f"""
            MATCH (chunk:{self._escaped_label})
            WHERE chunk.id = $chunk_id
            DELETE chunk
        """
        self._run(query, chunk_id=chunk_id).consume()
        return True

    def touch(
        self,
        chunk_id: str,
        *,
        access_count: int | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        chunk = self.get(chunk_id)
        if chunk is None:
            return False

        touched_at = updated_at or datetime.now(timezone.utc)
        next_access_count = chunk.access_count + 1 if access_count is None else access_count
        query = f"""
            MATCH (chunk:{self._escaped_label})
            WHERE chunk.id = $chunk_id
            SET chunk.access_count = $access_count,
                chunk.updated_at = $updated_at
        """
        self._run(
            query,
            chunk_id=chunk_id,
            access_count=next_access_count,
            updated_at=touched_at.isoformat(),
        ).consume()
        return True

    def get(self, chunk_id: str) -> MemoryChunk | None:
        query = f"""
            MATCH (chunk:{self._escaped_label})
            WHERE chunk.id = $chunk_id
            RETURN properties(chunk)['id'] AS id,
                   properties(chunk)['content'] AS content,
                   properties(chunk)['embedding'] AS embedding,
                   properties(chunk)['metadata_json'] AS metadata_json,
                   properties(chunk)['salience'] AS salience,
                   properties(chunk)['cognitive_state_json'] AS cognitive_state_json,
                   properties(chunk)['created_at'] AS created_at,
                   properties(chunk)['updated_at'] AS updated_at,
                   properties(chunk)['access_count'] AS access_count,
                   properties(chunk)['version'] AS version
        """
        record = self._run(query, chunk_id=chunk_id).single()
        if record is None:
            return None
        return self._record_to_chunk(record)

    def get_all(self) -> list[MemoryChunk]:
        query = f"""
            MATCH (chunk:{self._escaped_label})
            RETURN properties(chunk)['id'] AS id,
                   properties(chunk)['content'] AS content,
                   properties(chunk)['embedding'] AS embedding,
                   properties(chunk)['metadata_json'] AS metadata_json,
                   properties(chunk)['salience'] AS salience,
                   properties(chunk)['cognitive_state_json'] AS cognitive_state_json,
                   properties(chunk)['created_at'] AS created_at,
                   properties(chunk)['updated_at'] AS updated_at,
                   properties(chunk)['access_count'] AS access_count,
                   properties(chunk)['version'] AS version
        """
        result = self._run(query)
        return [self._record_to_chunk(record) for record in result]

    def get_graph_edges(self, chunk_ids: list[str] | None = None) -> dict[str, dict[str, float]]:
        if chunk_ids:
            query = f"""
                MATCH (source:{self._escaped_label})-[rel]->(target:{self._escaped_label})
                WHERE type(rel) = $edge_type
                  AND source.id IN $chunk_ids
                  AND target.id IN $chunk_ids
                RETURN source.id AS source_id,
                       target.id AS target_id,
                       rel.weight AS weight
            """
            result = self._run(query, chunk_ids=chunk_ids, edge_type=self._graph_edge_type())
        else:
            query = f"""
                MATCH (source:{self._escaped_label})-[rel]->(target:{self._escaped_label})
                WHERE type(rel) = $edge_type
                RETURN source.id AS source_id,
                       target.id AS target_id,
                       rel.weight AS weight
            """
            result = self._run(query, edge_type=self._graph_edge_type())

        edge_map: dict[str, dict[str, float]] = {}
        for record in result:
            source_id = str(record.get("source_id", "")).strip()
            target_id = str(record.get("target_id", "")).strip()
            if not source_id or not target_id or source_id == target_id:
                continue
            edge_map.setdefault(source_id, {})[target_id] = float(record.get("weight", 0.0))
        return edge_map

    def replace_graph_neighbors(self, chunk_id: str, neighbors: dict[str, float]) -> None:
        delete_query = f"""
            MATCH (source:{self._escaped_label})
            WHERE source.id = $chunk_id
            OPTIONAL MATCH (source)-[rel]->()
            WHERE type(rel) = $edge_type
            DELETE rel
        """
        self._run(delete_query, chunk_id=chunk_id, edge_type=self._graph_edge_type()).consume()
        if not neighbors:
            return

        edges = [
            {"target_id": target_id, "weight": float(weight)}
            for target_id, weight in neighbors.items()
            if target_id != chunk_id
        ]
        if not edges:
            return

        query = f"""
            MATCH (source:{self._escaped_label})
            WHERE source.id = $chunk_id
            UNWIND $edges AS edge
            MATCH (target:{self._escaped_label})
            WHERE target.id = edge.target_id
            MERGE (source)-[rel:{self._graph_edge_type()}]->(target)
            SET rel.weight = edge.weight
        """
        self._run(query, chunk_id=chunk_id, edges=edges).consume()

    def upsert_graph_edge(self, source_id: str, target_id: str, weight: float) -> None:
        edge_map = self.get_graph_edges([source_id]).get(source_id, {})
        edge_map[target_id] = float(weight)
        self.replace_graph_neighbors(source_id, edge_map)

        reverse_map = self.get_graph_edges([target_id]).get(target_id, {})
        reverse_map[source_id] = float(weight)
        self.replace_graph_neighbors(target_id, reverse_map)

    def get_stats(self) -> dict[str, Any]:
        query = f"""
            MATCH (chunk:{self._escaped_label})
            RETURN count(chunk) AS total_chunks,
                   count(properties(chunk)['embedding']) AS chunks_with_embeddings,
                   avg(properties(chunk)['salience']) AS average_salience,
                   avg(properties(chunk)['access_count']) AS average_access_count
        """
        record = self._run(query).single() or {}
        edge_query = f"""
            MATCH (:{self._escaped_label})-[rel]->(:{self._escaped_label})
            WHERE type(rel) = $edge_type
            RETURN count(rel) AS directed_related_edges
        """
        edge_record = self._run(edge_query, edge_type=self._graph_edge_type()).single() or {}
        return {
            "backend": "Neo4jStore",
            "name": self.name,
            "uri": self.uri,
            "database": self.database,
            "label": self.label,
            "total_chunks": int(record.get("total_chunks", 0)),
            "chunks_with_embeddings": int(record.get("chunks_with_embeddings", 0)),
            "average_salience": round(float(record.get("average_salience", 0.0) or 0.0), 4),
            "average_access_count": round(float(record.get("average_access_count", 0.0) or 0.0), 4),
            "related_edges": int(edge_record.get("directed_related_edges", 0)) // 2,
        }

    def close(self) -> None:
        with self._lock:
            self._driver.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class QdrantStore(MemoryStore):
    """
    Vector database backend powered by Qdrant.

    Supports both remote Qdrant servers (`url`) and embedded local mode (`path`).
    Collection creation is lazy by default: the first stored chunk defines vector
    size unless `vector_size` is provided at init.

    Requires optional dependency:
        pip install "mnemos-memory[qdrant]"
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
        collection_name: str = "mnemos_memory",
        vector_size: int | None = None,
        name: str = "qdrant",
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.path = path
        self.collection_name = collection_name
        self.name = name
        self._vector_size = vector_size
        self._lock = threading.RLock()
        self._retry_policy = RetryPolicy()
        self._pending_touches: dict[str, tuple[int, datetime]] = {}
        self._known_access_counts: dict[str, int] = {}

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as e:
            raise ImportError(
                "QdrantStore requires the 'qdrant-client' package. "
                "Install with: pip install 'mnemos-memory[qdrant]'"
            ) from e

        self._models = models
        self._client: Any
        if path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(
                url=url or "http://localhost:6333",
                api_key=api_key,
            )

        if self._vector_size is not None:
            self._ensure_collection(self._vector_size)

    def _call_client(self, operation: str, fn: Callable[[], T]) -> T:
        try:
            return call_with_retry(
                provider="qdrant",
                operation=operation,
                fn=fn,
                policy=self._retry_policy,
                should_retry=is_retryable_qdrant_exception,
            )
        except Exception as exc:
            log_event(
                "mnemos.provider_failure",
                level=logging.ERROR,
                provider="qdrant",
                operation=operation,
                error=str(exc),
            )
            raise

    def _collection_exists(self) -> bool:
        with self._lock:
            try:
                return bool(
                    self._call_client(
                        "collection_exists",
                        lambda: self._client.collection_exists(self.collection_name),
                    )
                )
            except Exception:
                try:
                    self._call_client(
                        "get_collection",
                        lambda: self._client.get_collection(self.collection_name),
                    )
                    return True
                except Exception:
                    return False

    def _extract_vector_size(self) -> int | None:
        try:
            info = self._call_client(
                "get_collection",
                lambda: self._client.get_collection(self.collection_name),
            )
        except Exception:
            return None

        # qdrant-client compatibility across versions: vector params can be
        # exposed in multiple shapes (object, dict, named vectors map).
        config = getattr(info, "config", None)
        params = getattr(config, "params", None) if config is not None else None
        vectors = getattr(params, "vectors", None) if params is not None else None

        if vectors is None:
            return None

        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)

        if isinstance(vectors, dict):
            if "size" in vectors and isinstance(vectors["size"], int):
                return int(vectors["size"])
            first = next(iter(vectors.values()), None)
            if first is None:
                return None
            first_size = getattr(first, "size", None)
            if first_size is not None:
                return int(first_size)
            if isinstance(first, dict) and "size" in first and isinstance(first["size"], int):
                return int(first["size"])

        return None

    def _ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("Qdrant vector_size must be positive.")

        with self._lock:
            exists = self._collection_exists()
            if not exists:
                self._call_client(
                    "create_collection",
                    lambda: self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=self._models.VectorParams(
                            size=vector_size,
                            distance=self._models.Distance.COSINE,
                        ),
                    ),
                )
                self._vector_size = vector_size
                return

            existing_size = self._extract_vector_size()
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    "Qdrant collection vector size mismatch: "
                    f"existing={existing_size}, attempted={vector_size}"
                )
            self._vector_size = existing_size or vector_size

    def _chunk_to_payload(self, chunk: MemoryChunk) -> dict[str, Any]:
        return {
            "content": chunk.content,
            "metadata": chunk.metadata,
            "salience": chunk.salience,
            "cognitive_state": (
                chunk.cognitive_state.model_dump() if chunk.cognitive_state is not None else None
            ),
            "created_at": chunk.created_at.isoformat(),
            "updated_at": chunk.updated_at.isoformat(),
            "access_count": chunk.access_count,
            "version": chunk.version,
        }

    def _apply_pending_touch(self, chunk: MemoryChunk) -> MemoryChunk:
        pending = self._pending_touches.get(chunk.id)
        if pending is None:
            self._known_access_counts[chunk.id] = chunk.access_count
            return chunk
        access_count, updated_at = pending
        chunk.access_count = access_count
        chunk.updated_at = updated_at
        self._known_access_counts[chunk.id] = access_count
        return chunk

    def _flush_pending_touches(self) -> None:
        if not self._pending_touches or not self._collection_exists():
            return

        pending = list(self._pending_touches.items())
        set_payload_fn = getattr(self._client, "set_payload", None)
        if callable(set_payload_fn):
            set_payload = cast(Callable[..., Any], set_payload_fn)
            with self._lock:
                for chunk_id, (access_count, updated_at) in pending:

                    def _set_payload(
                        *,
                        chunk_id: str = chunk_id,
                        access_count: int = access_count,
                        updated_at: datetime = updated_at,
                    ) -> Any:
                        return set_payload(
                            collection_name=self.collection_name,
                            payload={
                                "access_count": access_count,
                                "updated_at": updated_at.isoformat(),
                            },
                            points=[chunk_id],
                            wait=False,
                        )

                    self._call_client(
                        "set_payload",
                        _set_payload,
                    )
            for chunk_id, _ in pending:
                self._pending_touches.pop(chunk_id, None)
            return

        pending_map = dict(pending)
        self._pending_touches.clear()
        for chunk_id, (access_count, updated_at) in pending_map.items():
            chunk = self.get(chunk_id)
            if chunk is None:
                continue
            chunk.access_count = access_count
            chunk.updated_at = updated_at
            self.store(chunk)

    def _payload_matches_scope_filter(
        self,
        payload: dict[str, Any],
        *,
        scope_id: str | None,
        allowed_scopes: tuple[str, ...],
    ) -> bool:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        scope_raw = metadata.get("scope")
        scope = str(scope_raw).strip().lower() if isinstance(scope_raw, str) else "global"
        if scope not in {"project", "workspace", "global"}:
            scope = "global"

        if scope not in allowed_scopes:
            return False
        if scope == "global":
            return True
        if scope_id is None:
            return False

        scope_id_raw = metadata.get("scope_id")
        chunk_scope_id = (
            scope_id_raw.strip()
            if isinstance(scope_id_raw, str) and scope_id_raw.strip()
            else "default"
        )
        return chunk_scope_id == scope_id

    def _build_native_scope_filter(
        self,
        filter_fn: Callable[[MemoryChunk], bool] | None,
    ) -> Any | None:
        descriptor = getattr(filter_fn, "_mnemos_scope_filter", None)
        if not isinstance(descriptor, dict):
            return None

        scope_id_raw = descriptor.get("scope_id")
        scope_id = (
            scope_id_raw.strip() if isinstance(scope_id_raw, str) and scope_id_raw.strip() else None
        )
        allowed_scopes_raw = descriptor.get("allowed_scopes")
        if isinstance(allowed_scopes_raw, (list, tuple)):
            allowed_scopes = tuple(
                str(scope).strip().lower() for scope in allowed_scopes_raw if str(scope).strip()
            )
        else:
            allowed_scopes = ("project", "workspace", "global")

        if not allowed_scopes:
            return None

        required_model_names = ("Filter", "FieldCondition", "MatchValue", "MatchAny")
        if not all(hasattr(self._models, name) for name in required_model_names):
            return lambda payload: self._payload_matches_scope_filter(
                payload,
                scope_id=scope_id,
                allowed_scopes=allowed_scopes,
            )

        field_condition = self._models.FieldCondition
        filter_model = self._models.Filter
        match_value = self._models.MatchValue
        match_any = self._models.MatchAny

        should_conditions: list[Any] = []
        if "global" in allowed_scopes:
            should_conditions.append(
                field_condition(key="metadata.scope", match=match_value(value="global"))
            )
            should_conditions.append(
                filter_model(
                    must_not=[
                        field_condition(
                            key="metadata.scope",
                            match=match_any(any=["project", "workspace", "global"]),
                        )
                    ]
                )
            )

        if scope_id is not None:
            for scope in ("project", "workspace"):
                if scope not in allowed_scopes:
                    continue
                should_conditions.append(
                    filter_model(
                        must=[
                            field_condition(key="metadata.scope", match=match_value(value=scope)),
                            field_condition(
                                key="metadata.scope_id",
                                match=match_value(value=scope_id),
                            ),
                        ]
                    )
                )

        if not should_conditions:
            return None
        return filter_model(should=should_conditions)

    def _parse_vector(self, point: Any) -> list[float] | None:
        vector_obj = getattr(point, "vector", None)
        if vector_obj is None:
            return None
        if isinstance(vector_obj, list):
            return [float(x) for x in vector_obj]
        if isinstance(vector_obj, dict):
            first = next(iter(vector_obj.values()), None)
            if isinstance(first, list):
                return [float(x) for x in first]
        return None

    def _parse_payload(self, point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        if isinstance(payload, dict):
            return payload
        return {}

    def _point_to_chunk(self, point: Any) -> MemoryChunk:
        payload = self._parse_payload(point)

        created_at_raw = payload.get("created_at")
        updated_at_raw = payload.get("updated_at")
        created_at = (
            datetime.fromisoformat(created_at_raw) if isinstance(created_at_raw, str) else None
        )
        updated_at = (
            datetime.fromisoformat(updated_at_raw) if isinstance(updated_at_raw, str) else None
        )

        if created_at is None:
            created_at = datetime.now(timezone.utc)
        elif created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        if updated_at is None:
            updated_at = datetime.now(timezone.utc)
        elif updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        cognitive_state_payload = payload.get("cognitive_state")
        cognitive_state = (
            CognitiveState.model_validate(cognitive_state_payload)
            if isinstance(cognitive_state_payload, dict)
            else None
        )

        point_id = getattr(point, "id", "")
        return MemoryChunk(
            id=str(point_id),
            content=str(payload.get("content", "")),
            embedding=self._parse_vector(point),
            metadata=(payload["metadata"] if isinstance(payload.get("metadata"), dict) else {}),
            salience=float(payload.get("salience", 0.5)),
            cognitive_state=cognitive_state,
            created_at=created_at,
            updated_at=updated_at,
            access_count=int(payload.get("access_count", 0)),
            version=int(payload.get("version", 1)),
        )

    def store(self, chunk: MemoryChunk) -> None:
        """Insert or replace a chunk by ID."""
        self._pending_touches.pop(chunk.id, None)
        self._known_access_counts[chunk.id] = chunk.access_count
        vector = chunk.embedding
        if vector is None:
            if self._vector_size is None:
                raise ValueError(
                    "Cannot store chunk without embedding before Qdrant vector size is known."
                )
            vector = [0.0] * self._vector_size

        self._ensure_collection(len(vector))

        point = self._models.PointStruct(
            id=chunk.id,
            vector=vector,
            payload=self._chunk_to_payload(chunk),
        )

        with self._lock:
            self._call_client(
                "upsert",
                lambda: self._client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                    wait=True,
                ),
            )

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """
        Retrieve top-k chunks by cosine similarity.

        For generic Python callbacks we preserve exact MemoryStore semantics via
        a full scan. Engine-generated scope filters are translated to native
        Qdrant payload filters to avoid collection scans on normal retrieval.
        """
        if top_k <= 0:
            return []
        if not self._collection_exists():
            return []

        native_filter = self._build_native_scope_filter(filter_fn)
        if filter_fn is not None and native_filter is None:
            candidates = [chunk for chunk in self.get_all() if filter_fn(chunk)]
            scored: list[tuple[float, MemoryChunk]] = []
            for chunk in candidates:
                if chunk.embedding is not None and len(chunk.embedding) == len(query_embedding):
                    sim = cosine_similarity(query_embedding, chunk.embedding)
                else:
                    sim = 0.0
                scored.append((sim, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in scored[:top_k]]

        with self._lock:
            search_fn = getattr(self._client, "search", None)
            query_points_fn = getattr(self._client, "query_points", None)
            if callable(search_fn):
                search_kwargs = {
                    "collection_name": self.collection_name,
                    "query_vector": query_embedding,
                    "limit": top_k,
                    "with_payload": True,
                    "with_vectors": True,
                }
                if native_filter is not None:
                    search_kwargs["query_filter"] = native_filter
                hits = self._call_client("search", lambda: search_fn(**search_kwargs))
            elif callable(query_points_fn):
                query_kwargs = {
                    "collection_name": self.collection_name,
                    "query": query_embedding,
                    "limit": top_k,
                    "with_payload": True,
                    "with_vectors": True,
                }
                if native_filter is not None:
                    query_kwargs["query_filter"] = native_filter
                query_result = self._call_client(
                    "query_points",
                    lambda: query_points_fn(**query_kwargs),
                )
                points = getattr(query_result, "points", query_result)
                if isinstance(points, list):
                    hits = points
                else:
                    raise TypeError("Unsupported Qdrant query_points response shape.")
            else:
                raise RuntimeError("Qdrant client does not support search/query_points APIs.")

        chunks = [self._apply_pending_touch(self._point_to_chunk(hit)) for hit in hits]
        if filter_fn is not None:
            chunks = [chunk for chunk in chunks if filter_fn(chunk)]
        return chunks

    def update(self, chunk_id: str, chunk: MemoryChunk) -> bool:
        """Update an existing chunk; returns False if not found."""
        existing = self.get(chunk_id)
        if existing is None:
            return False

        if chunk.id != chunk_id:
            chunk = chunk.model_copy(update={"id": chunk_id})
        self.store(chunk)
        return True

    def touch(
        self,
        chunk_id: str,
        *,
        access_count: int | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Record access metadata and defer payload flush off the hot retrieval path."""
        touched_at = updated_at or datetime.now(timezone.utc)
        with self._lock:
            pending = self._pending_touches.get(chunk_id)
            base_access_count: int | None
            if pending is not None:
                base_access_count = pending[0]
            else:
                base_access_count = self._known_access_counts.get(chunk_id)

        if base_access_count is None:
            if not self._collection_exists():
                return False
            existing = self.get(chunk_id)
            if existing is None:
                return False
            base_access_count = existing.access_count

        next_access_count = base_access_count + 1 if access_count is None else access_count
        should_flush = False
        with self._lock:
            self._pending_touches[chunk_id] = (next_access_count, touched_at)
            self._known_access_counts[chunk_id] = next_access_count
            should_flush = len(self._pending_touches) >= 128
        if should_flush:
            self._flush_pending_touches()
        return True

    def delete(self, chunk_id: str) -> bool:
        """Delete chunk by ID; returns False if not found."""
        if self.get(chunk_id) is None:
            return False

        self._pending_touches.pop(chunk_id, None)
        self._pending_touches.pop(chunk_id, None)
        self._known_access_counts.pop(chunk_id, None)
        selector = self._models.PointIdsList(points=[chunk_id])
        with self._lock:
            self._call_client(
                "delete",
                lambda: self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=selector,
                    wait=True,
                ),
            )
        return True

    def get(self, chunk_id: str) -> MemoryChunk | None:
        """Lookup by point ID."""
        if not self._collection_exists():
            return None

        with self._lock:
            points = self._call_client(
                "retrieve",
                lambda: self._client.retrieve(
                    collection_name=self.collection_name,
                    ids=[chunk_id],
                    with_payload=True,
                    with_vectors=True,
                ),
            )
        if not points:
            return None
        return self._apply_pending_touch(self._point_to_chunk(points[0]))

    def get_all(self) -> list[MemoryChunk]:
        """Load all chunks from the collection using scroll pagination."""
        if not self._collection_exists():
            return []

        all_chunks: list[MemoryChunk] = []
        offset: Any | None = None

        while True:
            with self._lock:
                points, next_offset = self._call_client(
                    "scroll",
                    lambda: self._client.scroll(
                        collection_name=self.collection_name,
                        limit=256,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True,
                    ),
                )
            all_chunks.extend(
                self._apply_pending_touch(self._point_to_chunk(point)) for point in points
            )
            if next_offset is None:
                break
            offset = next_offset

        return all_chunks

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the Qdrant-backed store."""
        chunks = self.get_all()
        total = len(chunks)
        with_embedding = sum(1 for c in chunks if c.embedding is not None)
        avg_salience = sum(c.salience for c in chunks) / total if total > 0 else 0.0
        avg_access = sum(c.access_count for c in chunks) / total if total > 0 else 0.0

        if self._vector_size is None and self._collection_exists():
            self._vector_size = self._extract_vector_size()

        return {
            "backend": "QdrantStore",
            "name": self.name,
            "collection_name": self.collection_name,
            "qdrant_url": self.url,
            "qdrant_path": self.path,
            "vector_size": self._vector_size,
            "total_chunks": total,
            "chunks_with_embeddings": with_embedding,
            "average_salience": round(avg_salience, 4),
            "average_access_count": round(avg_access, 4),
        }

    def close(self) -> None:
        """Close the underlying client if it exposes close()."""
        self._flush_pending_touches()
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self) -> None:
        """Ensure client cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
