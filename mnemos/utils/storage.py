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

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast

from ..types import CognitiveState, MemoryChunk, RetrievalFeedbackEvent
from .embeddings import cosine_similarity


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
