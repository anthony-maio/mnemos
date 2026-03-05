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
long-term storage — slower, persistent, survives process restarts.

Both implement the same interface, so you can swap them transparently.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..types import CognitiveState, MemoryChunk
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

    def clear(self) -> None:
        """Remove all chunks from the store. Used for testing and consolidation."""
        for chunk in list(self.get_all()):
            self.delete(chunk.id)


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

    def __init__(self, db_path: str = "mnemos_memory.db", name: str = "sqlite") -> None:
        self.db_path = db_path
        self.name = name
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrent access
        self._conn.execute(self._CREATE_TABLE_SQL)
        self._conn.commit()

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

    def store(self, chunk: MemoryChunk) -> None:
        """Insert or replace a chunk by ID."""
        row = self._chunk_to_row(chunk)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_chunks
                (id, content, embedding, metadata, salience, cognitive_state,
                 created_at, updated_at, access_count, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            self._conn.commit()

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: Callable[[MemoryChunk], bool] | None = None,
    ) -> list[MemoryChunk]:
        """Retrieve top-k chunks by cosine similarity (loads all, scores in Python)."""
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
            row = self._chunk_to_row(chunk)
            self._conn.execute(
                """
                UPDATE memory_chunks SET
                    content=?, embedding=?, metadata=?, salience=?,
                    cognitive_state=?, created_at=?, updated_at=?,
                    access_count=?, version=?
                WHERE id=?
                """,
                row[1:] + (chunk_id,),
            )
            self._conn.commit()
            return True

    def delete(self, chunk_id: str) -> bool:
        """Delete chunk by ID; returns False if not found."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memory_chunks WHERE id = ?", (chunk_id,))
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

        return {
            "backend": "SQLiteStore",
            "name": self.name,
            "db_path": self.db_path,
            "total_chunks": total,
            "chunks_with_embeddings": with_embedding,
            "average_salience": round(avg_salience, 4),
            "average_access_count": round(avg_access, 4),
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
