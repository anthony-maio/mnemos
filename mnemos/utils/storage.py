"""
mnemos/utils/storage.py — Storage backends for the Mnemos memory system.

Storage backends handle persistence of MemoryChunks. The interface is
intentionally simple and synchronous — async wrappers are used at the
module level where needed.

Backends:
- MemoryStore (abstract): defines the interface
- InMemoryStore: dict-based, cosine similarity search (development default)
- SQLiteStore: persistent storage using Python's built-in sqlite3
- QdrantStore: scalable vector retrieval with Qdrant

The InMemoryStore is analogous to the hippocampus — fast, temporary,
in-process working memory. The SQLiteStore/QdrantStore are analogous to
neocortical long-term storage — persistent and restart-safe.

Both implement the same interface, so you can swap them transparently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

from ..types import CognitiveState, MemoryChunk
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

        When `filter_fn` is supplied, we do a full scan and local scoring to
        preserve exact callback semantics defined by MemoryStore.
        """
        if top_k <= 0:
            return []
        if not self._collection_exists():
            return []

        if filter_fn is not None:
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
                hits = self._call_client(
                    "search",
                    lambda: search_fn(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=True,
                    ),
                )
            elif callable(query_points_fn):
                query_result = self._call_client(
                    "query_points",
                    lambda: query_points_fn(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=True,
                    ),
                )
                points = getattr(query_result, "points", query_result)
                if isinstance(points, list):
                    hits = points
                else:
                    raise TypeError("Unsupported Qdrant query_points response shape.")
            else:
                raise RuntimeError("Qdrant client does not support search/query_points APIs.")

        return [self._point_to_chunk(hit) for hit in hits]

    def update(self, chunk_id: str, chunk: MemoryChunk) -> bool:
        """Update an existing chunk; returns False if not found."""
        existing = self.get(chunk_id)
        if existing is None:
            return False

        if chunk.id != chunk_id:
            chunk = chunk.model_copy(update={"id": chunk_id})
        self.store(chunk)
        return True

    def delete(self, chunk_id: str) -> bool:
        """Delete chunk by ID; returns False if not found."""
        if self.get(chunk_id) is None:
            return False

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
        return self._point_to_chunk(points[0])

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
            all_chunks.extend(self._point_to_chunk(point) for point in points)
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
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self) -> None:
        """Ensure client cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
