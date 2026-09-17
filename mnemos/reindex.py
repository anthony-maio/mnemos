"""Resumable embedding generations with an atomic activation boundary."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .utils.embeddings import EmbeddingProvider, embed_text_async
from .utils.storage import SQLiteStore


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare(store: SQLiteStore) -> None:
    with store._lock:
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_generations ("
            "id TEXT PRIMARY KEY, identity TEXT NOT NULL, status TEXT NOT NULL, dimension INTEGER)"
        )
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_generation_rows ("
            "generation_id TEXT NOT NULL, fact_id TEXT NOT NULL, version INTEGER NOT NULL, "
            "content_hash TEXT NOT NULL, embedding TEXT NOT NULL, PRIMARY KEY(generation_id,fact_id), "
            "FOREIGN KEY(fact_id) REFERENCES memory_chunks(id) ON DELETE CASCADE)"
        )
        store._conn.commit()


async def build_generation(
    store: SQLiteStore,
    embedder: EmbeddingProvider,
    identity: dict[str, Any],
    *,
    batch_size: int = 100,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if not 1 <= batch_size <= 1000:
        raise ValueError("Reindex batch size must be between 1 and 1000")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    _prepare(store)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    generation = _digest(encoded)
    with store._lock:
        store._conn.execute(
            "INSERT OR IGNORE INTO embedding_generations VALUES (?, ?, 'building', NULL)",
            (generation, encoded),
        )
        dimension = store._conn.execute(
            "SELECT dimension FROM embedding_generations WHERE id=?", (generation,)
        ).fetchone()[0]
        store._conn.commit()
    cursor, batches, embedded = "", 0, 0
    while max_batches is None or batches < max_batches:
        chunks = store.list_chunks(limit=batch_size, after_id=cursor)
        if not chunks:
            break
        embedded_before = embedded
        for chunk in chunks:
            fingerprint = _digest(chunk.content)
            with store._lock:
                existing = store._conn.execute(
                    "SELECT version, content_hash FROM embedding_generation_rows "
                    "WHERE generation_id=? AND fact_id=?",
                    (generation, chunk.id),
                ).fetchone()
            if existing == (chunk.version, fingerprint):
                continue
            vector = await embed_text_async(embedder, chunk.content)
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding provider returned an invalid vector")
            if dimension is not None and len(vector) != dimension:
                raise ValueError("Embedding dimension changed during reindex")
            dimension = len(vector)
            with store._lock:
                store._conn.execute(
                    "INSERT OR REPLACE INTO embedding_generation_rows VALUES (?, ?, ?, ?, ?)",
                    (generation, chunk.id, chunk.version, fingerprint, json.dumps(vector)),
                )
                store._conn.execute(
                    "UPDATE embedding_generations SET dimension=? WHERE id=?",
                    (dimension, generation),
                )
                store._conn.commit()
            embedded += 1
        cursor = chunks[-1].id
        if embedded != embedded_before:
            batches += 1
    with store._lock:
        store._conn.create_function("mnemos_sha256", 1, _digest, deterministic=True)
        missing = store._conn.execute(
            "SELECT COUNT(*) FROM memory_chunks AS fact WHERE NOT EXISTS "
            "(SELECT 1 FROM embedding_generation_rows AS staged WHERE staged.generation_id=? "
            "AND staged.fact_id=fact.id AND staged.version=fact.version "
            "AND staged.content_hash=mnemos_sha256(fact.content))",
            (generation,),
        ).fetchone()[0]
    return {
        "generation_id": generation,
        "embedded_this_run": embedded,
        "missing": missing,
        "ready": missing == 0,
        "activated": False,
        "dimension": dimension,
    }


def activate_generation(store: SQLiteStore, generation_id: str) -> dict[str, Any]:
    """Caller must replace its embedding provider after activation, before accepting requests."""
    _prepare(store)
    prior_dimension = store._sqlite_vec_dim
    with store._lock:
        store._conn.create_function("mnemos_sha256", 1, _digest, deterministic=True)
        store._conn.execute("BEGIN IMMEDIATE")
        try:
            generation = store._conn.execute(
                "SELECT identity, dimension FROM embedding_generations WHERE id=?", (generation_id,)
            ).fetchone()
            if generation is None:
                raise KeyError("Unknown embedding generation")
            missing = store._conn.execute(
                "SELECT COUNT(*) FROM memory_chunks AS fact WHERE NOT EXISTS "
                "(SELECT 1 FROM embedding_generation_rows AS staged WHERE staged.generation_id=? "
                "AND staged.fact_id=fact.id AND staged.version=fact.version "
                "AND staged.content_hash=mnemos_sha256(fact.content))",
                (generation_id,),
            ).fetchone()[0]
            if missing:
                raise ValueError(
                    "Embedding generation is incomplete or facts changed; resume reindex"
                )
            store._conn.execute(
                "UPDATE memory_chunks SET embedding=(SELECT embedding FROM embedding_generation_rows "
                "WHERE generation_id=? AND fact_id=memory_chunks.id)",
                (generation_id,),
            )
            if store._vec_table_exists():
                store._conn.execute("DROP TABLE memory_vec")
            store._sqlite_vec_dim = None
            if generation[1] is not None:
                store._ensure_vec_table(int(generation[1]))
                store._rebuild_vec_index()
            store._conn.execute("DELETE FROM memory_edges")
            store._conn.execute(
                "INSERT INTO mnemos_meta(key,value) VALUES ('embedding_identity',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (generation[0],),
            )
            store._conn.execute(
                "UPDATE embedding_generations SET status='retired' WHERE status='active'"
            )
            store._conn.execute(
                "UPDATE embedding_generations SET status='active' WHERE id=?", (generation_id,)
            )
            store._conn.commit()
        except Exception:
            store._conn.rollback()
            store._sqlite_vec_dim = prior_dimension
            raise
    return {
        "generation_id": generation_id,
        "activated": True,
        "identity": json.loads(generation[0]),
    }


async def reindex_configured_store(
    *, activate: bool = False, batch_size: int = 100, max_batches: int | None = None
) -> dict[str, Any]:
    """Offline maintenance using the service's settings and exclusive ownership lock.

    Select the desired provider in configuration before running this command.
    The service must be stopped; failed or partial builds retain active vectors.
    """
    from pathlib import Path
    from .runtime import (
        load_runtime_settings,
        build_embedder_from_settings,
        build_store_from_settings,
        embedding_identity,
    )
    from .service import default_descriptor_path, default_runtime_directory, _ProcessLock
    from .backup import create_backup
    from .scopes import ScopeRegistry, default_registry_path

    lock = _ProcessLock(default_descriptor_path().with_suffix(".lock"))
    if not lock.acquire():
        raise ValueError("Stop the Mnemos service before reindexing; its ownership lock is held")
    store = None
    try:
        settings = load_runtime_settings(
            default_store_type="sqlite", cwd=default_runtime_directory()
        ).settings
        if settings.storage.type != "sqlite":
            raise ValueError("Reindex requires persistent SQLite storage")
        if not Path(settings.storage.sqlite_path).is_file():
            raise FileNotFoundError("No existing memory database to reindex")
        store = build_store_from_settings(settings)
        assert isinstance(store, SQLiteStore)
        identity = embedding_identity(settings)
        embedder = build_embedder_from_settings(settings)
        result = await build_generation(
            store, embedder, identity, batch_size=batch_size, max_batches=max_batches
        )
        if activate:
            if not result["ready"]:
                raise ValueError("Reindex is incomplete; resume before activation")
            registry_path = default_registry_path()
            registry = (
                ScopeRegistry.load(registry_path) if registry_path.exists() else ScopeRegistry()
            )
            from uuid import uuid4

            archive = Path(settings.storage.sqlite_path).with_name(f"pre-reindex-{uuid4().hex}.zip")
            create_backup(store, archive, registry)
            result.update(activate_generation(store, result["generation_id"]))
            result["backup"] = str(archive.resolve())
        return {
            "schema_version": 1,
            **result,
            "identity": identity,
            "database": str(Path(settings.storage.sqlite_path).resolve()),
        }
    finally:
        if isinstance(store, SQLiteStore):
            store.close()
        lock.release()
