"""Scope-aware canonical and episodic storage regressions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.storage import (
    InMemoryStore,
    SQLiteStore,
    compute_canonical_key,
)


@pytest.mark.parametrize("native", [True, False])
def test_scoped_sql_cosine_precedes_limit_without_loading_foreign_rows(tmp_path, native):
    store = SQLiteStore(str(tmp_path / "scoped-search.sqlite"))
    try:

        def allowed(chunk):
            return chunk.metadata.get("user_profile_id") == "profile-a"

        allowed._mnemos_scope_filter = {
            "user_profile_id": "profile-a",
            "project_id": "repo",
            "allowed_scopes": ["project"],
        }
        for chunk_id, profile, vector in [
            ("foreign", "profile-b", [1.0, 0.0]),
            ("cosine-best", "profile-a", [100.0, 0.0]),
            ("l2-best", "profile-a", [0.9, 0.1]),
        ]:
            chunk = _chunk("SQLite " + chunk_id, profile, chunk_id)
            chunk.embedding = vector
            chunk.canonical_key = None
            store.store(chunk)
        store._sqlite_vec_enabled = native
        original_decode = store._row_to_chunk

        def decode_authorized(row):
            chunk = original_decode(row)
            assert chunk.metadata["user_profile_id"] == "profile-a"
            return chunk

        store._row_to_chunk = decode_authorized
        assert [c.id for c in store.retrieve([1.0, 0.0], 1, allowed)] == ["cosine-best"]
        lexical = store.retrieve_lexical('SQLite OR " DROP TABLE', 1, allowed)
        assert len(lexical) == 1
        assert lexical[0].metadata["user_profile_id"] == "profile-a"
        assert "scoped_exact_cosine" in store.last_retrieval_backend
    finally:
        store.close()


def test_migration_preserves_fact_and_rejects_future_database(tmp_path):
    path = str(tmp_path / "migration.sqlite")
    store = SQLiteStore(path)
    store.store(_chunk("Keep this fact", "profile-a", "stable"))
    store._conn.execute("DROP INDEX idx_memory_chunks_scope")
    store._conn.execute("UPDATE mnemos_meta SET value='4' WHERE key='schema_version'")
    store._conn.commit()
    store.close()
    store = SQLiteStore(path)
    assert store.get("stable").content == "Keep this fact"
    assert store.get_stats()["schema_version"] == int(SQLiteStore._SCHEMA_VERSION)
    import sqlite3

    assert store.migration_backup is not None
    with sqlite3.connect(store.migration_backup) as backup:
        assert backup.execute(
            "SELECT value FROM mnemos_meta WHERE key='schema_version'"
        ).fetchone() == ("4",)
        assert backup.execute("SELECT content FROM memory_chunks WHERE id='stable'").fetchone() == (
            "Keep this fact",
        )
    store._conn.execute("UPDATE mnemos_meta SET value='999' WHERE key='schema_version'")
    store._conn.commit()
    store.close()
    with pytest.raises(ValueError, match="newer"):
        SQLiteStore(path)


def test_exact_binary_vectors_survive_fallback_updates_and_cache_rebuild(tmp_path):
    path = str(tmp_path / "binary-vectors.sqlite")
    store = SQLiteStore(path)
    try:
        for identity, vector in [("a", [10.0, 0.0]), ("b", [0.0, 1.0])]:
            store.store(MemoryChunk(id=identity, content=identity, embedding=vector))
        assert store.retrieve([1.0, 0.0], 1)[0].id == "a"
        # A writer without the optional extension still maintains portable
        # vectors. Reopening with the extension must not expose the old value.
        store._sqlite_vec_enabled = False
        store.update("a", MemoryChunk(id="a", content="changed", embedding=[-1.0, 0.0]))
        store.close()
        store = SQLiteStore(path)
        assert store.retrieve([1.0, 0.0], 1)[0].id == "b"
        # Older stores have no binary cache. Reconstruct it from authoritative
        # vectors without changing fact contents or cosine ordering.
        store._conn.execute("DROP TRIGGER memory_vector_values_delete")
        store._conn.execute("DROP TABLE memory_vector_values")
        store._conn.commit()
        store.close()
        store = SQLiteStore(path)
        assert store.retrieve([1.0, 0.0], 1)[0].id == "b"
        assert store.get("a").content == "changed"
        assert store.delete("b") is True
        assert store.retrieve([1.0, 0.0], 1)[0].id == "a"
        assert store._conn.execute("SELECT COUNT(*) FROM memory_vector_values").fetchone() == (1,)
    finally:
        store.close()


def test_sqlite_legacy_review_filters_declared_profiles_before_limit(tmp_path):
    store = SQLiteStore(str(tmp_path / "legacy-review.sqlite"))
    try:
        for chunk_id, metadata in (
            ("a-profile", {"user_profile_id": "alice"}),
            ("b-legacy", {}),
            ("c-profile-key", {"profile_id": "alice"}),
            ("d-legacy", {}),
        ):
            store.store(MemoryChunk(id=chunk_id, content=chunk_id, metadata=metadata))

        first = store.list_legacy_chunks(limit=2)
        assert [chunk.id for chunk in first] == ["b-legacy", "d-legacy"]
        assert store.list_legacy_chunks(limit=1, after_id="b-legacy")[0].id == "d-legacy"
        assert store.list_legacy_chunks(limit=10) == first
    finally:
        store.close()


def test_batch_retrieval_metadata_preserves_facts_and_search_index(tmp_path):
    store = SQLiteStore(str(tmp_path / "batch-access.sqlite"))
    try:
        for identity in ("one", "two"):
            store.store(
                MemoryChunk(id=identity, content=f"sqlite {identity}", embedding=[1.0, 0.0])
            )
        statements = []
        store._conn.set_trace_callback(statements.append)
        assert store.mark_retrieved_many(["one", "two", "one", "missing"]) == 2
        store._conn.set_trace_callback(None)
        assert sum(sql.strip().upper() == "COMMIT" for sql in statements) == 1
        for identity in ("one", "two"):
            fact = store.get(identity)
            assert fact.access_count == 1
            assert fact.version == 1
            assert fact.last_retrieved_at is not None
        assert {fact.id for fact in store.retrieve_lexical("sqlite")} == {"one", "two"}
        changed = store.get("one").model_copy(update={"content": "postgres one"})
        store.update("one", changed)
        assert {fact.id for fact in store.retrieve_lexical("sqlite")} == {"two"}
        assert {fact.id for fact in store.retrieve_lexical("postgres")} == {"one"}
    finally:
        store.close()


def _chunk(content: str, profile: str, chunk_id: str) -> MemoryChunk:
    provider = SimpleEmbeddingProvider(dim=16)
    return MemoryChunk(
        id=chunk_id,
        content=content,
        embedding=provider.embed(content),
        canonical_key="stale-caller-key",
        metadata={
            "scope": "project",
            "scope_id": "repo",
            "user_profile_id": profile,
        },
    )


@pytest.mark.parametrize("store_factory", [InMemoryStore, SQLiteStore])
def test_canonical_key_is_case_sensitive_and_profile_partitioned(
    store_factory, tmp_path: Path
) -> None:
    store = (
        store_factory()
        if store_factory is InMemoryStore
        else store_factory(db_path=str(tmp_path / "scoped.sqlite"))
    )
    try:
        first = store.upsert_canonical(_chunk("Use Python", "profile-a", "a"))
        second = store.upsert_canonical(_chunk("use python", "profile-a", "b"))
        third = store.upsert_canonical(_chunk("Use Python", "profile-b", "c"))
        assert first.canonical_key == compute_canonical_key(
            "Use Python", "project", "repo", "profile-a"
        )
        assert second.canonical_key != first.canonical_key
        assert third.canonical_key != first.canonical_key
        assert len(store.get_all()) == 3
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


def _event(
    event_id: str,
    profile: str,
    session: str,
    occurred_at: datetime,
    *,
    source_event_id: str | None = None,
    host: str | None = None,
) -> EpisodicEvent:
    metadata = {
        "user_profile_id": profile,
        "session_id": session,
        "event_id": source_event_id or event_id,
    }
    if host is not None:
        metadata["host"] = host
    return EpisodicEvent(
        event_fingerprint=event_id,
        role="user",
        content=event_id,
        normalized_content=event_id,
        scope="project",
        scope_id="repo",
        source="manual",
        metadata=metadata,
        occurred_at=occurred_at,
    )


@pytest.mark.parametrize("store_factory", [InMemoryStore, SQLiteStore])
def test_episodic_filters_apply_before_limit_and_support_newest_first(
    store_factory, tmp_path: Path
) -> None:
    store = (
        store_factory()
        if store_factory is InMemoryStore
        else store_factory(db_path=str(tmp_path / "events.sqlite"))
    )
    try:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(3):
            store.append_episodic_event(
                _event(f"a-{index}", "profile-a", "session-a", base + timedelta(minutes=index))
            )
        store.append_episodic_event(
            _event("b-0", "profile-b", "session-b", base + timedelta(minutes=3))
        )
        events = store.list_episodic_events(
            user_profile_id="profile-a",
            session_id="session-a",
            before=base + timedelta(minutes=3),
            newest_first=True,
            limit=2,
        )
        assert [event.content for event in events] == ["a-2", "a-1"]
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


@pytest.mark.parametrize("store_factory", [InMemoryStore, SQLiteStore])
def test_source_event_ids_are_namespaced_by_profile_host_and_session(
    store_factory, tmp_path: Path
) -> None:
    store = (
        store_factory()
        if store_factory is InMemoryStore
        else store_factory(db_path=str(tmp_path / "source-identity.sqlite"))
    )
    try:
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.append_episodic_event(
            _event(
                "finger-a", "profile-a", "session", timestamp, source_event_id="same", host="codex"
            )
        )
        store.append_episodic_event(
            _event(
                "finger-b", "profile-b", "session", timestamp, source_event_id="same", host="codex"
            )
        )
        store.append_episodic_event(
            _event(
                "finger-c", "profile-a", "session", timestamp, source_event_id="same", host="hermes"
            )
        )
        assert len(store.list_episodic_events()) == 3
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


@pytest.mark.parametrize("store_factory", [InMemoryStore, SQLiteStore])
def test_source_event_identity_rejects_changed_payload(store_factory, tmp_path: Path) -> None:
    store = (
        store_factory()
        if store_factory is InMemoryStore
        else store_factory(db_path=str(tmp_path / "source-payload.sqlite"))
    )
    try:
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        original = _event(
            "finger-original",
            "profile-a",
            "session",
            timestamp,
            source_event_id="same",
            host="codex",
        )
        changed = original.model_copy(
            update={"event_fingerprint": "finger-changed", "content": "tampered"}
        )
        store.append_episodic_event(original)
        with pytest.raises(ValueError, match="different payload"):
            store.append_episodic_event(changed)
    finally:
        if isinstance(store, SQLiteStore):
            store.close()
