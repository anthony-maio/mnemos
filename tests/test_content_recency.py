"""Content chronology must survive access, persistence and legacy migration."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from mnemos.config import MnemosConfig, MutableRAGConfig
from mnemos.engine import MnemosEngine
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import MockLLMProvider
from mnemos.utils.storage import InMemoryStore, SQLiteStore

OLD = datetime(2025, 1, 1, tzinfo=timezone.utc)
NEW = OLD + timedelta(days=1)
REVISION = NEW + timedelta(days=1)
ACCESS = REVISION + timedelta(days=1)
SCOPE = {"scope": "project", "scope_id": "recency-contract"}


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SQLiteStore(str(tmp_path / "contract.sqlite"))
    )
    yield value
    if isinstance(value, SQLiteStore):
        value.close()


def chunk(identifier, created, *, revised=None, embedding=None):
    return MemoryChunk.model_validate(
        {
            "id": identifier,
            "content": "deployment setting " + identifier,
            "created_at": created,
            "updated_at": created,
            "content_updated_at": revised,
            "embedding": embedding,
            "metadata": dict(SCOPE),
        }
    )


def make_engine(store):
    return MnemosEngine(
        config=MnemosConfig(mutable_rag=MutableRAGConfig(enabled=False)),
        store=store,
        embedder=SimpleEmbeddingProvider(dim=64),
        llm=MockLLMProvider(),
    )


async def current_ids(engine, top_k):
    results = await engine.retrieve(
        "Which deployment setting is current?",
        top_k=top_k,
        reconsolidate=False,
        current_scope="project",
        scope_id=SCOPE["scope_id"],
        allowed_scopes=("project",),
    )
    return [result.id for result in results]


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("explicit_revision", [False, True])
async def test_access_does_not_promote_stale_content(store, top_k, explicit_revision):
    engine = make_engine(store)
    vector = engine.embedder.embed("Which deployment setting is current?")
    store.store(chunk("old", OLD, revised=OLD if explicit_revision else None, embedding=vector))
    store.store(chunk("new", NEW, revised=NEW if explicit_revision else None, embedding=vector))
    foreign = chunk("foreign", ACCESS, revised=ACCESS, embedding=vector)
    foreign.metadata["scope_id"] = "another-project"
    store.store(foreign)
    before = await current_ids(engine, top_k)
    assert before == ["new", "old"][:top_k]
    assert store.touch("old", updated_at=datetime.now(timezone.utc) + timedelta(days=1))
    after = await current_ids(engine, top_k)
    assert after == before
    assert "foreign" not in after


@pytest.mark.asyncio
async def test_explicit_revision_takes_precedence_over_creation(store):
    engine = make_engine(store)
    vector = engine.embedder.embed("Which deployment setting is current?")
    store.store(chunk("rewritten", OLD, revised=REVISION, embedding=vector))
    store.store(chunk("newer-creation", NEW, revised=NEW, embedding=vector))
    assert await current_ids(engine, 1) == ["rewritten"]


@pytest.mark.asyncio
async def test_access_does_not_break_a_content_time_tie(store):
    engine = make_engine(store)
    vector = engine.embedder.embed("Which deployment setting is current?")
    store.store(chunk("first", OLD, revised=NEW, embedding=vector))
    store.store(chunk("second", OLD, revised=NEW, embedding=vector))
    before = await current_ids(engine, 1)
    other = ({"first", "second"} - set(before)).pop()
    assert store.touch(other, updated_at=datetime.now(timezone.utc) + timedelta(days=1))
    assert await current_ids(engine, 1) == before


def test_touch_preserves_known_revision_and_serialization(store):
    original = chunk("known", OLD, revised=REVISION)
    store.store(original)
    assert store.touch(original.id, updated_at=ACCESS)
    result = store.get(original.id)
    assert result.content_updated_at == REVISION
    assert result.content_recency_at == REVISION
    assert result.updated_at == ACCESS
    assert result.access_count == 1
    assert result.content == original.content
    roundtrip = MemoryChunk.model_validate_json(result.model_dump_json())
    assert roundtrip.content_updated_at == REVISION


def test_unknown_revision_history_is_not_inferred_from_access(store):
    original = chunk("unknown", OLD)
    original.version = 4
    original.updated_at = ACCESS
    store.store(original)
    assert store.touch(original.id, updated_at=ACCESS + timedelta(days=1))
    result = store.get(original.id)
    assert result.content_updated_at is None
    assert result.content_recency_at == OLD
    assert result.model_dump(mode="json")["content_updated_at"] is None
    assert result.version == 4


def test_reconsolidation_records_one_content_revision_time(monkeypatch, store):
    original = chunk("same-id", OLD)
    monkeypatch.setattr("mnemos.types._utcnow", lambda: REVISION)
    updated = original.reconsolidate("replacement deployment setting")
    assert updated.id == original.id and updated.version == original.version + 1
    assert updated.created_at == OLD
    assert updated.content_updated_at == updated.updated_at == REVISION
    store.store(updated)
    assert store.touch(updated.id, updated_at=ACCESS)
    assert store.get(updated.id).content_updated_at == REVISION
    assert original.content_updated_at is None


@pytest.mark.parametrize("revised", [None, REVISION])
def test_model_touch_preserves_content_recency(monkeypatch, revised):
    original = chunk("model-touch", OLD, revised=revised)
    monkeypatch.setattr("mnemos.types._utcnow", lambda: ACCESS)
    original.touch()
    assert original.content_updated_at == revised
    assert original.content_recency_at == (revised or OLD)
    assert original.updated_at == ACCESS and original.access_count == 1


@pytest.mark.parametrize(
    "revised",
    [REVISION.replace(tzinfo=None), REVISION.astimezone(timezone(timedelta(hours=-5)))],
)
def test_revision_timezone_roundtrip_preserves_instant(store, revised):
    original = chunk("offset", OLD, revised=revised)
    store.store(original)
    result = store.get(original.id)
    assert result.content_updated_at == REVISION
    assert result.content_updated_at.utcoffset() == timedelta(0)
    assert result.content_recency_at > NEW


def legacy_database(path: Path):
    with sqlite3.connect(path) as connection:
        # Existing release databases already use WAL before migration starts.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE memory_chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "embedding TEXT, metadata TEXT NOT NULL DEFAULT '{}', "
            "salience REAL NOT NULL DEFAULT 0.5, cognitive_state TEXT, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "access_count INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1)"
        )
        connection.execute(
            "INSERT INTO memory_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "retained content",
                None,
                json.dumps(SCOPE),
                0.5,
                None,
                OLD.isoformat(),
                ACCESS.isoformat(),
                20,
                4,
            ),
        )


def test_migration_and_restart_preserve_unknown_history(tmp_path):
    path = tmp_path / "legacy.sqlite"
    legacy_database(path)
    for _ in range(2):
        store = SQLiteStore(str(path))
        try:
            result = store.get("legacy")
            assert result.content_updated_at is None
            assert result.created_at == OLD and result.updated_at == ACCESS
            assert result.version == 4 and result.access_count == 20
            assert result.content == "retained content" and result.metadata == SCOPE
            columns = [row[1] for row in store._conn.execute("PRAGMA table_info(memory_chunks)")]
            assert columns.count("content_updated_at") == 1
            assert store.get_stats()["schema_version"] == 3
        finally:
            store.close()


def test_sqlite_revision_survives_restart_and_another_client_touch(tmp_path):
    path = tmp_path / "revision.sqlite"
    first = SQLiteStore(str(path))
    first.store(chunk("known", OLD, revised=REVISION))
    first.close()
    reader = SQLiteStore(str(path))
    writer = SQLiteStore(str(path))
    try:
        assert writer.touch("known", updated_at=ACCESS)
        result = reader.get("known")
        assert result.content_updated_at == REVISION
        assert result.updated_at == ACCESS and result.access_count == 1
    finally:
        reader.close()
        writer.close()


def test_concurrent_legacy_open_adds_column_once(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    legacy_database(path)
    barrier = threading.Barrier(4)

    def open_store():
        barrier.wait(timeout=10)
        store = SQLiteStore(str(path))
        try:
            assert store.get("legacy").content_updated_at is None
            return store.get_stats()["schema_version"]
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: open_store(), range(4))) == [3, 3, 3, 3]
    with sqlite3.connect(path) as connection:
        assert [row[1] for row in connection.execute("PRAGMA table_info(memory_chunks)")].count(
            "content_updated_at"
        ) == 1
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_failed_migration_rolls_back_schema_and_retains_old_row(tmp_path):
    path = tmp_path / "rollback.sqlite"
    legacy_database(path)

    class FailingMigration(SQLiteStore):
        _CREATE_META_TABLE_SQL = "THIS IS NOT VALID SQL"

    with pytest.raises(sqlite3.OperationalError):
        FailingMigration(str(path))
    with sqlite3.connect(path) as connection:
        assert "content_updated_at" not in [
            row[1] for row in connection.execute("PRAGMA table_info(memory_chunks)")
        ]
        assert connection.execute(
            "SELECT content, updated_at, access_count, version FROM memory_chunks"
        ).fetchone() == ("retained content", ACCESS.isoformat(), 20, 4)
