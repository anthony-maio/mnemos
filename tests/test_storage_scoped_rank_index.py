"""Scoped rank optimization preserves complete rows and transactional visibility."""

from __future__ import annotations

import json
import re
import sqlite3

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def scope_filter(spec):
    def allowed(chunk):
        raise AssertionError("An engine-owned predicate must be compiled before ranking")

    allowed._mnemos_scope_filter = spec
    return allowed


def original_rows(store, query, limit, predicate, *, lexical):
    where, bound, compiled = store._scope_sql(predicate)
    assert compiled
    if lexical:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)[:64]
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        rows = store._conn.execute(
            f"SELECT {store._CHUNK_SELECT_COLUMNS_QUALIFIED} FROM memory_fts "
            "JOIN memory_chunks ON memory_chunks.rowid=memory_fts.rowid "
            f"WHERE {where} AND memory_fts MATCH ? "
            "ORDER BY bm25(memory_fts), memory_chunks.id LIMIT ?",
            [*bound, match, limit],
        ).fetchall()
    else:
        distance = """COALESCE((SELECT 1.0 - vec_distance_cosine(embedding, ?)
            FROM memory_vector_values WHERE rowid=memory_chunks.rowid AND dimension=?), 0.0)"""
        rows = store._conn.execute(
            f"SELECT {store._CHUNK_SELECT_COLUMNS}, {distance} AS similarity "
            f"FROM memory_chunks WHERE {where} ORDER BY similarity DESC, id LIMIT ?",
            [store._serialize_vec(query), len(query), *bound, limit],
        ).fetchall()
        rows = [row[:-1] for row in rows]
    return [store._row_to_chunk(row).model_dump(mode="json") for row in rows]


@pytest.fixture
def populated(tmp_path):
    store = SQLiteStore(str(tmp_path / "scoped-rank.sqlite"))
    entries = [
        ("a-project", {"scope": "project", "scope_id": "repo"}),
        ("b-global", {"scope": "global", "scope_id": None}),
        ("c-workspace", {"scope": "workspace", "scope_id": "workspace"}),
        ("d-member", {"scope": "project", "scope_id": "other", "workspace_id": "workspace"}),
        ("e-foreign", {"scope": "project", "scope_id": "repo", "user_profile_id": "other"}),
        ("f-captured", {"scope": "project", "scope_id": "repo", "source": "captured_passthrough"}),
        ("g-numeric", {"scope": "project", "scope_id": 7}),
        ("h-text", {"scope": "project", "scope_id": "7"}),
        ("i-boolean", {"scope": "project", "scope_id": "repo", "source": False}),
        ("j-object", {"scope": "project", "scope_id": "repo", "source": {"type": "manual"}}),
        ("k-unscoped", {}),
        ("l-legacy", {"scope": "project", "scope_id": None, "user_profile_id": None}),
        ("m-legacy", {"scope": "project", "scope_id": "  ", "user_profile_id": None}),
        ("n-legacy", {"scope": "global", "user_profile_id": None}),
    ]
    for identifier, metadata in reversed(entries):
        store.store(
            MemoryChunk(
                id=identifier,
                content="shared scope token " + identifier,
                embedding=[1.0, 0.0],
                metadata={"user_profile_id": "local", **metadata},
            )
        )
    try:
        yield store
    finally:
        store.close()


CASES = [
    (
        {
            "user_profile_id": "local",
            "project_id": "repo",
            "allowed_scopes": ["project"],
            "source_view": "memory",
        },
        ["a-project", "i-boolean", "j-object"],
    ),
    (
        {
            "user_profile_id": "local",
            "project_id": "repo",
            "allowed_scopes": ["project"],
            "source_view": "captured_evidence",
        },
        ["f-captured"],
    ),
    (
        {
            "user_profile_id": "local",
            "project_id": "repo",
            "workspace_id": "workspace",
            "allowed_scopes": ["project", "workspace", "global"],
            "source_view": "memory",
        },
        ["a-project", "b-global", "c-workspace", "i-boolean", "j-object"],
    ),
    (
        {
            "user_profile_id": "local",
            "member_workspace_id": "workspace",
            "allowed_scopes": ["project"],
        },
        ["d-member"],
    ),
    ({"user_profile_id": "local", "project_id": "7", "allowed_scopes": ["project"]}, ["h-text"]),
    (
        {"user_profile_id": "other", "project_id": "repo", "allowed_scopes": ["project"]},
        ["e-foreign"],
    ),
    (
        {"scope_id": "default", "allowed_scopes": ["project", "global"]},
        ["l-legacy", "m-legacy", "n-legacy"],
    ),
    ({"user_profile_id": "missing", "allowed_scopes": ["global"]}, []),
    ({"user_profile_id": "local", "allowed_scopes": []}, []),
]


@pytest.mark.parametrize("lexical", [False, True])
@pytest.mark.parametrize("spec,expected", CASES)
def test_scopes_views_types_and_complete_rows_match_original(populated, lexical, spec, expected):
    predicate = scope_filter(spec)
    query = "shared" if lexical else [1.0, 0.0]
    method = populated.retrieve_lexical if lexical else populated.retrieve
    for limit in (1, 30):
        actual = method(query, limit, predicate)
        assert [row.id for row in actual] == expected[:limit]
        assert [row.model_dump(mode="json") for row in actual] == original_rows(
            populated, query, limit, predicate, lexical=lexical
        )


@pytest.mark.parametrize("lexical", [False, True])
def test_unfiltered_and_all_profile_scopes_preserve_rows(populated, lexical):
    query = "shared" if lexical else [1.0, 0.0]
    method = populated.retrieve_lexical if lexical else populated.retrieve
    for predicate in (
        None,
        scope_filter(
            {
                "user_profile_id": "local",
                "all_profile_scopes": True,
                "allowed_scopes": ["project", "workspace", "global"],
            }
        ),
    ):
        actual = method(query, 30, predicate)
        assert [row.model_dump(mode="json") for row in actual] == original_rows(
            populated, query, 30, predicate, lexical=lexical
        )


def test_access_correction_deletion_and_reinsertion_keep_current_rows(populated):
    predicate = scope_filter(CASES[0][0])
    populated.mark_retrieved_many(["a-project", "i-boolean"])
    changed = populated.get("a-project").model_copy(
        update={"content": "corrected shared token", "version": 2}
    )
    assert populated.update("a-project", changed)
    assert populated.delete("i-boolean")
    for lexical, query in ((False, [1.0, 0.0]), (True, "shared")):
        method = populated.retrieve_lexical if lexical else populated.retrieve
        actual = method(query, 30, predicate)
        assert [row.model_dump(mode="json") for row in actual] == original_rows(
            populated, query, 30, predicate, lexical=lexical
        )
        assert "i-boolean" not in [row.id for row in actual]
        corrected = next(row for row in actual if row.id == "a-project")
        assert corrected.version == 2 and corrected.access_count == 1
    assert populated.delete("a-project")
    populated.store(changed)
    assert populated.retrieve([1.0, 0.0], 1, predicate)[0].id == "a-project"


def test_index_upgrade_is_visible_to_existing_writer_and_rollback(tmp_path):
    path = str(tmp_path / "existing.sqlite")
    writer = SQLiteStore(path)
    chunk = MemoryChunk(
        id="fact",
        content="shared current fact",
        embedding=[1.0, 0.0],
        metadata={"user_profile_id": "local", "scope": "project", "scope_id": "repo"},
    )
    writer.store(chunk)
    writer._conn.execute("DROP INDEX idx_memory_chunks_scope_rank")
    writer._conn.commit()
    reader = SQLiteStore(path)
    predicate = scope_filter(CASES[0][0])
    moved = {**chunk.metadata, "user_profile_id": "other", "source": "captured_passthrough"}
    try:
        for commit in (False, True):
            writer._conn.execute("BEGIN IMMEDIATE")
            writer._conn.execute(
                "UPDATE memory_chunks SET user_profile_id=?,metadata=? WHERE id='fact'",
                ("other", json.dumps(moved)),
            )
            assert [row.id for row in reader.retrieve([1.0, 0.0], 5, predicate)] == ["fact"]
            assert [row.id for row in reader.retrieve_lexical("shared", 5, predicate)] == ["fact"]
            if commit:
                writer._conn.commit()
            else:
                writer._conn.rollback()
                assert [row.id for row in reader.retrieve([1.0, 0.0], 5, predicate)] == ["fact"]
        assert reader.retrieve([1.0, 0.0], 5, predicate) == []
        assert reader.retrieve_lexical("shared", 5, predicate) == []
        assert reader.get("fact").metadata == moved
    finally:
        reader.close()
        writer.close()


def test_failed_index_creation_rolls_back_existing_database(tmp_path, monkeypatch):
    path = str(tmp_path / "rollback.sqlite")
    store = SQLiteStore(path)
    store.store(MemoryChunk(id="kept", content="Keep this fact", embedding=[1.0, 0.0]))
    store._conn.execute("DROP INDEX idx_memory_chunks_scope_rank")
    store._conn.commit()
    before = store._conn.execute("SELECT * FROM memory_chunks").fetchall()
    store.close()
    monkeypatch.setattr(
        SQLiteStore, "_CREATE_SCOPE_RANK_INDEX_SQL", "CREATE INDEX invalid ON absent(column)"
    )
    with pytest.raises(sqlite3.OperationalError):
        SQLiteStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM memory_chunks").fetchall() == before
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='idx_memory_chunks_scope_rank'"
        ).fetchone()


def test_custom_predicate_and_native_extension_fallback_remain_available(populated):
    selected = lambda chunk: chunk.id in {"a-project", "e-foreign"}
    assert [row.id for row in populated.retrieve([1.0, 0.0], 2, selected)] == [
        "a-project",
        "e-foreign",
    ]
    assert populated.retrieve_lexical("shared", 2, selected) == []
    populated._sqlite_vec_enabled = False
    assert [row.id for row in populated.retrieve([1.0, 0.0], 2, selected)] == [
        "a-project",
        "e-foreign",
    ]
