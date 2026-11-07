"""Evidence lookup stays bounded without changing ordering or visibility."""

from __future__ import annotations

import sqlite3

import pytest

from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def populate(store):
    for identity in ("first", "second"):
        store.store(MemoryChunk(id=identity, content=identity))
    for identity, fact, created in [
        ("tie-z", "first", "2026-11-07T00:00:00Z"),
        ("tie-a", "first", "2026-11-07T00:00:00Z"),
        ("earlier", "first", "2026-11-06T00:00:00Z"),
        ("other", "second", "2026-11-07T00:00:00Z"),
    ]:
        store._conn.execute(
            "INSERT INTO fact_evidence(evidence_id,fact_id,excerpt,created_at) VALUES (?,?,?,?)",
            (identity, fact, "retained " + identity, created),
        )
    store._conn.commit()


def snapshot(store):
    return {
        fact: [entry.model_dump(mode="json") for entry in store.list_fact_evidence(fact)]
        for fact in ("first", "second", "absent")
    }


def assert_bounded_lookup(store):
    executed = []
    store._conn.set_trace_callback(executed.append)
    try:
        store.list_fact_evidence("first")
    finally:
        store._conn.set_trace_callback(None)
    query = next(sql for sql in executed if "FROM fact_evidence" in sql)
    plan = store._conn.execute("EXPLAIN QUERY PLAN " + query).fetchall()
    details = " ".join(row[3] for row in plan)
    assert "SEARCH fact_evidence" in details, details
    assert "SCAN fact_evidence" not in details, details
    assert "TEMP B-TREE" not in details, details


@pytest.mark.parametrize("upgrade", [False, True], ids=["new-store", "existing-store"])
def test_evidence_lookup_is_bounded_and_preserves_complete_rows(tmp_path, upgrade):
    path = tmp_path / "evidence.sqlite"
    store = SQLiteStore(str(path))
    try:
        populate(store)
        before = snapshot(store)
    finally:
        store.close()
    if upgrade:
        with sqlite3.connect(path) as old:
            old.execute("DROP INDEX IF EXISTS idx_fact_evidence_fact_time")
    reopened = SQLiteStore(str(path))
    try:
        assert snapshot(reopened) == before
        assert [row["evidence_id"] for row in before["first"]] == ["earlier", "tie-z", "tie-a"]
        assert before["absent"] == []
        assert_bounded_lookup(reopened)
        assert reopened._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close()
    again = SQLiteStore(str(path))
    try:
        assert snapshot(again) == before
        assert_bounded_lookup(again)
    finally:
        again.close()


def test_indexed_lookup_keeps_caller_transaction_and_equal_time_order(tmp_path):
    store = SQLiteStore(str(tmp_path / "transaction.sqlite"))
    try:
        populate(store)
        before = snapshot(store)
        assert_bounded_lookup(store)
        store._conn.execute("BEGIN")
        store._conn.execute(
            "INSERT INTO fact_evidence(evidence_id,fact_id,excerpt,created_at) VALUES (?,?,?,?)",
            ("last", "first", "retained last", "2026-11-07T00:00:00Z"),
        )
        assert [entry.evidence_id for entry in store.list_fact_evidence("first")] == [
            "earlier",
            "tie-z",
            "tie-a",
            "last",
        ]
        assert store._conn.in_transaction
        store._conn.rollback()
        assert snapshot(store) == before
        assert_bounded_lookup(store)
    finally:
        store.close()
