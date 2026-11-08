import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from mnemos.client import Outbox
from mnemos.sqlite_outbox import SQLiteOutbox


@pytest.fixture
def queue(tmp_path):
    value = SQLiteOutbox(tmp_path / "queue.sqlite")
    yield value
    value.close()


def test_json_backend_identity_payload_and_order_parity(queue, tmp_path):
    original = Outbox(tmp_path / "queue.json")
    for session in ("one", "two"):
        payload = {
            "event_id": "same-source",
            "content": "Remember the release date",
            "context": {"host": "codex", "user_profile_id": "profile", "session_id": session},
        }
        assert queue.append(payload) == original.append(payload)
    assert queue.pending() == original.pending()
    first = queue.pending()[0]["event_id"]
    assert queue.acknowledge(first) == original.acknowledge(first) is True
    assert queue.pending() == original.pending()


def test_duplicate_conflict_and_reopen(queue):
    first = queue.append({"content": "one"}, event_id="stable")
    assert queue.append({"content": "one"}, event_id="stable") == first
    with pytest.raises(ValueError, match="different payload"):
        queue.append({"content": "two"}, event_id="stable")
    queue.close()
    assert len(queue.pending()) == 1 and queue.pending()[0]["payload"]["content"] == "one"
    assert queue.acknowledge(first) is True
    queue.close()
    assert queue.pending() == [] and queue.acknowledge(first) is False


def test_reopened_reader_observes_committed_backlog_during_another_write(queue):
    queue.append({"content": "committed"}, event_id="one")
    queue.close()
    writer = sqlite3.connect(queue.path, isolation_level=None)
    reader = SQLiteOutbox(queue.path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("DELETE FROM outbox_entries")
        # Reopening for inspection must not acquire a schema write lock, and
        # must not expose the other connection's uncommitted acknowledgment.
        assert [row["event_id"] for row in reader.pending()] == ["one"]
        writer.execute("COMMIT")
        assert reader.pending() == []
    finally:
        writer.close()
        reader.close()


def test_full_sync_wal_is_required(queue):
    queue.append({"content": "record"}, event_id="one")
    connection = queue._connection
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_rejected_commit_never_returns_receipt_and_rolls_back(queue):
    queue.pending()
    connection = queue._connection

    def deny_commit(action, value, *unused):
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
            else sqlite3.SQLITE_OK
        )

    connection.set_authorizer(deny_commit)
    with pytest.raises(ValueError, match="not acknowledged"):
        queue.append({"content": "record"}, event_id="one")
    connection.set_authorizer(None)
    assert not connection.in_transaction and queue.pending() == []


def test_failed_acknowledgment_retains_pending_record(queue):
    identity = queue.append({"content": "record"}, event_id="one")
    connection = queue._connection
    connection.set_authorizer(
        lambda action, value, *rest: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
            else sqlite3.SQLITE_OK
        )
    )
    with pytest.raises(ValueError):
        queue.acknowledge(identity)
    connection.set_authorizer(None)
    assert queue.pending()[0]["event_id"] == identity


def test_batch_acknowledgment_commits_once_and_preserves_new_entries(queue):
    for identity in ("one", "two", "new"):
        queue.append({"content": identity}, event_id=identity)
    statements = []
    queue._connection.set_trace_callback(statements.append)
    assert queue.acknowledge_many(["one", "two", "one", "missing"]) == 2
    assert sum(s == "COMMIT" for s in statements) == 1
    assert [r["event_id"] for r in queue.pending()] == ["new"]
    queue.close()
    assert [r["event_id"] for r in queue.pending()] == ["new"]


def test_failed_batch_commit_retains_every_entry(queue):
    for identity in ("one", "two"):
        queue.append({"content": identity}, event_id=identity)
    connection = queue._connection
    connection.set_authorizer(
        lambda action, value, *rest: sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
        else sqlite3.SQLITE_OK
    )
    with pytest.raises(ValueError, match="not acknowledged"):
        queue.acknowledge_many(["one", "two"])
    connection.set_authorizer(None)
    queue.close()
    assert [r["event_id"] for r in queue.pending()] == ["one", "two"]


def test_two_connections_append_without_lost_records(queue):
    other = SQLiteOutbox(queue.path)
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(
                    (queue if i % 2 else other).append, {"content": str(i)}, event_id=str(i)
                )
                for i in range(20)
            ]
            assert len({f.result() for f in futures}) == 20
        assert len(queue.pending()) == len(other.pending()) == 20
    finally:
        other.close()


def test_busy_external_writer_does_not_acknowledge(queue):
    queue.pending()
    external = sqlite3.connect(queue.path, isolation_level=None)
    try:
        external.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="not acknowledged"):
            queue.append({"content": "record"}, event_id="one")
        external.execute("ROLLBACK")
        assert queue.pending() == []
    finally:
        external.close()


def test_foreign_database_not_repurposed(queue):
    connection = sqlite3.connect(queue.path)
    connection.execute("CREATE TABLE user_data (value TEXT)")
    connection.execute("INSERT INTO user_data VALUES ('preserve')")
    connection.commit()
    connection.close()
    original = queue.path.read_bytes()
    with pytest.raises(ValueError, match="supported SQLite outbox"):
        queue.append({"content": "record"})
    assert queue.path.read_bytes() == original


def test_json_outbox_not_automatically_migrated(queue):
    queue.path.write_text('[{"event_id":"preserve"}]')
    original = queue.path.read_bytes()
    with pytest.raises(ValueError):
        queue.append({"content": "record"})
    assert queue.path.read_bytes() == original


def test_payload_copy_and_firewall_failure_leave_no_partial_entry(queue, monkeypatch):
    payload = {"content": "record", "nested": {"value": "original"}}
    queue.append(payload, event_id="one")
    payload["nested"]["value"] = "changed"
    assert queue.pending()[0]["payload"]["nested"]["value"] == "original"

    def reject(value):
        raise ValueError("blocked")

    monkeypatch.setattr(queue, "_sanitize", reject)
    with pytest.raises(ValueError, match="blocked"):
        queue.append(payload, event_id="two")
    assert len(queue.pending()) == 1


def test_committed_wal_survives_process_exit_without_close(queue):
    code = "from mnemos.sqlite_outbox import SQLiteOutbox; import os,sys; q=SQLiteOutbox(sys.argv[1]); q.append({'content':'record'},event_id='one'); os._exit(0)"
    result = subprocess.run(
        [sys.executable, "-c", code, str(queue.path)],
        capture_output=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr.decode()
    assert queue.pending()[0]["event_id"] == "one"


def test_separate_processes_commit_without_losing_events(queue):
    queue.pending()
    code = (
        "from mnemos.sqlite_outbox import SQLiteOutbox; import sys; "
        "q=SQLiteOutbox(sys.argv[1]); "
        "[q.append({'content':str(i)},event_id=sys.argv[2]+str(i)) for i in range(10)]; "
        "q.close()"
    )
    children = []
    try:
        for prefix in ("a", "b"):
            children.append(
                subprocess.Popen(
                    [sys.executable, "-c", code, str(queue.path), prefix],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            )
        for child in children:
            _, errors = child.communicate(timeout=20)
            assert child.returncode == 0, errors.decode()
        assert {r["event_id"] for r in queue.pending()} == {
            prefix + str(i) for prefix in ("a", "b") for i in range(10)
        }
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_foreign_view_database_is_not_changed(queue):
    connection = sqlite3.connect(queue.path)
    connection.execute("CREATE VIEW user_data AS SELECT 1")
    connection.close()
    original = queue.path.read_bytes()
    with pytest.raises(ValueError, match="supported SQLite outbox"):
        queue.pending()
    assert queue.path.read_bytes() == original


def test_replaced_open_path_never_acknowledges(queue):
    queue.append({"content": "original"}, event_id="one")
    moved = queue.path.with_name("moved.sqlite")
    try:
        queue.path.rename(moved)
    except PermissionError:
        pytest.skip("Operating system prevents renaming this open SQLite database")
    queue.path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="replaced"):
        queue.append({"content": "new"}, event_id="two")
    assert queue.path.read_bytes() == b"replacement"
