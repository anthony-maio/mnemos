"""Experimental SQLite outbox; no adapter selects this backend by default.

Use a separate path from the JSON outbox. No automatic migration is performed.
The owner must close the connection when its client lifecycle ends.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
from typing import Any, Iterator, Mapping

from .client import Outbox
from .service import _protect_private_path


class SQLiteOutbox(Outbox):
    # Bounded replay batches amortize deletion commits. Capture append receipts
    # remain individually durable. Interruption leaves delivered IDs replayable.
    replay_ack_batch_size = 32
    _connection: sqlite3.Connection | None = None
    _owner_pid: int | None = None
    _file_identity: tuple[int, int] | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None and self._owner_pid != os.getpid():
            raise ValueError("Outbox connection belongs to another process")
        if self._connection is not None:
            try:
                current = self.path.stat()
            except OSError as exc:
                raise ValueError("Open SQLite outbox path became unavailable") from exc
            if (current.st_dev, current.st_ino) != self._file_identity:
                raise ValueError("Open SQLite outbox path was replaced")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _protect_private_path(self.path.parent)
        for suffix in ("", "-wal", "-shm"):
            path = self.path.with_name(self.path.name + suffix)
            if path.exists():
                _protect_private_path(path)
        if self._connection is not None:
            return self._connection
        connection = sqlite3.connect(
            self.path, timeout=0.5, isolation_level=None, check_same_thread=False
        )
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            extra_objects = connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('view', 'trigger')"
            ).fetchall()
            if extra_objects or tables not in (set(), {"outbox_entries"}) or version not in (0, 1):
                raise ValueError("Not a supported SQLite outbox")
            if tables and (
                version != 1
                or connection.execute("PRAGMA table_info(outbox_entries)").fetchall()
                != [(0, "event_id", "TEXT", 1, None, 1), (1, "entry_json", "TEXT", 1, None, 0)]
            ):
                raise ValueError("Invalid SQLite outbox schema")
            if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise ValueError("SQLite outbox requires WAL journaling")
            connection.execute("PRAGMA synchronous=FULL")
            if connection.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise ValueError("SQLite outbox requires full synchronous commits")
            # A validated existing schema needs no write transaction. In WAL
            # mode, inspection can then proceed while another owner writes.
            if not tables:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS outbox_entries ("
                    "event_id TEXT PRIMARY KEY NOT NULL, "
                    "entry_json TEXT NOT NULL)"
                )
                connection.execute("PRAGMA user_version=1")
                connection.execute("COMMIT")
            _protect_private_path(self.path)
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._owner_pid = os.getpid()
        current = self.path.stat()
        self._file_identity = (current.st_dev, current.st_ino)
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = None
            try:
                connection = self._connect()
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                # Never return a receipt before this FULL synchronous commit.
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                raise ValueError("SQLite outbox operation failed; not acknowledged") from exc
            finally:
                if connection is not None and connection.in_transaction:
                    connection.execute("ROLLBACK")

    def append(
        self, payload: Mapping[str, Any], *, event_id: str | None = None, operation: str = "capture"
    ) -> str:
        entry = self._prepare_entry(payload, event_id=event_id, operation=operation)
        stable_id = str(entry["event_id"])
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT entry_json FROM outbox_entries WHERE event_id=?", (stable_id,)
            ).fetchone()
            if existing is not None:
                if json.loads(existing[0]) != entry:
                    raise ValueError("Outbox event identity already has a different payload")
            else:
                connection.execute(
                    "INSERT INTO outbox_entries VALUES (?, ?)",
                    (stable_id, json.dumps(entry, ensure_ascii=False, sort_keys=True)),
                )
        return stable_id

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            try:
                rows = (
                    self._connect()
                    .execute("SELECT entry_json FROM outbox_entries ORDER BY rowid")
                    .fetchall()
                )
                return [json.loads(row[0]) for row in rows]
            except sqlite3.Error as exc:
                raise ValueError("Invalid SQLite outbox") from exc

    def acknowledge(self, event_id: str) -> bool:
        return self.acknowledge_many([event_id]) == 1

    def acknowledge_many(self, event_ids: list[str]) -> int:
        if not isinstance(event_ids, list) or any(
            not isinstance(event_id, str) or not event_id for event_id in event_ids
        ):
            raise ValueError("Acknowledgment identities must be nonempty strings")
        if not event_ids:
            return 0
        identities = list(dict.fromkeys(event_ids))
        with self._transaction() as connection:
            removed = connection.executemany(
                "DELETE FROM outbox_entries WHERE event_id=?",
                [(event_id,) for event_id in identities],
            ).rowcount
        return removed

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._owner_pid = None
                self._file_identity = None
