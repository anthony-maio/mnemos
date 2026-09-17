from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from mnemos.backup import create_backup, restore_backup
from mnemos.scopes import ScopeRegistry
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore


def test_backup_during_uncommitted_multirow_write_restores_committed_snapshot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.sqlite"
    archive = tmp_path / "snapshot.zip"
    restored_path = tmp_path / "restored.sqlite"
    store = SQLiteStore(str(database))
    store.store(MemoryChunk(id="one", content="before one", embedding=[1.0, 0.0]))
    store.store(MemoryChunk(id="two", content="before two", embedding=[0.0, 1.0]))
    registry = ScopeRegistry(profiles=["profile-a"])
    writer_ready = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []

    def write_transaction() -> None:
        connection = sqlite3.connect(str(database), timeout=10)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE memory_chunks SET content=? WHERE id=?",
                ("after one", "one"),
            )
            connection.execute(
                "UPDATE memory_chunks SET content=? WHERE id=?",
                ("after two", "two"),
            )
            writer_ready.set()
            if not release_writer.wait(timeout=10):
                raise TimeoutError("backup did not finish before writer release")
            connection.commit()
        except BaseException as exc:
            writer_errors.append(exc)
            connection.rollback()
        finally:
            connection.close()

    writer = threading.Thread(target=write_transaction)
    writer.start()
    try:
        assert writer_ready.wait(timeout=10)
        assert store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        manifest = create_backup(store, archive, registry)
    finally:
        release_writer.set()
        writer.join(timeout=10)
        store.close()

    assert not writer.is_alive()
    assert writer_errors == []
    assert manifest["facts"] == 2

    live = SQLiteStore(str(database))
    try:
        assert live.get("one").content == "after one"
        assert live.get("two").content == "after two"
    finally:
        live.close()

    restored_manifest, restored_registry = restore_backup(archive, restored_path)
    assert restored_manifest["database_sha256"]
    assert restored_registry.profiles == ["profile-a"]
    restored = SQLiteStore(str(restored_path))
    try:
        assert restored.get("one").content == "before one"
        assert restored.get("two").content == "before two"
        assert restored._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        restored.close()
