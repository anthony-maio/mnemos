"""Consistent SQLite snapshots and explicit restore into a new database path."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .scopes import ScopeRegistry
from .service import _protect_private_path
from .utils.storage import SQLiteStore


def _integrity(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise ValueError("Snapshot failed SQLite integrity check")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Snapshot has broken foreign keys")


def create_backup(
    store: SQLiteStore, destination: str | Path, registry: ScopeRegistry
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Backup destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mnemos-backup-", dir=destination.parent) as stage:
        stage_path = Path(stage)
        _protect_private_path(stage_path)
        snapshot = stage_path / "memory.sqlite"
        with sqlite3.connect(snapshot) as target:
            with store._lock:
                store._conn.backup(target)
            _integrity(target)
            schema = target.execute(
                "SELECT value FROM mnemos_meta WHERE key='schema_version'"
            ).fetchone()[0]
            facts = target.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
            suppressions = target.execute("SELECT COUNT(*) FROM memory_suppressions").fetchone()[0]
        target.close()
        payload = snapshot.read_bytes()
        registry_payload = registry.model_dump_json(indent=2).encode("utf-8")
        manifest = {
            "format_version": 1,
            "schema_version": int(schema),
            "facts": facts,
            "suppressions": suppressions,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database_sha256": hashlib.sha256(payload).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_payload).hexdigest(),
            "contains_credentials": False,
            "restore_note": "An old backup can predate later forget operations; merge current suppressions.",
        }
        archive = stage_path / "backup.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
            bundle.writestr("memory.sqlite", payload)
            bundle.writestr("scopes.json", registry_payload)
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
        _protect_private_path(archive)
        # Exclusive destination creation avoids overwriting another completed backup.
        os.link(archive, destination)
        return manifest


def restore_backup(
    archive: str | Path,
    destination: str | Path,
    *,
    current_store: SQLiteStore | None = None,
    current_registry: ScopeRegistry | None = None,
) -> tuple[dict[str, Any], ScopeRegistry]:
    """Restore into a new path; never overwrite or merge a live database implicitly."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Restore requires a new database destination")
    with ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != ["manifest.json", "memory.sqlite", "scopes.json"]:
            raise ValueError("Unexpected backup entries")
        manifest = json.loads(bundle.read("manifest.json"))
        payload, registry_payload = bundle.read("memory.sqlite"), bundle.read("scopes.json")
    if (
        manifest.get("format_version") != 1
        or hashlib.sha256(payload).hexdigest() != manifest.get("database_sha256")
        or hashlib.sha256(registry_payload).hexdigest() != manifest.get("registry_sha256")
    ):
        raise ValueError("Backup manifest or digest mismatch")
    registry = ScopeRegistry.model_validate_json(registry_payload)
    if registry.schema_version != 1:
        raise ValueError("Unsupported scope registry schema")
    if current_registry is not None:
        for identity, project in registry.projects.items():
            current = current_registry.projects.get(identity)
            if current is not None and (
                current.root != project.root or current.user_profile_id != project.user_profile_id
            ):
                raise ValueError("Restore would conflict with a current project identity")
        for identity, workspace in registry.workspaces.items():
            current_workspace = current_registry.workspaces.get(identity)
            if current_workspace is not None and current_workspace != workspace:
                raise ValueError("Restore would conflict with a current workspace identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mnemos-restore-", dir=destination.parent) as stage:
        stage_path = Path(stage)
        _protect_private_path(stage_path)
        snapshot = stage_path / "memory.sqlite"
        snapshot.write_bytes(payload)
        with sqlite3.connect(snapshot) as connection:
            _integrity(connection)
        connection.close()
        restored = SQLiteStore(str(snapshot))
        try:
            if current_store is not None:
                with current_store._lock:
                    rows = current_store._conn.execute(
                        "SELECT * FROM memory_suppressions"
                    ).fetchall()
                restored._conn.executemany(
                    "INSERT OR IGNORE INTO memory_suppressions VALUES (?, ?, ?, ?, ?, ?)", rows
                )
                restored._conn.commit()
                for chunk in restored.get_all():
                    if restored.is_suppressed(fact_id=chunk.id, canonical_key=chunk.canonical_key):
                        restored.forget_fact(chunk.id, expected_version=chunk.version)
                restored._conn.execute(
                    "DELETE FROM episodic_events WHERE id IN "
                    "(SELECT source_event_id FROM memory_suppressions WHERE source_event_id IS NOT NULL)"
                )
                restored._conn.execute(
                    "UPDATE memory_jobs SET status='completed', "
                    'result=\'{"decision":"suppressed"}\', lease_owner=NULL, lease_token=NULL, lease_until=NULL '
                    "WHERE json_extract(payload, '$.event_id') IN "
                    "(SELECT source_event_id FROM memory_suppressions WHERE source_event_id IS NOT NULL)"
                )
                restored._conn.commit()
            _integrity(restored._conn)
            restored._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            restored.close()
        _protect_private_path(snapshot)
        os.link(snapshot, destination)
    return {**manifest, "current_suppressions_merged": current_store is not None}, registry
