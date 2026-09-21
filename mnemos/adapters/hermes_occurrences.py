"""Keep capture occurrence times stable across Hermes provider restarts."""

from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Mapping

from ..service import _protect_private_path


def occurrence_times(hermes_home: str, observed: Mapping[str, str]) -> dict[str, str]:
    """Record first observation times without storing message content.

    The host owns this directory and includes it in its normal home backup.
    An atomic insert lets concurrent providers agree on one original payload.
    Do not discard or regenerate this ledger while its captures may replay.
    Every event in one call shares a transaction, because each commit is a
    full sync and a resumed session can bring hundreds of unseen rows.
    """
    home = Path(hermes_home).expanduser()
    if not home.is_absolute():
        raise ValueError("Hermes home must be absolute for durable capture identity")
    if not observed:
        return {}
    directory = home / "mnemos"
    directory.mkdir(parents=True, exist_ok=True)
    _protect_private_path(directory)
    path = directory / "occurrences.sqlite"
    recorded: dict[str, str] = {}
    with closing(sqlite3.connect(path, timeout=3.0)) as db:
        _protect_private_path(path)
        db.execute("PRAGMA synchronous=FULL")
        with db:
            # Reserve the writer before reading/creating the schema. Two
            # first-use providers must not both acquire a read lock and then
            # compete to upgrade it during CREATE or INSERT.
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS occurrences "
                "(event_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL)"
            )
            db.executemany(
                "INSERT OR IGNORE INTO occurrences(event_id, occurred_at) VALUES (?, ?)",
                list(observed.items()),
            )
            for event_id in observed:
                row = db.execute(
                    "SELECT occurred_at FROM occurrences WHERE event_id=?", (event_id,)
                ).fetchone()
                if row is None or not isinstance(row[0], str):
                    raise ValueError("Hermes capture occurrence record is invalid")
                recorded[event_id] = row[0]
    for value in recorded.values():
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("Hermes capture occurrence time must include a timezone")
    return recorded


def occurrence_time(hermes_home: str, event_id: str, observed_at: str) -> str:
    """Record one event's first observation time."""
    return occurrence_times(hermes_home, {event_id: observed_at})[event_id]
