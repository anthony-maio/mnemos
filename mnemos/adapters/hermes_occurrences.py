"""Keep capture occurrence times stable across Hermes provider restarts."""

from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3

from ..service import _protect_private_path


def occurrence_time(hermes_home: str, event_id: str, observed_at: str) -> str:
    """Record first observation time without storing message content.

    The host owns this directory and includes it in its normal home backup.
    An atomic insert lets concurrent providers agree on one original payload.
    Do not discard or regenerate this ledger while its captures may replay.
    """
    home = Path(hermes_home).expanduser()
    if not home.is_absolute():
        raise ValueError("Hermes home must be absolute for durable capture identity")
    directory = home / "mnemos"
    directory.mkdir(parents=True, exist_ok=True)
    _protect_private_path(directory)
    path = directory / "occurrences.sqlite"
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
            db.execute(
                "INSERT OR IGNORE INTO occurrences(event_id, occurred_at) VALUES (?, ?)",
                (event_id, observed_at),
            )
            row = db.execute(
                "SELECT occurred_at FROM occurrences WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None or not isinstance(row[0], str):
            raise ValueError("Hermes capture occurrence record is invalid")
        result = row[0]
        if datetime.fromisoformat(result).tzinfo is None:
            raise ValueError("Hermes capture occurrence time must include a timezone")
        return result
