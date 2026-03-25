from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mnemos.types import RetrievalFeedbackEvent
from mnemos.utils.storage import SQLiteStore


def test_sqlite_store_persists_feedback_event(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "feedback.sqlite"))
    event = RetrievalFeedbackEvent(
        event_type="helpful",
        query="deploy flow",
        scope="project",
        scope_id="repo-alpha",
        chunk_ids=["abc123", "def456"],
        notes="The deploy memory was exactly right.",
        created_at=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
    )
    try:
        store.store_feedback_event(event)
        events = store.list_feedback_events()
    finally:
        store.close()

    assert len(events) == 1
    assert events[0].event_type == "helpful"
    assert events[0].query == "deploy flow"
    assert events[0].scope == "project"
    assert events[0].scope_id == "repo-alpha"
    assert events[0].chunk_ids == ["abc123", "def456"]
    assert events[0].notes == "The deploy memory was exactly right."
    assert events[0].created_at == datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc)
