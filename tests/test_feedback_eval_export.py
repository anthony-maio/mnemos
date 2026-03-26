from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mnemos.benchmark import feedback_events_to_eval_rows, write_feedback_eval_dataset
from mnemos.types import RetrievalFeedbackEvent


def test_feedback_events_to_eval_rows_preserves_helpful_and_missed_memory_signals() -> None:
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
    events = [
        RetrievalFeedbackEvent(
            id="event-helpful",
            event_type="helpful",
            query="deploy flow",
            scope="project",
            scope_id="repo-alpha",
            chunk_ids=["chunk-deploy"],
            notes="Exactly the right deployment note.",
            created_at=now,
        ),
        RetrievalFeedbackEvent(
            id="event-missed",
            event_type="missed_memory",
            query="incident playbook",
            scope="workspace",
            scope_id="client-a",
            chunk_ids=[],
            notes="Should have recalled the runbook.",
            created_at=now + timedelta(minutes=1),
        ),
    ]

    rows = feedback_events_to_eval_rows(events)

    assert len(rows) == 2
    assert rows[0]["id"] == "event-helpful"
    assert rows[0]["query"] == "deploy flow"
    assert rows[0]["feedback_event_type"] == "helpful"
    assert rows[0]["current_scope"] == "project"
    assert rows[0]["scope_id"] == "repo-alpha"
    assert rows[0]["relevant_chunk_ids"] == ["chunk-deploy"]
    assert rows[0]["retrieved_chunk_ids"] == ["chunk-deploy"]
    assert rows[1]["id"] == "event-missed"
    assert rows[1]["feedback_event_type"] == "missed_memory"
    assert rows[1]["relevant_chunk_ids"] == []
    assert rows[1]["retrieved_chunk_ids"] == []


def test_write_feedback_eval_dataset_writes_jsonl(tmp_path: Path) -> None:
    event = RetrievalFeedbackEvent(
        id="event-not-helpful",
        event_type="not_helpful",
        query="redis ttl",
        scope="project",
        scope_id="repo-alpha",
        chunk_ids=["chunk-wrong"],
        notes="Returned a cache note from another service.",
        created_at=datetime(2026, 3, 26, 12, 30, tzinfo=timezone.utc),
    )

    output_path = tmp_path / "real-world-feedback.jsonl"
    count = write_feedback_eval_dataset([event], output_path)

    assert count == 1
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "event-not-helpful"
    assert payload["feedback_event_type"] == "not_helpful"
    assert payload["query"] == "redis ttl"
    assert payload["relevant_chunk_ids"] == []
    assert payload["retrieved_chunk_ids"] == ["chunk-wrong"]
