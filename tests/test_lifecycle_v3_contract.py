from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.contracts import load_dataset
from benchmarks.local_memory.generate_lifecycle_v3 import derive_v3

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "benchmarks/local_memory/datasets/local-memory-heldout-v2.jsonl"
V3 = ROOT / "benchmarks/local_memory/datasets/local-memory-heldout-v3.jsonl"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_v3_preserves_v2_counts_and_records_amendment() -> None:
    v2 = load_dataset(V2)
    v3 = load_dataset(V3)
    assert v3.version == "local-memory-heldout-v3"
    assert len(v3.sessions) == len(v2.sessions) == 220
    assert len(v3.heldout_sessions) == len(v2.heldout_sessions) == 200
    assert len(v3.facts) == len(v2.facts) == 240
    assert len(v3.queries) == len(v2.queries) == 220
    assert len(v3.state_pairs) == len(v2.state_pairs) == 50
    assert v3.manifest["amendment_v2_sha256"] == hashlib.sha256(V2.read_bytes()).hexdigest()
    assert v3.manifest["human_acceptance"] == "not_run"
    assert v3.manifest["advertised_as_real"] is False
    assert "repaired rerun" in v3.manifest["amendment_disclosure"]
    assert "no_new_events_or_base_source_event_field_edits" in v3.manifest["amendment_scope"]
    assert (
        "v2 event order"
        in v3.manifest["protocol"]["lifecycle_target_contract"]["source_event_preservation"]
    )

    old = {session.session_id: session for session in v2.sessions}
    for session in v3.sessions:
        prior = old[session.session_id]
        assert [(fact.fact_id, fact.content) for fact in session.facts] == [
            (fact.fact_id, fact.content) for fact in prior.facts
        ]
        assert [(query.query_id, query.text, query.kind) for query in session.queries] == [
            (query.query_id, query.text, query.kind) for query in prior.queries
        ]
        assert len(session.events) == len(prior.events)
        for amended, original in zip(session.events, prior.events):
            amended_base = {
                key: value
                for key, value in amended.__dict__.items()
                if key not in {"target_source_event_id", "target_fact_id", "replacement_content"}
            }
            original_base = {
                key: value
                for key, value in original.__dict__.items()
                if key not in {"target_source_event_id", "target_fact_id", "replacement_content"}
            }
            assert amended_base == original_base


def test_v2_remains_backwards_compatible() -> None:
    dataset = load_dataset(V2)
    assert dataset.version == "local-memory-heldout-v2"


def test_v3_generator_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "existing.jsonl"
    destination.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        derive_v3(V2, destination)
    assert destination.read_text(encoding="utf-8") == "preserve\n"


def test_v3_correct_and_forget_targets_are_source_backed() -> None:
    dataset = load_dataset(V3)
    corrections = [
        event
        for session in dataset.heldout_sessions
        for event in session.events
        if event.lifecycle_operation == "correct"
    ]
    forgets = [
        event
        for session in dataset.heldout_sessions
        for event in session.events
        if event.lifecycle_operation == "forget"
    ]
    assert len(corrections) == len(forgets) == 20
    assert all(event.target_source_event_id and event.target_fact_id for event in corrections)
    assert all(event.replacement_content for event in corrections)
    assert all(event.target_source_event_id and event.target_fact_id for event in forgets)
    assert all(event.replacement_content is None for event in forgets)
    for session in dataset.heldout_sessions:
        events = {event.event_id: event for event in session.events}
        for action in (*corrections, *forgets):
            if action.event_id not in events:
                continue
            assert action.target_source_event_id is not None
            target = events[action.target_source_event_id]
            assert target.role == "user"
            assert target.lifecycle_operation in {"capture", "remember"}
            if target.lifecycle_operation == "remember":
                assert target.write_mode == "explicit"


def test_v3_rejects_missing_target_event(tmp_path: Path) -> None:
    rows = _rows(V3)
    session = next(row for row in rows if row.get("session_id") == "heldout-041")
    action = next(event for event in session["events"] if event["lifecycle_operation"] == "correct")
    action["target_source_event_id"] = "does-not-exist"
    path = tmp_path / "invalid.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="target_source_event_id does not exist"):
        load_dataset(path)


def test_v3_requires_gold_evidence_to_match_target_event(tmp_path: Path) -> None:
    rows = _rows(V3)
    session = next(row for row in rows if row.get("session_id") == "heldout-041")
    target_fact = next(fact for fact in session["facts"] if fact["gold_status"] == "superseded")
    target_fact["evidence_event_ids"] = ["heldout-041-event-03"]
    path = tmp_path / "wrong-evidence.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="not gold evidence"):
        load_dataset(path)


def test_v3_rejects_one_sided_context_override(tmp_path: Path) -> None:
    rows = _rows(V3)
    session = next(row for row in rows if row.get("session_id") == "heldout-041")
    action = next(event for event in session["events"] if event["lifecycle_operation"] == "correct")
    action["target_project_id"] = "unrelated-project"
    path = tmp_path / "context-mismatch.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="target event context mismatch for target_project_id"):
        load_dataset(path)


def test_v3_requires_replacement_fact_in_action_evidence(tmp_path: Path) -> None:
    rows = _rows(V3)
    session = next(row for row in rows if row.get("session_id") == "heldout-041")
    action = next(event for event in session["events"] if event["lifecycle_operation"] == "correct")
    action["evidence_fact_ids"] = []
    path = tmp_path / "missing-replacement-evidence.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="replacement fact must be evidenced by correct action"):
        load_dataset(path)


def test_v3_rejects_late_target_and_forget_replacement(tmp_path: Path) -> None:
    rows = _rows(V3)
    session = next(row for row in rows if row.get("session_id") == "heldout-041")
    action = next(event for event in session["events"] if event["lifecycle_operation"] == "correct")
    action["target_source_event_id"] = "heldout-041-event-03"
    path = tmp_path / "late.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="target event must precede action"):
        load_dataset(path)

    rows = _rows(V3)
    session = next(row for row in rows if row.get("session_id") == "heldout-161")
    forget = next(event for event in session["events"] if event["lifecycle_operation"] == "forget")
    forget["replacement_content"] = "must be rejected"
    path = tmp_path / "forget-replacement.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="forget cannot have replacement_content"):
        load_dataset(path)
