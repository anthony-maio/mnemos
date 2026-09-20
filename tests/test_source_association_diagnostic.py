from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnemos.jobs import DurableJob, FactEvidence
from mnemos.types import EpisodicEvent, MemoryChunk
from mnemos.utils.storage import SQLiteStore

from benchmarks.local_memory.source_association_diagnostic import (
    inspect_source_associations,
)


def _event(
    event_id: str,
    *,
    profile: str | None = "profile-a",
    scope: str = "project",
    scope_id: str = "project-a",
    workspace_id: str | None = "workspace-a",
) -> EpisodicEvent:
    metadata: dict[str, object] = {
        "user_profile_id": profile,
        "project_id": scope_id if scope == "project" else "project-a",
        "workspace_id": workspace_id,
    }
    return EpisodicEvent(
        id=event_id,
        event_fingerprint=f"fingerprint-{event_id}",
        role="user",
        content=f"source {event_id}",
        normalized_content=f"source {event_id}",
        scope=scope,
        scope_id=None if scope == "global" else scope_id,
        source="manual",
        metadata=metadata,
    )


def _commit_group(
    store: SQLiteStore,
    event: EpisodicEvent,
    facts: list[tuple[str, str, str | None, dict[str, object] | None]],
    *,
    source_event_id_override: str | None = None,
    null_source_event_id: bool = False,
) -> None:
    store.append_episodic_event(event)
    job = DurableJob(
        id=f"job-{event.id}",
        kind="extract",
        payload={"event_id": event.id},
    )
    store.enqueue_job(job)
    claimed = store.claim_jobs(owner=f"owner-{event.id}")[0]
    chunks: list[MemoryChunk] = []
    evidence: list[FactEvidence] = []
    for fact_id, content, excerpt, overrides in facts:
        metadata: dict[str, object] = {
            "scope": event.scope,
            "scope_id": event.scope_id,
            "project_id": event.metadata.get("project_id"),
            "workspace_id": event.metadata.get("workspace_id"),
            "user_profile_id": event.metadata.get("user_profile_id"),
            "assertion_status": "asserted",
            "source": "explicit_remember",
        }
        metadata.update(overrides or {})
        chunks.append(
            MemoryChunk(
                id=fact_id,
                content=content,
                embedding=[1.0, 0.0],
                metadata=metadata,
                version=1,
            )
        )
        evidence.append(
            FactEvidence(
                evidence_id=f"evidence-{fact_id}",
                fact_id=fact_id,
                source_event_id=(
                    None
                    if null_source_event_id
                    else event.id if source_event_id_override is None else source_event_id_override
                ),
                excerpt=content if excerpt is None else excerpt,
            )
        )
    assert store.commit_fact_and_job(
        claimed.id,
        owner=f"owner-{event.id}",
        lease_token=claimed.lease_token,
        chunks=chunks,
        evidence=evidence,
        processed_event_ids=[event.id],
        result={"decision": "facts"},
    )


def _run(db_path: Path, *, fanout_limit: int = 16) -> dict[str, Any]:
    return inspect_source_associations(db_path, fanout_limit=fanout_limit)


def test_shared_source_event_derives_deterministic_fact_pair(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        _commit_group(
            store,
            _event("event-shared"),
            [("fact-a", "Alpha fact", None, None), ("fact-b", "Beta fact", None, None)],
        )
    finally:
        store.close()

    before_bytes = db_path.read_bytes()
    result = _run(db_path)
    assert db_path.read_bytes() == before_bytes
    assert result["relations"] == [
        {
            "relation_id": result["relations"][0]["relation_id"],
            "source_event_id": "event-shared",
            "fact_ids": ["fact-a", "fact-b"],
            "fact_versions": {"fact-a": 1, "fact-b": 1},
            "evidence_ids": ["evidence-fact-a", "evidence-fact-b"],
            "scope": {
                "scope": "project",
                "scope_id": "project-a",
                "user_profile_id": "profile-a",
            },
            "fact_scopes": {
                "fact-a": {
                    "scope": "project",
                    "scope_id": "project-a",
                    "user_profile_id": "profile-a",
                },
                "fact-b": {
                    "scope": "project",
                    "scope_id": "project-a",
                    "user_profile_id": "profile-a",
                },
            },
            "source_scope": {
                "scope": "project",
                "scope_id": "project-a",
                "user_profile_id": "profile-a",
            },
            "source_role": "user",
            "source_type": "manual",
        }
    ]
    assert result["summary"] == {
        "source_event_groups": 1,
        "eligible_source_event_groups": 1,
        "relation_count": 1,
        "rejected_source_event_groups": 0,
        "insufficient_source_event_groups": 0,
        "overflow_source_event_groups": 0,
    }


def test_distinct_source_events_do_not_form_a_pair(tmp_path: Path) -> None:
    db_path = tmp_path / "distinct.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        _commit_group(store, _event("event-a"), [("fact-a", "Alpha", None, None)])
        _commit_group(store, _event("event-b"), [("fact-b", "Beta", None, None)])
    finally:
        store.close()

    result = _run(db_path)
    assert result["relations"] == []
    assert all(group["status"] == "insufficient_facts" for group in result["groups"])


def test_restart_and_reverse_fact_insertion_are_byte_equal_diagnostics(tmp_path: Path) -> None:
    forward_path = tmp_path / "forward.sqlite"
    reverse_path = tmp_path / "reverse.sqlite"
    for db_path, order in ((forward_path, False), (reverse_path, True)):
        store = SQLiteStore(db_path=str(db_path))
        try:
            facts: list[tuple[str, str, str | None, dict[str, object] | None]] = [
                ("fact-a", "Alpha", None, None),
                ("fact-b", "Beta", None, None),
            ]
            _commit_group(store, _event("event-shared"), list(reversed(facts)) if order else facts)
        finally:
            store.close()

    forward = _run(forward_path)
    reopened = SQLiteStore(db_path=str(forward_path))
    reopened.close()
    restarted = _run(forward_path)
    reverse = _run(reverse_path)
    assert forward["relations"] == restarted["relations"] == reverse["relations"]
    assert forward["groups"] == restarted["groups"] == reverse["groups"]


@pytest.mark.parametrize(
    ("kind", "expected_reason"),
    [
        ("foreign", "scope_mismatch"),
        ("foreign_project", "scope_mismatch"),
        ("missing", "missing_source_event"),
        ("stale", "unverifiable_current_content"),
        ("suppressed", "suppressed_source_event"),
        ("suppressed_fact", "suppressed_fact"),
        ("malformed", "profile_metadata_mismatch"),
        ("invalid_role", "unsupported_event_role"),
        ("invalid_type", "missing_source_type"),
        ("null_source", "missing_source_event_id"),
        ("invalid_version", "invalid_fact_version"),
    ],
)
def test_invalid_source_groups_are_rejected_with_explicit_reason(
    tmp_path: Path, kind: str, expected_reason: str
) -> None:
    db_path = tmp_path / f"{kind}.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        facts: list[tuple[str, str, str | None, dict[str, object] | None]]
        if kind == "foreign":
            event = _event("event-foreign", profile="profile-a")
            facts = [
                ("fact-a", "Alpha", None, {"user_profile_id": "profile-b"}),
                ("fact-b", "Beta", None, {"user_profile_id": "profile-b"}),
            ]
        elif kind == "foreign_project":
            event = _event("event-foreign-project", profile="profile-a")
            facts = [
                (
                    "fact-a",
                    "Alpha",
                    None,
                    {
                        "scope_id": "project-b",
                        "project_id": "project-b",
                        "workspace_id": "workspace-b",
                    },
                ),
                (
                    "fact-b",
                    "Beta",
                    None,
                    {
                        "scope_id": "project-b",
                        "project_id": "project-b",
                        "workspace_id": "workspace-b",
                    },
                ),
            ]
        elif kind == "missing":
            event = _event("event-retained")
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        elif kind == "suppressed":
            event = _event("event-suppressed")
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        elif kind == "suppressed_fact":
            event = _event("event-suppressed-fact")
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        elif kind == "invalid_role":
            event = _event("event-invalid-role")
            event.role = "bot"
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        elif kind == "invalid_type":
            event = _event("event-invalid-type")
            event.source = ""
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        elif kind == "null_source":
            event = _event("event-null-source")
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        elif kind == "invalid_version":
            event = _event("event-invalid-version")
            facts = [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)]
        else:
            event = _event("event-stale" if kind == "stale" else "event-malformed")
            if kind == "malformed":
                event.metadata["profile_id"] = "profile-foreign"
            facts = [("fact-a", "Alpha", "old alpha", None), ("fact-b", "Beta", None, None)]
        _commit_group(
            store,
            event,
            facts,
            source_event_id_override="event-does-not-exist" if kind == "missing" else None,
            null_source_event_id=kind == "null_source",
        )
        if kind == "invalid_version":
            store._conn.execute("UPDATE memory_chunks SET version='bad' WHERE id='fact-a'")
            store._conn.commit()
        if kind == "suppressed":
            store.record_suppression(source_event_id=event.id, reason="diagnostic test")
        if kind == "suppressed_fact":
            store.record_suppression(fact_id="fact-a", reason="diagnostic test")
    finally:
        store.close()

    result = _run(db_path)
    assert result["relations"] == []
    reasons = {reason for group in result["groups"] for reason in group["rejection_reasons"]}
    assert expected_reason in reasons


def test_fanout_bound_rejects_group_without_emitting_clique(tmp_path: Path) -> None:
    db_path = tmp_path / "fanout.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        _commit_group(
            store,
            _event("event-large"),
            [(f"fact-{index:02d}", f"Fact {index}", None, None) for index in range(17)],
        )
    finally:
        store.close()

    result = _run(db_path, fanout_limit=16)
    assert result["relations"] == []
    assert result["groups"][0]["status"] == "overflow"
    assert result["groups"][0]["fact_count"] == 17
    assert "fanout_limit_exceeded" in result["groups"][0]["rejection_reasons"]


def test_scope_compatible_workspace_project_bridge_is_allowed(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace-bridge.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        event = _event(
            "event-workspace",
            scope="workspace",
            scope_id="workspace-a",
            workspace_id="workspace-a",
        )
        fact_overrides: dict[str, object] = {
            "scope": "project",
            "scope_id": "project-a",
            "project_id": "project-a",
            "workspace_id": "workspace-a",
        }
        _commit_group(
            store,
            event,
            [("fact-a", "Alpha", None, fact_overrides), ("fact-b", "Beta", None, fact_overrides)],
        )
    finally:
        store.close()

    result = _run(db_path)
    assert len(result["relations"]) == 1


def test_scope_incompatible_workspace_project_bridge_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace-foreign.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        event = _event(
            "event-workspace",
            scope="workspace",
            scope_id="workspace-a",
            workspace_id="workspace-a",
        )
        fact_overrides: dict[str, object] = {
            "scope": "project",
            "scope_id": "project-b",
            "project_id": "project-b",
            "workspace_id": "workspace-b",
        }
        _commit_group(
            store,
            event,
            [("fact-a", "Alpha", None, fact_overrides), ("fact-b", "Beta", None, fact_overrides)],
        )
    finally:
        store.close()

    result = _run(db_path)
    assert result["relations"] == []
    assert "scope_mismatch" in result["groups"][0]["rejection_reasons"]


def test_fanout_limit_rejects_invalid_values(tmp_path: Path) -> None:
    db_path = tmp_path / "fanout-validation.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    store.close()
    with pytest.raises(ValueError, match="fanout_limit"):
        _run(db_path, fanout_limit=1)


def test_nonasserted_and_captured_facts_are_not_association_inputs(tmp_path: Path) -> None:
    db_path = tmp_path / "excluded-facts.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        _commit_group(
            store,
            _event("event-excluded"),
            [
                ("fact-captured", "Captured", None, {"source": "captured_passthrough"}),
                ("fact-unknown", "Unknown", None, {"assertion_status": "unknown"}),
            ],
        )
    finally:
        store.close()

    result = _run(db_path)
    assert result["relations"] == []
    reasons = {reason for group in result["groups"] for reason in group["rejection_reasons"]}
    assert {"captured_evidence", "nonasserted_fact"} <= reasons


def test_correction_and_forget_remove_stale_relation_but_keep_remaining_fact(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "lifecycle.sqlite"
    store = SQLiteStore(db_path=str(db_path))
    try:
        event = _event("event-lifecycle")
        _commit_group(
            store,
            event,
            [("fact-a", "Alpha", None, None), ("fact-b", "Beta", None, None)],
        )
        before = _run(db_path)
        assert len(before["relations"]) == 1

        corrected = store.get("fact-a")
        assert corrected is not None
        corrected.content = "Alpha corrected"
        corrected.version = 2
        store.store(corrected)
        after_correction = _run(db_path)
        assert after_correction["relations"] == []
        assert "unverifiable_current_content" in {
            reason for group in after_correction["groups"] for reason in group["rejection_reasons"]
        }

        assert store.forget_fact("fact-a", expected_version=2)
        store.close()
        reopened = SQLiteStore(db_path=str(db_path))
        try:
            after_forget = _run(db_path)
            assert after_forget["relations"] == []
            assert reopened.get("fact-a") is None
            assert reopened.get("fact-b") is not None
        finally:
            reopened.close()
            store = None  # type: ignore[assignment]
    finally:
        if store is not None:
            store.close()
