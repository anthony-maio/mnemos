"""Exercise the authored correction, forget, and replay lifecycle on a fixture.

This is a benchmark control, rather than a production helper.  It deliberately
uses the storage lease and suppression APIs so that the report describes the
same durable transitions used by the runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from mnemos.jobs import DurableJob, FactEvidence, FactRevision
from mnemos.types import MemoryChunk
from mnemos.utils.storage import SQLiteStore

from .source_association_diagnostic import inspect_source_associations


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _vector(vectors: Mapping[str, Any], content: str) -> list[float]:
    raw = vectors.get(content)
    if isinstance(raw, Mapping):
        raw = raw.get("vector")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"missing frozen vector for {content!r}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        raise ValueError(f"invalid frozen vector for {content!r}")
    result = [float(item) for item in raw]
    if any(not isfinite(item) for item in result):
        raise ValueError(f"invalid frozen vector for {content!r}")
    if not result or not any(item != 0.0 for item in result):
        raise ValueError(f"invalid frozen vector for {content!r}")
    return result


def _snapshot(store: SQLiteStore, case: Mapping[str, Any], label: str) -> dict[str, Any]:
    fact_ids = [str(item["id"]) for item in case.get("facts", [])]
    facts = []
    for fact_id in fact_ids:
        chunk = store.get(fact_id)
        if chunk is None:
            facts.append({"id": fact_id, "present": False})
        else:
            facts.append(
                {
                    "id": chunk.id,
                    "present": True,
                    "content": chunk.content,
                    "version": chunk.version,
                    "metadata": dict(chunk.metadata),
                    "embedding": list(chunk.embedding or []),
                }
            )
    evidence = {
        fact_id: [item.model_dump(mode="json") for item in store.list_fact_evidence(fact_id)]
        for fact_id in fact_ids
    }
    revisions = {
        fact_id: [item.model_dump(mode="json") for item in store.list_fact_revisions(fact_id)]
        for fact_id in fact_ids
    }
    jobs = [item.model_dump(mode="json") for item in store.list_jobs()]
    events = [
        {"id": item.id, "status": item.status, "content": item.content}
        for item in store.list_episodic_events()
    ]
    suppressions = {
        "fact_ids": [fact_id for fact_id in fact_ids if store.is_suppressed(fact_id=fact_id)],
        "source_event_ids": [
            str(event["id"])
            for event in case.get("events", [])
            if store.is_suppressed(source_event_id=str(event["id"]))
        ],
    }
    return {
        "label": label,
        "facts": facts,
        "evidence": evidence,
        "revisions": revisions,
        "jobs": jobs,
        "episodic_events": events,
        "suppressions": suppressions,
        "source_associations": inspect_source_associations(store.db_path),
    }


def _reopen(db_path: Path, case: Mapping[str, Any], label: str) -> dict[str, Any]:
    reopened = SQLiteStore(db_path=str(db_path))
    try:
        return _snapshot(reopened, case, label)
    finally:
        reopened.close()


def run_lifecycle_control(
    case: Mapping[str, Any], vectors: Mapping[str, Any], db_path: str | Path
) -> dict[str, Any]:
    """Run the authored ``revision_forget_replay`` control on a materialized DB."""

    case_id = _text(case.get("id"), "case.id")
    if case_id != "revision_forget_replay":
        raise ValueError("lifecycle control requires the revision_forget_replay case")
    actions = case.get("lifecycle_actions")
    if not isinstance(actions, list) or [item.get("action") for item in actions] != [
        "correct",
        "forget",
        "replay",
        "restart",
    ]:
        raise ValueError("case lifecycle_actions must be correct, forget, replay, restart")
    destination = Path(db_path).expanduser().resolve()
    if not destination.is_file():
        raise FileNotFoundError(destination)
    correct, forget, replay, restart = actions
    correction_target_id = _text(correct.get("fact_id"), "correct.fact_id")
    forgotten_id = _text(forget.get("fact_id"), "forget.fact_id")
    target = None
    correction_event_id = _text(correct.get("new_source_event_id"), "correct.new_source_event_id")
    shared_event_id = _text(forget.get("source_event_id"), "forget.source_event_id")
    steps: dict[str, dict[str, Any]] = {}
    store = SQLiteStore(db_path=str(destination))
    try:
        initial = _snapshot(store, case, "initial")
        steps["initial"] = _reopen(destination, case, "initial-restart")
        target = store.get(correction_target_id)
        if target is None:
            raise ValueError(f"correction target is absent: {correction_target_id}")
        correction_event = store.get_episodic_event(correction_event_id)
        if correction_event is None:
            raise ValueError(f"correction source event is absent: {correction_event_id}")
        replay_event = store.get_episodic_event(shared_event_id)
        if replay_event is None:
            raise ValueError(f"replay source event is absent: {shared_event_id}")
        corrected_content = _text(correct.get("new_content"), "correct.new_content")
        authored_correction_version = int(correct.get("expected_version", 0))
        if target.version != authored_correction_version:
            raise ValueError("correction target version does not match authored expected_version")
        corrected_vector = _vector(vectors, corrected_content)
        if target.embedding is not None and len(corrected_vector) != len(target.embedding):
            raise ValueError("correction vector dimension does not match the fixture dimension")
        corrected = target.model_copy(
            update={
                "content": corrected_content,
                "embedding": corrected_vector,
                "version": target.version + 1,
                "updated_at": datetime.now(timezone.utc),
                "metadata": {
                    **target.metadata,
                    "previous_content": target.content,
                    "source": "explicit_correction",
                    "source_kind": "explicit_correction",
                    "assertion_status": "asserted",
                },
            }
        )
        owner = f"{case_id}:lifecycle-control"
        now = datetime.now(timezone.utc)
        correction_job = DurableJob(
            id=f"{case_id}:correction",
            kind="correct",
            payload={
                "event_id": correction_event_id,
                "target_fact_id": correction_target_id,
                "case_id": case_id,
            },
            created_at=now,
            updated_at=now,
            available_at=now,
        )
        store.capture_event_and_job(correction_event, correction_job)
        claimed = store.claim_jobs(owner=owner, job_id=correction_job.id, now=now)
        if len(claimed) != 1:
            raise RuntimeError("correction job was not leased")
        committed = store.commit_fact_and_job(
            claimed[0].id,
            owner=owner,
            lease_token=claimed[0].lease_token,
            chunk=corrected,
            revision=FactRevision(
                fact_id=correction_target_id,
                version=corrected.version,
                content=corrected_content,
                reason="explicit_correction",
                created_at=now,
            ),
            evidence=[
                FactEvidence(
                    fact_id=correction_target_id,
                    source_event_id=correction_event_id,
                    excerpt=corrected_content,
                    source_ref=f"{case_id}:{correction_event_id}",
                    created_at=now,
                )
            ],
            processed_event_ids=[correction_event_id],
            expected_versions={correction_target_id: target.version},
            result={"decision": "facts", "lifecycle_control": "correction"},
            now=now,
        )
        if not committed:
            raise RuntimeError("correction job lease was not committed")
        steps["correct"] = _reopen(destination, case, "correct-restart")

        current = store.get(forgotten_id)
        if current is None:
            raise RuntimeError("corrected target disappeared before forget")
        authored_forget_version = int(forget.get("expected_version", 0))
        if current.version != authored_forget_version:
            raise ValueError("forget target version does not match authored expected_version")
        forgotten = store.forget_fact(forgotten_id, expected_version=authored_forget_version)
        steps["forget"] = _reopen(destination, case, "forget-restart")

        original_job_id = f"source-association-development-v1:{case_id}:{shared_event_id}"
        replay_job = store.get_job(original_job_id)
        if replay_job is None:
            raise ValueError(f"original source job is absent: {original_job_id}")
        replay_result = store.capture_event_and_job(replay_event, replay_job)
        steps["replay"] = _reopen(destination, case, "replay-restart")
        steps["restart"] = _reopen(destination, case, "final-restart")
    except BaseException as exc:
        destination.with_suffix(".lifecycle-failed.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "steps": steps,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        store.close()

    final = steps["restart"]
    corrected_snapshot = steps["correct"]
    surviving = {str(item) for item in forget.get("expect_surviving_fact_ids", [])}
    present = {str(item["id"]) for item in final["facts"] if item.get("present")}
    stale: dict[str, Any] = next(
        (item for item in final["facts"] if item["id"] == forgotten_id), {}
    )
    shared_group: dict[str, Any] = next(
        (
            item
            for item in final["source_associations"]["groups"]
            if item["source_event_id"] == shared_event_id
        ),
        {},
    )
    corrected_row: dict[str, Any] = next(
        (item for item in corrected_snapshot["facts"] if item["id"] == correction_target_id), {}
    )
    restart_relations = steps["replay"]["source_associations"]["relations"]
    final_relations = final["source_associations"]["relations"]
    correction_pass = (
        corrected_row.get("present") is True
        and corrected_row.get("version") == 2
        and corrected_row.get("content") == correct.get("new_content")
        and len(corrected_snapshot["revisions"].get(correction_target_id, [])) >= 1
        and any(
            item.get("source_event_id") == correction_event_id
            for item in corrected_snapshot["evidence"].get(correction_target_id, [])
        )
        and any(
            item["id"] == correction_target_id
            and item.get("content") == correct.get("new_content")
            and item.get("version") == 2
            for item in final["facts"]
        )
    )
    source_preserved = (
        shared_event_id in {item.get("id") for item in final["episodic_events"]}
        and shared_event_id not in final["suppressions"]["source_event_ids"]
    )
    restart_pass = restart_relations == final_relations
    old_source_relation_absent = not any(
        relation.get("source_event_id") == shared_event_id
        and correction_target_id in relation.get("fact_ids", [])
        for relation in corrected_snapshot["source_associations"]["relations"]
    )
    replay_safe = (
        steps["replay"]["facts"] == steps["forget"]["facts"]
        and steps["replay"]["source_associations"] == steps["forget"]["source_associations"]
        and all(
            job["status"] == "completed"
            for job in steps["replay"]["jobs"]
            if job["id"] == replay_job.id
        )
    )
    controls: dict[str, bool] = {
        "correction_pass": correction_pass,
        "forget_pass": forgotten,
        "replay_suppression_pass": replay_safe,
        "restart_pass": restart_pass,
        "source_preserved_pass": source_preserved,
        "correction_committed": correction_pass,
        "forget_committed": forgotten,
        "authored_forget_version_was_current": True,
        "forgotten_fact_absent_after_replay": not stale.get("present", False),
        "surviving_facts_preserved": surviving.issubset(present),
        "shared_source_retained": source_preserved,
        "shared_source_no_longer_relates_forgotten_fact": forgotten_id
        not in shared_group.get("eligible_fact_ids", []),
        "stale_old_source_relation_absent": old_source_relation_absent,
    }
    operational = all(
        controls[key]
        for key in ("correction_pass", "forget_pass", "replay_suppression_pass", "restart_pass")
    )
    return {
        "schema_version": "source-association-lifecycle-v1",
        "status": (
            "completed" if operational and source_preserved else "completed_with_control_failures"
        ),
        "case_id": case_id,
        "db_path": str(destination),
        "steps": steps,
        "observations": {
            "authored_forget_version": authored_forget_version,
            "effective_forget_version": authored_forget_version,
            "replay_job_id": replay_job.id,
            "replay_capture_returned_event": replay_result is not None,
            "replay_suppressed": replay_result is None,
            "replay_capture_mode": (
                "duplicate_acknowledgement" if replay_result is not None else "suppressed"
            ),
        },
        "controls": controls,
    }


__all__ = ["run_lifecycle_control"]
