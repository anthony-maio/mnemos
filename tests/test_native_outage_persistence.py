"""An acknowledgement alone cannot satisfy the outage persistence gate."""

import json

import pytest

from benchmarks.native_hosts.claude_service_outage_v1 import require_pending_sources


def example():
    context = {
        "host": "claude-code",
        "session_id": "s",
        "user_profile_id": "p",
        "project_id": "project",
        "workspace_id": None,
    }
    payload = {
        "context": context,
        "role": "user",
        "content": "Outage marker",
        "source_ref": "native-uuid",
        "event_id": "host-event",
        "source_kind": "claude_code_transcript",
    }
    pending = [{"event_id": "durable-event", "operation": "capture", "payload": payload}]
    metadata = {
        **context,
        "source_event_id": "host-event",
        "source_ref": "native-uuid",
        "source_kind": "claude_code_transcript",
        "identity_status": "normal",
    }
    state = {
        "events": [
            {
                "id": "durable-event",
                "role": "user",
                "content": "Outage marker",
                "metadata": json.dumps(metadata),
                "status": "pending",
            }
        ],
        "jobs": [
            {
                "id": "event:durable-event",
                "kind": "extract",
                "status": "pending",
                "payload": json.dumps({"event_id": "durable-event"}),
            }
        ],
        "fact_count": 0,
        "integrity": "ok",
    }
    return state, pending


def test_raw_event_and_pending_job_match_saved_outbox():
    state, pending = example()
    require_pending_sources(state, pending)


@pytest.mark.parametrize(
    "change",
    [
        "missing_event",
        "missing_job",
        "wrong_content",
        "wrong_role",
        "wrong_scope",
        "wrong_source",
        "wrong_job_source",
        "completed_job",
        "fact_created",
        "integrity_error",
    ],
)
def test_receipts_cannot_hide_lost_or_changed_persistence(change):
    state, pending = example()
    if change == "missing_event":
        state["events"] = []
    elif change == "missing_job":
        state["jobs"] = []
    elif change == "wrong_content":
        state["events"][0]["content"] = "Different"
    elif change == "wrong_role":
        state["events"][0]["role"] = "assistant"
    elif change in {"wrong_scope", "wrong_source"}:
        metadata = json.loads(state["events"][0]["metadata"])
        metadata["project_id" if change == "wrong_scope" else "source_event_id"] = "different"
        state["events"][0]["metadata"] = json.dumps(metadata)
    elif change == "wrong_job_source":
        state["jobs"][0]["payload"] = json.dumps({"event_id": "different"})
    elif change == "completed_job":
        state["jobs"][0]["status"] = "completed"
    elif change == "fact_created":
        state["fact_count"] = 1
    elif change == "integrity_error":
        state["integrity"] = "error"
    with pytest.raises(ValueError):
        require_pending_sources(state, pending)


def test_provisional_source_stays_deferred_and_has_capture_only_job():
    state, pending = example()
    pending[0]["payload"]["identity_status"] = "provisional_unassociated"
    metadata = json.loads(state["events"][0]["metadata"])
    metadata["identity_status"] = "provisional_unassociated"
    state["events"][0]["metadata"] = json.dumps(metadata)
    state["events"][0]["status"] = "deferred"
    state["jobs"][0]["kind"] = "capture_only"
    require_pending_sources(state, pending)
    state["jobs"][0]["kind"] = "extract"
    with pytest.raises(ValueError):
        require_pending_sources(state, pending)


@pytest.mark.asyncio
@pytest.mark.parametrize("provisional", [False, True])
async def test_snapshot_gate_matches_actual_capture_schema(tmp_path, provisional):
    from datetime import datetime, timezone
    from mnemos.claude_identity import scoped_event_id
    from benchmarks.native_hosts.claude_service_outage_v1 import rows
    from tests.test_memory_service import service

    memory, context = service(tmp_path)
    context = context.model_copy(update={"host": "claude-code"})
    content = "The outage marker is birch-window-618."
    source_ref = "prompt:synthetic" if provisional else "native-uuid"
    payload = {
        "context": context.model_dump(mode="json"),
        "event_id": scoped_event_id(context, source_ref, "user", content),
        "identity_version": 2,
        "identity_status": "provisional_unassociated" if provisional else "normal",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": content,
        "source_kind": "claude_code_transcript",
        "source_ref": source_ref,
    }
    try:
        result = await memory.dispatch("capture", payload)
        assert result["status"] == "committed"
        pending = [{"event_id": result["event_id"], "operation": "capture", "payload": payload}]
        require_pending_sources(rows(tmp_path / "memory.sqlite"), pending)
    finally:
        await memory.close()
