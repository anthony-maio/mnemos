"""Synthetic mutation coverage for the successor outage gates."""

import json
from datetime import datetime, timezone

import pytest

from benchmarks.native_hosts.outage_acceptance import (
    require_health_gate,
    require_native_transcript,
    require_pending_sources_v2,
)


def health():
    return {
        "schema_version": 1,
        "status": "running",
        "store": {"db_path": "C:/fixture/memory.sqlite"},
        "runtime": {"store_type": "SQLiteStore", "database_path": "C:/fixture/memory.sqlite"},
        "capabilities": {"background_maintenance": False},
        "maintenance": {"scheduler_running": False},
    }


def transcript():
    return [
        {
            "type": "user",
            "message": {"role": "user", "content": "Submitted marker 618."},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Acknowledged marker 618."}],
            },
        },
    ]


def pending_state():
    context = {
        "host": "claude-code",
        "session_id": "session-1",
        "user_profile_id": "profile-1",
        "project_id": "project-1",
        "workspace_id": None,
        "scope": "project",
        "scope_id": "project-1",
    }
    payload = {
        "context": context,
        "event_id": "host-event",
        "identity_status": "normal",
        "role": "user",
        "content": "Submitted marker 618.",
        "source_kind": "claude_code_transcript",
        "source_ref": "native-uuid",
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
                "content": "Submitted marker 618.",
                "metadata": json.dumps(metadata),
                "status": "pending",
                "source": "claude-code",
                "scope": "project",
                "scope_id": "project-1",
                "source_event_id": "host-event",
                "user_profile_id": "profile-1",
                "session_id": "session-1",
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


def test_health_gate_accepts_pinned_running_dispatcher():
    require_health_gate(
        health(),
        expected_database="C:/fixture/memory.sqlite",
        module_config={
            "surprisal": False,
            "mutable_rag": False,
            "affective": False,
            "sleep": False,
            "spreading": False,
        },
    )


@pytest.mark.asyncio
async def test_health_gate_accepts_actual_local_service_health(tmp_path):
    from benchmarks.native_hosts.claude_service_outage_v1 import rows
    from tests.test_memory_service import service

    memory, _ = service(tmp_path)
    try:
        observed = await memory.dispatch("health", {})
        require_health_gate(
            observed,
            expected_database=tmp_path / "memory.sqlite",
            module_config={
                "surprisal": False,
                "mutable_rag": False,
                "affective": False,
                "sleep": False,
                "spreading": False,
            },
        )
        assert rows(tmp_path / "memory.sqlite")["integrity"] == "ok"
    finally:
        await memory.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(status="degraded"),
        lambda value: value["runtime"].update(database_path="C:/other.sqlite"),
        lambda value: value["runtime"].update(store_type="OtherStore"),
        lambda value: value["capabilities"].update(background_maintenance=True),
        lambda value: value["maintenance"].update(scheduler_running=True),
    ],
)
def test_health_gate_rejects_unpinned_state(mutation):
    value = health()
    mutation(value)
    with pytest.raises(ValueError):
        require_health_gate(
            value,
            expected_database="C:/fixture/memory.sqlite",
            module_config={
                "surprisal": False,
                "mutable_rag": False,
                "affective": False,
                "sleep": False,
                "spreading": False,
            },
        )


def test_transcript_gate_requires_exact_marker_ack_and_model():
    require_native_transcript(
        transcript(),
        submitted_marker="Submitted marker 618.",
        expected_ack="Acknowledged marker 618.",
        expected_model="claude-sonnet-4-6",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0]["message"].update(content="other"),
        lambda rows: rows[1]["message"].update(model="other-model"),
        lambda rows: rows[1]["message"].update(content=[{"type": "text", "text": "other"}]),
        lambda rows: rows[1]["message"].update(content=[{"type": "tool_use", "name": "capture"}]),
        lambda rows: rows[1]["message"].update(
            content=[{"type": "text", "text": "Acknowledged marker 618."}, {"type": "tool_result"}]
        ),
    ],
)
def test_transcript_gate_rejects_marker_model_ack_and_nested_tools(mutation):
    rows = transcript()
    mutation(rows)
    with pytest.raises(ValueError):
        require_native_transcript(
            rows,
            submitted_marker="Submitted marker 618.",
            expected_ack="Acknowledged marker 618.",
            expected_model="claude-sonnet-4-6",
        )


def test_successor_persistence_gate_checks_physical_identity_columns():
    state, pending = pending_state()
    require_pending_sources_v2(state, pending)
    state["events"][0]["session_id"] = "other-session"
    with pytest.raises(ValueError):
        require_pending_sources_v2(state, pending)


@pytest.mark.asyncio
@pytest.mark.parametrize("provisional", [False, True])
async def test_successor_gate_accepts_production_capture_schema(tmp_path, provisional):
    from benchmarks.native_hosts.claude_service_outage_v1 import rows
    from mnemos.claude_identity import scoped_event_id
    from tests.test_memory_service import service

    memory, context = service(tmp_path)
    context = context.model_copy(update={"host": "claude-code"})
    payload = {
        "context": context.model_dump(mode="json"),
        "event_id": scoped_event_id(context, "prompt:synthetic", "user", "Marker"),
        "identity_version": 2,
        "identity_status": "provisional_unassociated" if provisional else "normal",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": "Marker",
        "source_kind": "claude_code_transcript",
        "source_ref": "prompt:synthetic",
    }
    try:
        result = await memory.dispatch("capture", payload)
        pending = [{"event_id": result["event_id"], "operation": "capture", "payload": payload}]
        require_pending_sources_v2(rows(tmp_path / "memory.sqlite"), pending)
    finally:
        await memory.close()


@pytest.mark.parametrize(
    "field", ["scope", "scope_id", "source", "source_event_id", "user_profile_id"]
)
def test_successor_persistence_gate_rejects_physical_scope_mutations(field):
    state, pending = pending_state()
    state["events"][0][field] = "other"
    with pytest.raises(ValueError):
        require_pending_sources_v2(state, pending)
