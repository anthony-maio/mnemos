"""Offline outage acceptance rejects invented capture and wrong physical identity."""

from copy import deepcopy
import json

import pytest

from benchmarks.native_hosts.pilot_outage import (
    ACK,
    PROMPT,
    require_buffered,
    require_no_tools,
    require_recovered,
)


def buffered():
    return [
        {
            "event_id": role,
            "operation": "capture",
            "payload": {
                "role": role,
                "content": content,
                "event_id": "native-" + role,
                "source_ref": event + ":turn",
                "source_kind": "codex_hook",
                "identity_status": "normal",
                "context": {
                    "host": "codex",
                    "session_id": "session",
                    "project_id": "project",
                    "user_profile_id": "profile",
                },
            },
        }
        for role, content, event in [
            ("user", PROMPT, "UserPromptSubmit"),
            ("assistant", ACK, "Stop"),
        ]
    ]


def recovered():
    events, jobs, chunks = [], [], []
    for entry in buffered():
        payload = entry["payload"]
        events.append(
            {
                "id": entry["event_id"],
                "content": payload["content"],
                "role": payload["role"],
                "scope": "project",
                "scope_id": "project",
                "source": "codex",
                "user_profile_id": "profile",
                "session_id": "session",
                "metadata": json.dumps(
                    {
                        "source_event_id": payload["event_id"],
                        "source_ref": payload["source_ref"],
                        "identity_status": "normal",
                    }
                ),
                "status": "consolidated",
            }
        )
        jobs.append(
            {
                "id": "event:" + entry["event_id"],
                "kind": "extract",
                "status": "completed",
                "payload": json.dumps({"event_id": entry["event_id"]}),
            }
        )
        chunks.append(
            {
                "id": entry["event_id"],
                "metadata": json.dumps(
                    {"source": "captured_passthrough", "assertion_status": "unknown"}
                ),
            }
        )
    return {"events": events, "jobs": jobs, "chunks": chunks}


def test_exact_native_buffer_and_recovery_are_accepted():
    require_buffered(
        buffered(), session_id="session", turn_id="turn", project_id="project", profile="profile"
    )
    require_no_tools([{"type": "message", "content": ACK}], ACK)
    require_recovered(
        recovered(), {"events": [], "jobs": [], "chunks": []}, buffered(), completed=True
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("project_id", "other"),
        ("user_profile_id", "other"),
        ("host", "claude-code"),
        ("scope", "global"),
    ],
)
def test_buffered_context_drift_is_rejected(field, value):
    entries = buffered()
    entries[0]["payload"]["context"][field] = value
    with pytest.raises(ValueError, match="identity or content"):
        require_buffered(
            entries, session_id="session", turn_id="turn", project_id="project", profile="profile"
        )


@pytest.mark.parametrize(
    "kind",
    [
        "function_call",
        "custom_tool_call",
        "tool_result",
        "function_call_output",
        "tool_search_call",
    ],
)
def test_any_nested_tool_activity_is_rejected(kind):
    with pytest.raises(ValueError, match="used a tool"):
        require_no_tools([{"payload": {"items": [{"type": kind}]}}], ACK)


@pytest.mark.parametrize(
    "table,key,value",
    [
        ("events", "scope_id", "other"),
        ("events", "role", "assistant"),
        ("jobs", "status", "pending"),
        ("chunks", "metadata", '{"source":"extracted","assertion_status":"asserted"}'),
    ],
)
def test_missing_identity_or_unfinished_processing_is_not_recovery(table, key, value):
    state = deepcopy(recovered())
    state[table][0][key] = value
    with pytest.raises(ValueError):
        require_recovered(
            state, {"events": [], "jobs": [], "chunks": []}, buffered(), completed=True
        )


def test_duplicate_or_missing_buffer_cannot_count_as_two_messages():
    for entries in [buffered()[:1], [buffered()[0], buffered()[0]]]:
        with pytest.raises(ValueError, match="two distinct"):
            require_buffered(
                entries,
                session_id="session",
                turn_id="turn",
                project_id="project",
                profile="profile",
            )
