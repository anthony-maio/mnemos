"""Native text must be matched independently of successful arrival counts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.native_hosts.capture_coverage import expected_messages, reconcile_captures


def claude_records() -> list[dict]:
    return [
        {
            "type": "user",
            "sessionId": "s1",
            "uuid": "u1",
            "message": {"role": "user", "content": "Keep this."},
        },
        {
            "type": "assistant",
            "sessionId": "s1",
            "uuid": "a1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Stored."}]},
        },
    ]


SCOPE = {"user_profile_id": "person", "project_id": "project", "workspace_id": None}
HOST_IDS = {
    "u1": "claude:v2:7b7397871f6c97cf790a1ee39a670d427aa0521c0a236960d66260ee7ca5f547",
    "a1": "claude:v2:e6e35216cfb55bc2ec13fa7299da36971a45234c9e4048fd9d133e5f367fcc6a",
}
DURABLE_IDS = {
    "u1": "6ffbc8cc4289b8fb06e27d9d204d97148c3cd31697e6bc75812eb953bb2d64f5",
    "a1": "06217cdf83215727481d5e9362e04e3339bbad0bb01bf80c15601d6259fecb37",
}


def receipts() -> list[dict]:
    return [
        {
            "phase": "teach",
            "operation": "capture",
            "payload": {
                "context": {**SCOPE, "host": "claude-code", "session_id": "s1"},
                "role": role,
                "source_ref": ref,
                "source_kind": "claude_code_transcript",
                "content": content,
                "event_id": HOST_IDS[ref],
            },
            "result": {
                "status": "committed",
                "event_id": DURABLE_IDS[ref],
                "job_id": "job-" + ref,
                "identity_status": "normal",
                "fact_eligible": True,
                "processing_status": "pending",
            },
        }
        for role, ref, content in [("user", "u1", "Keep this."), ("assistant", "a1", "Stored.")]
    ]


def audit(rows: list[dict]) -> dict:
    expected = expected_messages("claude-code", "teach", claude_records())
    return reconcile_captures(
        expected, rows, scope=SCOPE, jobs=[{"id": "job-u1", "status": "completed"}]
    )


def test_duplicate_arrivals_do_not_inflate_logical_coverage() -> None:
    rows = receipts()
    rows.append(deepcopy(rows[0]))
    result = audit(rows)
    assert result["status"] == "bounded_text_receipts_matched"
    assert result["expected_messages"] == result["covered_messages"] == 2
    assert result["duplicate_receipts"] == 1
    assert result["messages"][0]["job_statuses"] == ["completed"]
    assert result["messages"][1]["job_statuses"] == ["not_present"]
    assert "content" not in result["messages"][0]


@pytest.mark.parametrize(
    "field,value",
    [("role", "user"), ("content", "Changed."), ("source_ref", "wrong"), ("source_kind", "manual")],
)
def test_text_or_identity_mismatch_is_missing_even_with_same_arrival_count(
    field: str, value: str
) -> None:
    rows = receipts()
    rows[1]["payload"][field] = value
    result = audit(rows)
    assert result["status"] == "incomplete"
    assert result["covered_messages"] == 1
    assert result["unmatched_receipts"] == [1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_profile_id", "foreign"),
        ("project_id", "foreign"),
        ("workspace_id", "foreign"),
        ("session_id", "foreign"),
        ("host", "hermes"),
    ],
)
def test_other_context_cannot_satisfy_message(field: str, value: str) -> None:
    rows = receipts()
    rows[1]["payload"]["context"][field] = value
    assert audit(rows)["covered_messages"] == 1


@pytest.mark.parametrize(
    "result_patch",
    [
        {"status": "queued"},
        {"event_id": ""},
        {"identity_status": "provisional_unassociated", "fact_eligible": False},
    ],
)
def test_uncommitted_or_provisional_result_cannot_satisfy_native_identity(
    result_patch: dict,
) -> None:
    rows = receipts()
    rows[1]["result"].update(result_patch)
    assert audit(rows)["covered_messages"] == 1


def test_claude_provisional_prompt_is_separate_and_never_substitutes_for_native_user() -> None:
    rows = receipts()
    provisional = deepcopy(rows[0])
    provisional["payload"]["source_ref"] = "prompt:fingerprint"
    provisional["payload"][
        "event_id"
    ] = "claude:v2:0897ce613cbe9208dcdd9cf51c3f3bc558ba9450b5d849fe15149e713144c1ed"
    provisional["result"].update(
        event_id="61b9da0dab5a1c7419f4fa59332f206cf326febd6ebc715598e93ebd148d411c",
        identity_status="provisional_unassociated",
        fact_eligible=False,
        processing_status="deferred",
    )
    result = audit(rows + [provisional])
    assert result["provisional_receipts"] == [2]
    assert result["covered_messages"] == 2
    assert result["status"] == "bounded_text_receipts_matched"
    result = audit([rows[1], provisional])
    assert result["covered_messages"] == 1
    assert result["status"] == "incomplete"


def test_conflicting_durable_identity_is_not_a_duplicate_delivery() -> None:
    rows = receipts()
    extra = deepcopy(rows[0])
    extra["result"]["event_id"] = "different-durable-id"
    result = audit(rows + [extra])
    assert result["status"] == "incomplete"
    assert result["unmatched_receipts"] == [2]


def test_one_durable_identity_cannot_cover_two_messages() -> None:
    rows = receipts()
    rows[1]["result"]["event_id"] = rows[0]["result"]["event_id"]
    assert audit(rows)["status"] == "incomplete"


def test_suppressed_and_job_completion_are_reported_separately() -> None:
    rows = receipts()
    rows[0]["result"].update(processing_status="suppressed", fact_eligible=False)
    result = audit(rows)
    assert result["messages"][0]["suppressed"] is True
    assert result["messages"][0]["job_statuses"] == ["completed"]


def test_other_phase_replay_is_not_native_delivery() -> None:
    rows = receipts()
    replay = deepcopy(rows[0])
    replay["phase"] = "scripted-replay"
    result = audit(rows + [replay])
    assert result["excluded_receipts"] == [2]
    assert result["duplicate_receipts"] == 0


def test_claude_ignores_tool_and_thinking_blocks_but_keeps_identical_distinct_texts() -> None:
    records = claude_records()
    records.extend(
        [
            {
                "type": "assistant",
                "sessionId": "s1",
                "uuid": "a2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private"},
                        {"type": "tool_use", "name": "tool"},
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "s1",
                "uuid": "u2",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "result"}],
                },
            },
            {
                "type": "assistant",
                "sessionId": "s1",
                "uuid": "a3",
                "message": {"role": "assistant", "content": "Stored."},
            },
        ]
    )
    assert len(expected_messages("claude-code", "teach", records)) == 3
    del records[-1]["uuid"]
    with pytest.raises(ValueError, match="identity"):
        expected_messages("claude-code", "teach", records)


def test_codex_binds_actual_user_and_final_to_turn_not_session_or_injected_context() -> None:
    records = [
        {"type": "session_meta", "payload": {"id": "session"}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "injected instructions"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "user_message", "message": "User prompt"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Answer"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "turn"}},
    ]
    result = expected_messages("codex", "teach", records)
    assert [row["source_ref"] for row in result] == ["UserPromptSubmit:turn", "Stop:turn"]
    assert {row["session_id"] for row in result} == {"session"}
    records.pop()
    with pytest.raises(ValueError, match="complete"):
        expected_messages("codex", "teach", records)


def test_codex_intermediate_text_is_exposed_as_unsupported_not_silently_omitted() -> None:
    records = [
        {"type": "session_meta", "payload": {"id": "s"}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "Do work"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Working"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t"}},
    ]
    result = expected_messages("codex", "teach", records)
    assert len(result) == 3
    assert result[1]["capture_supported"] is False


def test_hermes_snapshot_repeats_deduplicate_by_position_not_content() -> None:
    messages = [
        {"role": "user", "content": "same", "_row_id": 1},
        {"role": "assistant", "content": "", "tool_calls": [{}]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "same", "_row_id": 4},
    ]
    record = {"operation": "transcript", "session_id": "s", "messages": messages}
    result = expected_messages("hermes", "teach", [record, deepcopy(record)])
    assert [row["source_ref"] for row in result] == ["transcript:v2:1", "transcript:v2:4"]
    changed = deepcopy(record)
    changed["messages"][0]["content"] = "different"
    with pytest.raises(ValueError, match="conflicting"):
        expected_messages("hermes", "teach", [record, changed])


def test_empty_expectations_cannot_pass() -> None:
    with pytest.raises(ValueError, match="empty"):
        reconcile_captures([], receipts(), scope=SCOPE)


def test_suppression_receipt_uses_verified_payload_identity_without_claiming_storage() -> None:
    rows = receipts()
    rows[0]["payload"]["identity_status"] = "normal"
    rows[0]["result"] = {
        "status": "committed",
        "event_id": DURABLE_IDS["u1"],
        "processing_status": "suppressed",
        "job_id": None,
    }
    result = audit(rows)
    assert result["covered_messages"] == 2
    assert result["messages"][0]["suppressed"] is True
    assert result["messages"][0]["job_statuses"] == ["not_present"]
    del rows[0]["payload"]["identity_status"]
    assert audit(rows)["covered_messages"] == 1


def test_forged_source_event_id_cannot_cover_native_message() -> None:
    rows = receipts()
    rows[0]["payload"]["event_id"] = "forged"
    assert audit(rows)["covered_messages"] == 1


def test_hermes_uses_native_row_id_and_rejects_missing_identity() -> None:
    row = {
        "operation": "transcript",
        "session_id": "s",
        "messages": [{"role": "user", "content": "Hello", "_row_id": 21}],
    }
    assert expected_messages("hermes", "teach", [row])[0]["source_ref"] == "transcript:v2:21"
    del row["messages"][0]["_row_id"]
    with pytest.raises(ValueError, match="identity"):
        expected_messages("hermes", "teach", [row])


def test_hermes_partial_snapshot_is_not_filled_from_other_snapshot() -> None:
    full = {
        "operation": "transcript",
        "session_id": "s",
        "messages": [
            {"role": "user", "content": "Hello", "_row_id": 1},
            {"role": "assistant", "content": "Hi", "_row_id": 2},
        ],
    }
    partial = deepcopy(full)
    partial["messages"].pop()
    with pytest.raises(ValueError, match="conflicting"):
        expected_messages("hermes", "teach", [full, partial])


def test_hermes_durable_row_hash_is_content_independent_but_coverage_is_not() -> None:
    scope = {
        "user_profile_id": "native-three-host-v7",
        "project_id": "2124842e-69de-4513-afc7-6c1764df75c2",
        "workspace_id": None,
    }
    session = "20260920_082819_6e2df1"
    expected = expected_messages(
        "hermes",
        "recall",
        [
            {
                "operation": "transcript",
                "session_id": session,
                "messages": [{"role": "user", "content": "Original text", "_row_id": 1}],
            }
        ],
    )
    receipt = {
        "phase": "recall",
        "operation": "capture",
        "payload": {
            "context": {**scope, "host": "hermes", "session_id": session},
            "role": "user",
            "source_kind": "hermes_memory_provider",
            "source_ref": "transcript:v2:1",
            "content": "Original text",
            "event_id": "hermes:d662756c7c3826763f574b4c30c73c4816cdad7f97ec0cc2533f056e4ad4b2e3",
        },
        "result": {
            "status": "committed",
            "event_id": "7fab853e12b1b9266a65e7bfc7a0d5424dcdb642c7939ff4e7c6cd701b6dacee",
            "identity_status": "normal",
            "processing_status": "pending",
        },
    }
    assert reconcile_captures(expected, [receipt], scope=scope)["covered_messages"] == 1
    receipt["payload"]["content"] = "Changed text"
    assert reconcile_captures(expected, [receipt], scope=scope)["covered_messages"] == 0
