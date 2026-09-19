"""Native process evidence must be valid independently of a zero exit code."""

import json

import pytest

from benchmarks.native_hosts.three_host_run import windows_trace


def test_codex_direct_model_call_links_to_native_result_and_final_answer(tmp_path):
    sessions = tmp_path / "home/sessions"
    sessions.mkdir(parents=True)
    encoded = json.dumps({"schema_version": 1, "facts": []})
    rows = [
        {"type": "session_meta", "payload": {"id": "session"}},
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "mcp__mnemos__mnemos_retrieve",
                "call_id": "model-1",
                "arguments": '{"query":"x"}',
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "mcp_tool_call_end",
                "call_id": "model-1",
                "invocation": {
                    "server": "mnemos",
                    "tool": "mnemos_retrieve",
                    "arguments": {"query": "x"},
                },
                "result": {
                    "Ok": {
                        "isError": False,
                        "structuredContent": {"result": encoded},
                        "content": [{"type": "text", "text": encoded}],
                    }
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "final_answer",
                "message": "No current fact is known.",
            },
        },
    ]
    (sessions / "rollout-test.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    process = {
        "host": "codex",
        "status": "completed",
        "auth_copy_removed": True,
        "pid": 99,
        "home": str(sessions.parent),
    }
    evidence = windows_trace(process, tmp_path)
    assert evidence["calls"][0]["model_call_correlated"] is True
    assert evidence["provenance"]["session_id"] == "session"
    assert evidence["final_answer"] == "No current fact is known."
    rows.pop()
    (sessions / "rollout-test.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="final answer"):
        windows_trace(process, tmp_path)


@pytest.mark.parametrize("fault", ["another-session", "terminal-error", "no-terminal"])
def test_claude_zero_exit_cannot_override_stream_failure(tmp_path, fault):
    terminal = {
        "type": "result",
        "session_id": "session",
        "is_error": False,
        "subtype": "success",
        "result": "Done",
    }
    if fault == "another-session":
        terminal["session_id"] = "different"
    elif fault == "terminal-error":
        terminal["is_error"] = True
    rows = [] if fault == "no-terminal" else [terminal]
    (tmp_path / "stdout.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    process = {
        "host": "claude-code",
        "status": "completed",
        "auth_copy_removed": True,
        "pid": 99,
        "session_id": "session",
    }
    with pytest.raises(ValueError):
        windows_trace(process, tmp_path)
