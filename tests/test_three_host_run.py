"""Native process evidence must be valid independently of a zero exit code."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.native_hosts.three_host_run import windows_trace
from benchmarks.native_hosts import three_host_run as runner


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


@pytest.mark.parametrize(
    "unexpected_operation", [None, "model_request", "bridge_response", "provider_error"]
)
def test_hermes_preflight_requires_discovery_without_inference_or_writes(
    tmp_path, monkeypatch, unexpected_operation
):
    from mnemos.adapters.hermes_tools import get_tool_schemas

    runtime = SimpleNamespace(
        root=tmp_path,
        project=tmp_path / "project",
        descriptor=tmp_path / "runtime.json",
        registry_path=tmp_path / "scopes.json",
        profile="fixture",
    )
    runtime.project.mkdir()
    calls = []

    def api(route, payload):
        calls.append(route)
        assert route == "show"
        return {
            "parameters": "temperature 0\npresence_penalty 0\nseed 42\nnum_ctx 65536\ntop_k 20\ntop_p 0.95"
        }

    class Process:
        pid = 123
        returncode = 0

        def __init__(self, command, **kwargs):
            assert "60s" in command
            assert command[-2].endswith("hermes_native_bootstrap_v6.py.txt")
            root = tmp_path / "preflight"
            rows = [
                {
                    "operation": "preflight_complete",
                    "tool_names": [row["name"] for row in get_tool_schemas()],
                }
            ]
            if unexpected_operation:
                rows.append({"operation": unexpected_operation})
            (root / "trace.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

        def wait(self, timeout):
            assert timeout == 80

    monkeypatch.setattr(runner, "api", api)
    monkeypatch.setattr(runner, "host_environment", lambda path: {})
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "terminate_owned_process", lambda process: None)
    bindings = {"wheel": str(tmp_path / "fixture.whl"), "wheel_sha256": "f" * 64}
    if unexpected_operation:
        with pytest.raises(ValueError, match="import-only"):
            runner.run_hermes(runtime, bindings, "preflight", "", mode="preflight")
    else:
        result = runner.run_hermes(runtime, bindings, "preflight", "", mode="preflight")
        assert result["status"] == "preflight_passed"
        assert len(result["tool_names"]) == 8
    assert calls == ["show", "show"]
    assert not runtime.descriptor.exists()


@pytest.mark.parametrize(
    "fault", ["live-process", "auth-left", "cleanup-error", "hermes-uncertain"]
)
def test_parent_rejects_uncertain_native_cleanup(tmp_path, monkeypatch, fault):
    directory = tmp_path / "phase"
    directory.mkdir()
    row = {"host": "codex", "pid": 123, "auth_copy_removed": True, "cleanup_errors": []}
    if fault == "auth-left":
        (directory / "auth.json").write_text("synthetic-placeholder")
    elif fault == "cleanup-error":
        row["cleanup_errors"] = ["process_termination:OSError"]
    elif fault == "hermes-uncertain":
        row = {"host": "hermes", "launcher_pid": 123, "cleanup_established": False}
    (directory / "process.json").write_text(json.dumps(row))
    monkeypatch.setattr(runner, "process_exists", lambda pid: fault == "live-process")
    with pytest.raises(ValueError):
        runner.require_phase_cleanup(tmp_path)


def test_parent_accepts_terminated_phase_and_removed_auth(tmp_path, monkeypatch):
    directory = tmp_path / "phase"
    directory.mkdir()
    (directory / "process.json").write_text(
        json.dumps(
            {"host": "claude-code", "pid": 123, "auth_copy_removed": True, "cleanup_errors": []}
        )
    )
    monkeypatch.setattr(runner, "process_exists", lambda pid: False)
    runner.require_phase_cleanup(tmp_path)
