"""Offline failure evidence and raw-request preservation in the development runner."""

import argparse
import asyncio
from copy import deepcopy
import json

from benchmarks.native_hosts import model_forget_development_v1 as runner


def test_failed_model_request_remains_recorded_with_cleanup(tmp_path, monkeypatch):
    async def fail(*args):
        raise TimeoutError("synthetic transport timeout")

    monkeypatch.setattr(runner, "_bounded_api", fail)
    result = asyncio.run(
        runner._run_case(tmp_path, "unused", "qwen3.5:9b", False, runner.CASES[0], 0)
    )
    assert result["status"] == "failed"
    assert len(result["requests"]) == 1
    assert result["requests"][0]["error_type"] == "TimeoutError"
    assert result["responses"] == []
    assert result["cleanup"]["ok"] is True
    assert result["state_before"] == result["state_after_cleanup"]
    saved = json.loads((tmp_path / "case-1-backup-window/report.json").read_text())
    assert saved["status"] == "failed"
    assert saved["finished_at"]


def test_raw_requests_do_not_gain_later_messages_or_synthetic_ids(tmp_path, monkeypatch):
    responses = [
        {
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "mnemos_retrieve",
                            "arguments": {"query": "backup window"},
                        }
                    }
                ],
            },
        },
        {
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "I did not delete it."},
        },
    ]
    original = deepcopy(responses)

    async def respond(*args):
        return responses.pop(0)

    monkeypatch.setattr(runner, "_bounded_api", respond)
    result = asyncio.run(
        runner._run_case(tmp_path, "unused", "qwen3.5:9b", False, runner.CASES[0], 0)
    )
    assert result["status"] == "failed"
    assert result["responses"] == original
    assert len(result["requests"][0]["payload"]["messages"]) == 2
    assert len(result["requests"][1]["payload"]["messages"]) == 4
    assert result["trace"][0]["tool_calls"][0]["id"] == "response-0-call-0"
    assert result["trace"][1]["call_id"] == "response-0-call-0"
    assert result["requests"][1]["payload"]["messages"][-1]["tool_name"] == "mnemos_retrieve"
    assert result["cleanup"]["ok"] is True


def test_rejected_dispatch_fails_without_retry(tmp_path, monkeypatch):
    requests = []

    async def respond(*args):
        requests.append(args)
        return {
            "done": True,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "mnemos_inspect", "arguments": {"fact_id": "missing"}}}
                ],
            },
        }

    monkeypatch.setattr(runner, "_bounded_api", respond)
    result = asyncio.run(
        runner._run_case(tmp_path, "unused", "qwen3.5:9b", False, runner.CASES[0], 0)
    )
    assert result["status"] == "failed"
    assert len(requests) == 1
    assert len(result["tool_results"]) == 1
    assert result["cleanup"]["ok"] is True


def test_invalid_wheel_archives_preflight_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        output=tmp_path / "output", model="qwen3.5:9b", wheel=tmp_path / "missing.whl"
    )
    assert asyncio.run(runner._main(args)) == 1
    saved = json.loads((args.output / "report.json").read_text())
    assert saved["status"] == "failed"
    assert saved["error_type"] == "ValueError"
    assert saved["cases"] == []
    assert saved["finished_at"]
