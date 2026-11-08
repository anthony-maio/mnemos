"""Diagnostic build preserves dispatch semantics and emits only fixed stage names."""

import json
import os
from pathlib import Path

import pytest

from tests import test_native_hook_launcher as launcher
from tests.test_native_hook_launcher import isolated  # noqa: F401

BINARY = os.environ.get("MNEMOS_NATIVE_HOOK_TIMING_EXE")
pytestmark = pytest.mark.skipif(not BINARY, reason="Diagnostic native build not configured")
STAGES = {
    "main_begin",
    "launch_prepared",
    "stdin_read",
    "payload_parsed",
    "request_begin",
    "descriptor_validated",
    "request_serialized",
    "connected",
    "request_written",
    "response_first_byte",
    "response_headers_read",
    "response_read",
    "response_validated",
    "fallback_begin",
    "run_finished",
}


def trace(result):
    records = [json.loads(line) for line in result.stderr.splitlines() if line.startswith(b"{")]
    assert len(records) == 1
    record = records[0]
    assert set(record) == {"diagnostic", "unit", "marks"}
    assert record["diagnostic"] == "native_stage_timing_v1"
    assert record["unit"] == "milliseconds_since_main"
    names, times = zip(*record["marks"])
    assert set(names) <= STAGES and len(names) == len(set(names))
    assert names[0] == "main_begin" and names[-1] == "run_finished"
    assert list(times) == sorted(times) and all(value >= 0 for value in times)
    assert b"private-synthetic-token" not in result.stderr
    assert b"sensitive-source-marker" not in result.stderr
    return list(names)


@pytest.mark.parametrize("host", ["codex", "claude-code"])
def test_handled_timing_preserves_output_and_single_dispatch(isolated, monkeypatch, host):
    monkeypatch.setattr(launcher, "BINARY", BINARY)
    root, env = isolated
    payload = {"content": "sensitive-source-marker", "number": 2**90 + 1}
    with launcher.server(lambda r: launcher.http(launcher.reply(r))) as (port, requests):
        launcher.save_descriptor(env, port, host=host)
        result = launcher.invoke(root, env, payload, host=host)
    assert result.returncode == 0 and json.loads(result.stdout) == {"continue": True}
    assert len(requests) == 1 and requests[0][0]["payload"]["payload"] == payload
    assert trace(result) == [
        "main_begin",
        "launch_prepared",
        "stdin_read",
        "payload_parsed",
        "request_begin",
        "descriptor_validated",
        "request_serialized",
        "connected",
        "request_written",
        "response_first_byte",
        "response_headers_read",
        "response_read",
        "response_validated",
        "run_finished",
    ]
    assert not Path(env["FALLBACK_MARKER"]).exists()


def test_lost_response_does_not_dispatch_again(isolated, monkeypatch):
    monkeypatch.setattr(launcher, "BINARY", BINARY)
    root, env = isolated
    with launcher.server(lambda r: None) as (port, requests):
        launcher.save_descriptor(env, port)
        result = launcher.invoke(root, env, {"content": "sensitive-source-marker"})
    assert result.returncode == 0 and json.loads(result.stdout) == {"continue": True}
    assert len(requests) == 1 and not Path(env["FALLBACK_MARKER"]).exists()
    names = trace(result)
    assert "request_written" in names and "fallback_begin" not in names
    assert "response_validated" not in names


def test_missing_worker_preserves_fallback_input(isolated, monkeypatch):
    monkeypatch.setattr(launcher, "BINARY", BINARY)
    root, env = isolated
    raw = b'{"content":"sensitive-source-marker"}'
    result = launcher.invoke(root, env, raw)
    assert result.returncode == 0 and json.loads(result.stdout)["fallback"] is True
    assert Path(env["FALLBACK_MARKER"]).read_bytes() == raw
    assert "fallback_begin" in trace(result)


def test_invalid_input_never_exposes_payload_or_dispatches(isolated, monkeypatch):
    monkeypatch.setattr(launcher, "BINARY", BINARY)
    root, env = isolated
    result = launcher.invoke(root, env, b"sensitive-source-marker")
    assert result.returncode == 0 and json.loads(result.stdout) == {"continue": True}
    assert not Path(env["FALLBACK_MARKER"]).exists()
    assert trace(result) == ["main_begin", "launch_prepared", "stdin_read", "run_finished"]
