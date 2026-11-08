import copy
import json
from pathlib import Path
import runpy
import io
import threading
import time
import sqlite3

import pytest

HELPERS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/native_hosts/registry_claude_session.py")
)
PARSE = HELPERS["parse_calls"]


def records():
    return [
        dict(
            type="assistant",
            message=dict(
                role="assistant",
                id="response-1",
                content=[
                    dict(
                        type="tool_use",
                        id="call-1",
                        name="mcp__mnemos__mnemos_retrieve",
                        input={"query": "label"},
                    )
                ],
            ),
        ),
        dict(
            type="user",
            message=dict(
                role="user",
                content=[
                    dict(
                        type="tool_result",
                        tool_use_id="call-1",
                        content=json.dumps(
                            {
                                "result": json.dumps(
                                    {
                                        "status": "degraded",
                                        "committed": False,
                                        "error": "scope_unavailable",
                                    }
                                )
                            }
                        ),
                    )
                ],
            ),
        ),
    ]


def test_structured_denial_is_correlated_without_becoming_success():
    calls = PARSE(records())
    assert len(calls) == 1 and calls[0]["provider_call_id"] == "call-1"
    assert calls[0]["result"]["status"] == "degraded" and not calls[0]["result"]["committed"]


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "wrong_namespace", "orphan", "wrong_role"]
)
def test_uncorrelated_or_wrong_namespace_records_fail(mutation):
    rows = records()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.insert(1, copy.deepcopy(rows[0]))
    elif mutation == "wrong_namespace":
        rows[0]["message"]["content"][0]["name"] = "mcp__other__mnemos_retrieve"
    elif mutation == "orphan":
        rows[1]["message"]["content"][0]["tool_use_id"] = "other"
    else:
        rows[1]["message"]["role"] = "assistant"
    with pytest.raises((AssertionError, KeyError)):
        PARSE(rows)


def test_assistant_prose_is_not_tool_evidence():
    assert (
        PARSE(
            [
                dict(
                    type="assistant",
                    message=dict(
                        role="assistant",
                        content=[dict(type="text", text="I called mnemos_store successfully.")],
                    ),
                )
            ]
        )
        == []
    )


def test_overlapping_calls_fail_even_when_results_arrive_in_expected_order():
    first = records()
    second = copy.deepcopy(first)
    second[0]["message"]["content"][0]["id"] = "call-2"
    second[1]["message"]["content"][0]["tool_use_id"] = "call-2"
    with pytest.raises(AssertionError, match="sequential"):
        PARSE([first[0], second[0], first[1], second[1]])
    assert len(PARSE(first + second)) == 2


def test_wrapper_cannot_hide_extra_fields():
    rows = records()
    node = rows[1]["message"]["content"][0]
    wrapped = json.loads(node["content"])
    wrapped["error"] = "unexpected failure"
    node["content"] = json.dumps(wrapped)
    with pytest.raises(AssertionError, match="wrapper"):
        PARSE(rows)


def test_expired_deadline_fails_before_writing():
    stream = io.BytesIO()
    with pytest.raises(TimeoutError):
        HELPERS["send_frame"](stream, {"type": "user"}, time.monotonic() - 1)
    assert stream.getvalue() == b""


def test_blocked_pipe_write_is_bounded():
    release = threading.Event()
    finished = threading.Event()

    class BlockedPipe:
        def write(self, data):
            release.wait()

        def flush(self):
            finished.set()

    try:
        with pytest.raises(TimeoutError):
            HELPERS["send_frame"](BlockedPipe(), {}, time.monotonic() + 0.05)
    finally:
        release.set()
        assert finished.wait(2)


def test_pipe_error_is_propagated():
    class BrokenPipe:
        def write(self, data):
            raise BrokenPipeError("closed")

    with pytest.raises(BrokenPipeError, match="closed"):
        HELPERS["send_frame"](BrokenPipe(), {}, time.monotonic() + 2)


@pytest.mark.parametrize("wrong", [None, "fact_scope", "event_scope", "profile", "count"])
def test_durable_scope_proof_rejects_wrong_partition(tmp_path, wrong):
    path = tmp_path / "memory.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE memory_chunks(metadata TEXT, user_profile_id TEXT)")
        db.execute("CREATE TABLE episodic_events(scope TEXT, scope_id TEXT, user_profile_id TEXT)")
        db.execute(
            "INSERT INTO memory_chunks VALUES (?, ?)",
            (
                json.dumps(
                    dict(
                        scope="global" if wrong == "fact_scope" else "project",
                        scope_id="project-id",
                    )
                ),
                "other" if wrong == "profile" else "registry-session",
            ),
        )
        db.execute(
            "INSERT INTO episodic_events VALUES (?, ?, ?)",
            (
                "project",
                "other" if wrong == "event_scope" else "project-id",
                "registry-session",
            ),
        )
    if wrong is None:
        HELPERS["require_durable_project"](path, "project-id", 1)
    else:
        with pytest.raises(AssertionError):
            HELPERS["require_durable_project"](path, "project-id", 2 if wrong == "count" else 1)
