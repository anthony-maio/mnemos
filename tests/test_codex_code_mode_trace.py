"""Synthetic trace integrity tests. These do not establish native acceptance."""

import copy
import hashlib
import json

import pytest

from benchmarks.native_hosts import shared_trace as trace


def fixture():
    args = {"query": "release label", "scope": "project"}
    answer = {"status": "ok", "facts": [{"id": "fixture-id"}]}
    ok = {
        "content": [{"type": "text", "text": json.dumps(answer)}],
        "structuredContent": {"result": json.dumps(answer)},
        "isError": False,
    }
    model_call = {
        "type": "custom_tool_call",
        "call_id": "model-1",
        "name": "exec",
        "input": "const answer = await tools.example(); text(answer);",
    }
    invocation = {"server": "mnemos", "tool": "mnemos_retrieve", "arguments": args}
    runtime = {"call_id": "tool-1", "turn_id": "turn-1", "invocation": invocation}
    payloads = {}

    def ref(value, kind):
        n = str(len(payloads) + 1)
        payloads[n] = value
        return {
            "raw_payload_id": "raw_payload:" + n,
            "kind": {"type": kind},
            "path": "payloads/" + n + ".json",
        }

    events = [
        {"type": "rollout_started", "trace_id": "trace-1", "root_thread_id": "session-1"},
        {"type": "thread_started", "thread_id": "session-1"},
        {"type": "codex_turn_started", "thread_id": "session-1", "codex_turn_id": "turn-1"},
        {
            "type": "inference_started",
            "inference_call_id": "inference-1",
            "model": "pinned",
            "request_payload": ref({"model": "pinned"}, "inference_request"),
        },
        {
            "type": "code_cell_started",
            "runtime_cell_id": "cell-1",
            "model_visible_call_id": "model-1",
            "source_js": model_call["input"],
        },
        {
            "type": "tool_call_started",
            "tool_call_id": "tool-1",
            "model_visible_call_id": None,
            "code_mode_runtime_tool_id": "local-tool-1",
            "requester": {"type": "code_cell", "runtime_cell_id": "cell-1"},
            "invocation_payload": ref(
                {
                    "tool_name": "mnemos_retrieve",
                    "tool_namespace": "mcp__mnemos",
                    "payload": {"type": "function", "arguments": json.dumps(args)},
                },
                "tool_invocation",
            ),
        },
        {
            "type": "tool_call_runtime_started",
            "tool_call_id": "tool-1",
            "runtime_payload": ref(copy.deepcopy(runtime), "tool_runtime_event"),
        },
        # Streaming may finish inference after the cell/tool has already started.
        {
            "type": "inference_completed",
            "inference_call_id": "inference-1",
            "response_id": "resp-1",
            "response_payload": ref(
                {"response_id": "resp-1", "output_items": [model_call]}, "inference_response"
            ),
        },
        {
            "type": "tool_call_runtime_ended",
            "tool_call_id": "tool-1",
            "status": "completed",
            "runtime_payload": ref({**runtime, "result": {"Ok": ok}}, "tool_runtime_event"),
        },
        {
            "type": "tool_call_ended",
            "tool_call_id": "tool-1",
            "status": "completed",
            "result_payload": ref(
                {
                    "type": "code_mode_response",
                    "value": {"content": ok["content"], "isError": False},
                },
                "tool_result",
            ),
        },
        {"type": "code_cell_ended", "runtime_cell_id": "cell-1", "status": "completed"},
        {"type": "codex_turn_ended", "codex_turn_id": "turn-1", "status": "completed"},
        {"type": "thread_ended", "thread_id": "session-1", "status": "completed"},
        {"type": "rollout_ended", "status": "completed"},
    ]
    session = [
        {"type": "session_meta", "payload": {"id": "session-1"}},
        {"type": "response_item", "payload": copy.deepcopy(model_call)},
    ]
    return events, payloads, session


def parse(tmp_path, events, payloads, session, *, bad_path=False):
    root = tmp_path / "trace"
    (root / "payloads").mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "trace_id": "trace-1",
        "rollout_id": "session-1",
        "root_thread_id": "session-1",
        "raw_event_log": "trace.jsonl",
        "payloads_dir": "payloads",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    records = [
        {
            "schema_version": 1,
            "seq": i,
            "rollout_id": "session-1",
            "thread_id": "session-1",
            "codex_turn_id": "turn-1",
            "payload": e,
        }
        for i, e in enumerate(events, 1)
    ]
    (root / "trace.jsonl").write_text("\n".join(map(json.dumps, records)))
    for n, value in payloads.items():
        (root / "payloads" / (n + ".json")).write_text(json.dumps(value))
    source = tmp_path / "session.jsonl"
    source.write_text("\n".join(map(json.dumps, session)))
    provenance = trace.TraceProvenance(
        "codex", 42, "session-1", str(source), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    return trace.parse_codex_code_mode_trace(root, provenance=provenance, model="pinned")


def test_native_parent_ids_bind_streamed_model_cell_and_mcp_result(tmp_path):
    calls, hashes = parse(tmp_path, *fixture())
    assert len(calls) == 1
    assert calls[0].model_call_correlated and calls[0].provider_call_id == "model-1"
    assert calls[0].name == "mnemos_retrieve" and calls[0].result["facts"][0]["id"] == "fixture-id"
    assert "manifest.json" in hashes and "payloads/4.json" in hashes


@pytest.mark.parametrize(
    "mutation",
    [
        "parent",
        "model",
        "model-call",
        "source",
        "other-server",
        "args",
        "failed",
        "ambiguous-result",
        "result-content",
        "tool-result",
        "missing-end",
        "duplicate",
        "cross-turn",
        "payload-path",
        "payload-kind",
        "unknown-event",
        "extra-model-call",
        "missing-inference",
        "missing-cell-end",
        "wrong-response",
        "runtime-id",
        "extra-tool",
        "finished-before-start",
        "same-source-wrong-id",
        "wrong-session",
    ],
)
def test_broken_native_chain_cannot_be_accepted(tmp_path, mutation):
    events, payloads, session = fixture()
    if mutation == "parent":
        events[5]["requester"]["runtime_cell_id"] = "other"
    elif mutation == "model":
        events[3]["model"] = "other"
    elif mutation == "model-call":
        events[4]["model_visible_call_id"] = "other"
    elif mutation == "source":
        events[4]["source_js"] = "different"
    elif mutation == "other-server":
        payloads["3"]["invocation"]["server"] = "other"
    elif mutation == "args":
        payloads["3"]["invocation"]["arguments"] = {}
    elif mutation == "failed":
        events[8]["status"] = "failed"
    elif mutation == "ambiguous-result":
        payloads["5"]["result"]["Err"] = "blocked"
    elif mutation == "result-content":
        payloads["5"]["result"]["Ok"]["content"][0]["text"] = "{}"
    elif mutation == "tool-result":
        payloads["6"]["value"]["isError"] = True
    elif mutation == "missing-end":
        events.pop(9)
    elif mutation == "duplicate":
        events.insert(6, copy.deepcopy(events[5]))
    elif mutation == "cross-turn":
        payloads["3"]["turn_id"] = "other"
    elif mutation == "payload-path":
        events[5]["invocation_payload"]["path"] = "../session.jsonl"
    elif mutation == "payload-kind":
        events[5]["invocation_payload"]["kind"]["type"] = "inference_response"
    elif mutation == "unknown-event":
        events.insert(9, {"type": "unrecognized_tool_event"})
    elif mutation == "extra-model-call":
        session.append(
            {
                "type": "response_item",
                "payload": {**session[1]["payload"], "call_id": "unaccounted"},
            }
        )
    elif mutation == "missing-inference":
        events.pop(7)
    elif mutation == "missing-cell-end":
        events.pop(10)
    elif mutation == "wrong-response":
        payloads["4"]["response_id"] = "other"
    elif mutation == "runtime-id":
        payloads["3"]["call_id"] = "other"
    elif mutation == "extra-tool":
        events.insert(9, {**events[5], "tool_call_id": "unaccounted"})
    elif mutation == "finished-before-start":
        events[5], events[9] = events[9], events[5]
    elif mutation == "same-source-wrong-id":
        session[1]["payload"]["call_id"] = "other"
    elif mutation == "wrong-session":
        session[0]["payload"]["id"] = "other"
    with pytest.raises(ValueError):
        parse(tmp_path, events, payloads, session)
