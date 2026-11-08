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
    events.insert(
        7,
        {
            "type": "mcp_tool_call_correlation_assigned",
            "tool_call_id": "tool-1",
            "mcp_call_id": "native-mcp-1",
        },
    )
    session = [
        {"type": "session_meta", "payload": {"id": "session-1"}},
        {"type": "response_item", "payload": copy.deepcopy(model_call)},
    ]
    return events, payloads, session


def parse(tmp_path, events, payloads, session, *, bad_path=False, observe=False):
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
    reader = trace.observe_codex_code_mode_trace if observe else trace.parse_codex_code_mode_trace
    return reader(root, provenance=provenance, model="pinned")


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
        "missing-correlation",
        "unbound-correlation",
        "duplicate-correlation",
        "late-correlation",
    ],
)
@pytest.mark.parametrize("observe", [False, True])
def test_broken_native_chain_cannot_be_accepted(tmp_path, mutation, observe):
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
        events[9]["status"] = "failed"
    elif mutation == "ambiguous-result":
        payloads["5"]["result"]["Err"] = "blocked"
    elif mutation == "result-content":
        payloads["5"]["result"]["Ok"]["content"][0]["text"] = "{}"
    elif mutation == "tool-result":
        payloads["6"]["value"]["isError"] = True
    elif mutation == "missing-end":
        events.pop(10)
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
        events.pop(8)
    elif mutation == "missing-cell-end":
        events.pop(11)
    elif mutation == "wrong-response":
        payloads["4"]["response_id"] = "other"
    elif mutation == "runtime-id":
        payloads["3"]["call_id"] = "other"
    elif mutation == "extra-tool":
        events.insert(9, {**events[5], "tool_call_id": "unaccounted"})
    elif mutation == "finished-before-start":
        events[5], events[10] = events[10], events[5]
    elif mutation == "same-source-wrong-id":
        session[1]["payload"]["call_id"] = "other"
    elif mutation == "wrong-session":
        session[0]["payload"]["id"] = "other"
    elif mutation == "missing-correlation":
        events.pop(7)
    elif mutation == "unbound-correlation":
        events[7]["tool_call_id"] = "other"
    elif mutation == "duplicate-correlation":
        events.insert(8, copy.deepcopy(events[7]))
    elif mutation == "late-correlation":
        events.insert(11, events.pop(7))
    with pytest.raises(ValueError):
        parse(tmp_path, events, payloads, session, observe=observe)


def discovery_fixture(tool):
    events, payloads, session = fixture()
    events.pop(7)  # Built-in calls have no MCP dispatch correlation.
    payloads["2"].update(tool_name=tool, tool_namespace=None)
    for key in ["3", "5"]:
        payloads[key]["invocation"].update(server="codex", tool=tool)
    answer = {"resources" if tool == "list_mcp_resources" else "resourceTemplates": []}
    content = json.dumps(answer)
    payloads["5"]["result"] = {
        "Ok": {"content": [{"type": "text", "text": content}], "isError": False}
    }
    payloads["6"]["value"] = content
    return events, payloads, session


@pytest.mark.parametrize("tool", ["list_mcp_resources", "list_mcp_resource_templates"])
def test_discovery_is_observed_with_model_parent_but_never_invented_mcp_id(tmp_path, tool):
    calls, hashes = parse(tmp_path / "observation", *discovery_fixture(tool), observe=True)
    assert len(calls) == 1 and calls[0].name == tool
    assert calls[0].dispatch_kind == "codex_discovery" and calls[0].outcome == "success"
    assert calls[0].model_call_correlated and calls[0].provider_call_id == "model-1"
    assert len(hashes) == 8
    with pytest.raises(ValueError, match="MCP dispatch correlation is incomplete"):
        parse(tmp_path / "strict", *discovery_fixture(tool))


@pytest.mark.parametrize(
    "mutation",
    [
        "namespace",
        "server",
        "arguments",
        "runtime-id",
        "tool",
        "unknown-helper",
        "correlation",
        "content",
        "error",
        "transport",
        "parent",
    ],
)
def test_discovery_observation_rejects_tampered_or_unsupported_calls(tmp_path, mutation):
    events, payloads, session = discovery_fixture("list_mcp_resources")
    if mutation == "namespace":
        payloads["2"]["tool_namespace"] = "mcp__other"
    elif mutation == "server":
        payloads["5"]["invocation"]["server"] = "mnemos"
    elif mutation == "arguments":
        payloads["5"]["invocation"]["arguments"] = {"server": "other"}
    elif mutation == "runtime-id":
        payloads["5"]["call_id"] = "other"
    elif mutation == "tool":
        payloads["3"]["invocation"]["tool"] = "list_mcp_resource_templates"
    elif mutation == "unknown-helper":
        payloads["2"]["tool_name"] = "read_mcp_resource"
    elif mutation == "correlation":
        events.insert(
            7,
            {
                "type": "mcp_tool_call_correlation_assigned",
                "tool_call_id": "tool-1",
                "mcp_call_id": "fake",
            },
        )
    elif mutation == "content":
        payloads["6"]["value"] = "{}"
    elif mutation == "error":
        payloads["5"]["result"]["Ok"]["isError"] = True
    elif mutation == "transport":
        payloads["5"]["result"] = {"Err": "unavailable"}
    elif mutation == "parent":
        events[5]["requester"]["runtime_cell_id"] = "other"
    with pytest.raises(ValueError):
        parse(tmp_path, events, payloads, session, observe=True)


@pytest.mark.parametrize(
    "answer",
    [
        {
            "status": "degraded",
            "committed": False,
            "error": "Mnemos service returned HTTP 404 (fact_not_found)",
        },
        {"status": "rejected", "committed": False},
        {"error": "connection unavailable"},
        {"status": "ok", "isError": True},
    ],
)
def test_application_failure_is_retained_as_unsuccessful_not_accepted(tmp_path, answer):
    events, payloads, session = fixture()
    ok = payloads["5"]["result"]["Ok"]
    ok["content"][0]["text"] = json.dumps(answer)
    ok["structuredContent"]["result"] = json.dumps(answer)
    # fixture shares the original content list with the visible response.
    calls, _ = parse(tmp_path / "observation", events, payloads, session, observe=True)
    assert len(calls) == 1 and dict(calls[0].result) == answer
    assert calls[0].outcome == "unsuccessful" and calls[0].dispatch_kind == "mnemos_mcp"
    with pytest.raises(ValueError, match="unsuccessful"):
        trace.require_tool_sequence(calls, ["mnemos_retrieve"])
    with pytest.raises(ValueError):
        parse(tmp_path / "strict", events, payloads, session)
