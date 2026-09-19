import hashlib
import json
from dataclasses import replace

import pytest

from benchmarks.native_hosts.shared_trace import (
    TraceProvenance,
    parse_claude_tool_records,
    parse_codex_mcp_records,
    require_tool_sequence,
    load_source_records,
)

PROVENANCE = TraceProvenance(
    host="codex",
    process_id=1234,
    session_id="session-a",
    source_file="sessions/a.jsonl",
    source_file_sha256="f" * 64,
)


def _codex_event(kind, call_id, arguments, result=None, line=1):
    invocation = {
        "server": "mnemos",
        "tool": "mnemos_retrieve",
        "arguments": arguments,
    }
    payload = {"type": f"mcp_tool_call_{kind}", "call_id": call_id, "invocation": invocation}
    if result is not None:
        encoded = json.dumps(result, sort_keys=True)
        payload["result"] = {
            "Ok": {
                "isError": False,
                "structuredContent": {"result": encoded},
                "content": [{"text": encoded}],
            }
        }
    return {"line": line, "line_sha256": f"{line:064x}", "payload": payload}


def _claude_line(nodes, line):
    role = "assistant" if any(node.get("type") == "tool_use" for node in nodes) else "user"
    return {
        "line": line,
        "line_sha256": f"{line:064x}",
        "type": role,
        "message": {"id": f"response-{line}", "role": role, "content": nodes},
    }


def test_codex_parser_returns_provenance_bound_call():
    result = {"schema_version": 1, "facts": []}
    calls = parse_codex_mcp_records(
        [
            _codex_event("begin", "call-1", {"query": "labels"}),
            _codex_event("end", "call-1", {"query": "labels"}, result, line=2),
        ],
        provenance=PROVENANCE,
    )

    assert len(calls) == 1
    assert calls[0].call_id == "call-1"
    assert calls[0].name == "mnemos_retrieve"
    assert calls[0].arguments == {"query": "labels"}
    assert calls[0].result == result
    assert calls[0].provenance == PROVENANCE
    assert calls[0].line_number == 2
    assert calls[0].model_call_correlated is False
    with pytest.raises(ValueError, match="model-call correlation"):
        require_tool_sequence(calls, ("mnemos_retrieve",))
    assert (
        require_tool_sequence(calls, ("mnemos_retrieve",), require_model_correlation=False) == calls
    )


def test_expected_tool_sequence_rejects_missing_independent_retrieval():
    result = {"schema_version": 1, "facts": []}
    calls = parse_codex_mcp_records(
        [
            _codex_event("begin", "call-1", {"query": "labels"}),
            _codex_event("end", "call-1", {"query": "labels"}, result, line=2),
        ],
        provenance=PROVENANCE,
    )
    with pytest.raises(ValueError, match="count"):
        require_tool_sequence(calls, ("mnemos_retrieve", "mnemos_retrieve"))


def test_codex_completion_alone_is_observation_not_model_correlation():
    calls = parse_codex_mcp_records(
        [_codex_event("end", "call-1", {"query": "labels"}, {"facts": []})],
        provenance=PROVENANCE,
    )
    assert calls[0].model_call_correlated is False


def test_codex_outer_response_item_must_match_mcp_completion():
    result = {"schema_version": 1, "facts": []}
    model = {
        "type": "response_item",
        "line": 1,
        "line_sha256": "1" * 64,
        "payload": {
            "type": "function_call",
            "call_id": "model-call",
            "name": "mnemos_retrieve",
            "arguments": '{"query":"labels"}',
        },
    }
    begin = _codex_event("begin", "model-call", {"query": "labels"}, line=2)
    end = _codex_event("end", "model-call", {"query": "labels"}, result, line=3)
    assert (
        parse_codex_mcp_records([model, begin, end], provenance=PROVENANCE)[0].call_id
        == "model-call"
    )
    assert parse_codex_mcp_records([model, begin, end], provenance=PROVENANCE)[
        0
    ].model_call_correlated

    model["payload"]["arguments"] = '{"query":"different"}'
    with pytest.raises(ValueError, match="differ"):
        parse_codex_mcp_records([model, begin, end], provenance=PROVENANCE)

    with pytest.raises(ValueError, match="arguments"):
        parse_codex_mcp_records(
            [
                _codex_event("begin", "call-1", {"query": "labels"}),
                _codex_event("end", "call-1", {"query": "other"}, {"facts": []}, line=2),
            ],
            provenance=PROVENANCE,
        )


def test_codex_parser_rejects_duplicate_and_error_completions():
    event = _codex_event("begin", "call-1", {"query": "labels"})
    duplicate = _codex_event("begin", "call-1", {"query": "labels"}, line=2)
    with pytest.raises(ValueError, match="duplicate"):
        parse_codex_mcp_records([event, duplicate], provenance=PROVENANCE)

    bad = _codex_event("end", "call-1", {"query": "labels"}, {"status": "degraded"}, line=2)
    with pytest.raises(ValueError, match="unsuccessful"):
        parse_codex_mcp_records(
            [_codex_event("begin", "call-1", {"query": "labels"}), bad],
            provenance=PROVENANCE,
        )


def test_codex_parser_rejects_ok_and_err_result():
    event = _codex_event("begin", "call-1", {"query": "labels"})
    end = _codex_event("end", "call-1", {"query": "labels"}, {"facts": []}, line=2)
    end["payload"]["result"]["Err"] = {"message": "also failed"}
    with pytest.raises(ValueError, match="Ok and/or Err"):
        parse_codex_mcp_records([event, end], provenance=PROVENANCE)


def test_claude_parser_correlates_response_scoped_tool_ids():
    value = {"schema_version": 1, "facts": []}
    raw_result = json.dumps({"result": json.dumps(value)})
    records = [
        _claude_line(
            [
                {
                    "type": "tool_use",
                    "id": "same-id",
                    "name": "mcp__plugin_mnemos-memory_mnemos__mnemos_retrieve",
                    "input": {"query": "labels"},
                }
            ],
            4,
        ),
        _claude_line(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "same-id",
                    "is_error": False,
                    "content": raw_result,
                }
            ],
            5,
        ),
    ]
    calls = parse_claude_tool_records(
        records,
        provenance=TraceProvenance(
            host="claude-code",
            process_id=4321,
            session_id="claude-session",
            source_file="stdout.jsonl",
            source_file_sha256="a" * 64,
        ),
    )

    assert calls[0].call_id == "response-4:same-id"
    assert calls[0].name.endswith("mnemos_retrieve")
    assert calls[0].result == value
    assert calls[0].line_number == 5


def test_claude_parser_rejects_mismatch_duplicate_and_final_text_only():
    with pytest.raises(ValueError, match="tool_result"):
        parse_claude_tool_records(
            [_claude_line([{"type": "tool_result", "tool_use_id": "missing", "content": "{}"}], 1)],
            provenance=TraceProvenance("claude", 1, "s", "x", "b" * 64),
        )

    call = {
        "type": "tool_use",
        "id": "same",
        "name": "mcp__plugin_mnemos-memory_mnemos__mnemos_retrieve",
        "input": {"query": "x"},
    }
    records = [_claude_line([call, call], 1)]
    with pytest.raises(ValueError, match="duplicate"):
        parse_claude_tool_records(
            records, provenance=TraceProvenance("claude", 1, "s", "x", "b" * 64)
        )

    with pytest.raises(ValueError, match="no tool_use"):
        parse_claude_tool_records(
            [_claude_line([{"type": "text", "text": "memory is fine"}], 1)],
            provenance=TraceProvenance("claude", 1, "s", "x", "b" * 64),
        )


def test_claude_parser_rejects_unmatched_call_and_error_result():
    call = {
        "type": "tool_use",
        "id": "one",
        "name": "mcp__plugin_mnemos-memory_mnemos__mnemos_retrieve",
        "input": {"query": "x"},
    }
    provenance = TraceProvenance("claude", 1, "s", "x", "b" * 64)
    with pytest.raises(ValueError, match="undispatched"):
        parse_claude_tool_records([_claude_line([call], 1)], provenance=provenance)

    with pytest.raises(ValueError, match="unsuccessful"):
        parse_claude_tool_records(
            [
                _claude_line([call], 1),
                _claude_line(
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": "one",
                            "is_error": True,
                            "content": "{}",
                        }
                    ],
                    2,
                ),
            ],
            provenance=provenance,
        )


def test_claude_parser_accepts_forget_review_tools_and_rejects_wrong_prefix():
    value = {"schema_version": 1, "preview": {"fact_id": "target"}}
    records = [
        _claude_line(
            [
                {
                    "type": "tool_use",
                    "id": "preview-1",
                    "name": "mcp__plugin_mnemos-memory_mnemos__mnemos_forget_preview",
                    "input": {"fact_id": "target"},
                }
            ],
            1,
        ),
        _claude_line(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "preview-1",
                    "content": json.dumps(value),
                }
            ],
            2,
        ),
    ]
    calls = parse_claude_tool_records(
        records, provenance=TraceProvenance("claude", 1, "s", "x", "b" * 64)
    )
    assert calls[0].name == "mnemos_forget_preview"

    records[0]["message"]["content"][0]["name"] = "mnemos_forget_related"
    with pytest.raises(ValueError, match="verified Mnemos prefix"):
        parse_claude_tool_records(
            records, provenance=TraceProvenance("claude", 1, "s", "x", "b" * 64)
        )


def test_codex_extra_mcp_call_cannot_borrow_another_calls_model_correlation():
    model = {
        "type": "response_item",
        "line": 1,
        "line_sha256": "a" * 64,
        "payload": {
            "type": "function_call",
            "call_id": "direct-1",
            "name": "mcp__mnemos__mnemos_retrieve",
            "arguments": '{"query":"x"}',
        },
    }
    good = _codex_event("end", "direct-1", {"query": "x"}, {"facts": []}, line=2)
    extra = _codex_event("end", "unissued", {"query": "y"}, {"facts": []}, line=3)
    with pytest.raises(ValueError, match="call sets differ"):
        parse_codex_mcp_records([model, good, extra], provenance=PROVENANCE)


def test_codex_actual_exec_transport_shape_is_preserved_as_uncorrelated():
    records = [
        {
            "type": "response_item",
            "line": 1,
            "line_sha256": "a" * 64,
            "payload": {"type": "message", "role": "user", "content": []},
        },
        {
            "type": "response_item",
            "line": 2,
            "line_sha256": "b" * 64,
            "payload": {
                "type": "custom_tool_call",
                "name": "exec",
                "call_id": "model-cell",
                "input": "const r = await tools.mnemos_retrieve({query: 'x'}); text(r);",
            },
        },
        _codex_event("end", "exec-unrelated-id", {"query": "x"}, {"facts": []}, line=3),
        {
            "type": "response_item",
            "line": 4,
            "line_sha256": "c" * 64,
            "payload": {
                "type": "function_call",
                "name": "wait",
                "call_id": "wait-cell",
                "arguments": '{"cell_id":"1"}',
            },
        },
    ]
    calls = parse_codex_mcp_records(records, provenance=PROVENANCE)
    assert len(calls) == 1 and not calls[0].model_call_correlated
    with pytest.raises(ValueError, match="correlation"):
        require_tool_sequence(calls, ["mnemos_retrieve"])


def test_duplicate_codex_completion_rejected_without_begin_record():
    records = [_codex_event("end", "dup", {}, {}, line=n) for n in (1, 2)]
    with pytest.raises(ValueError, match="duplicate"):
        parse_codex_mcp_records(records, provenance=PROVENANCE)


def test_claude_pending_id_reuse_is_ambiguous_even_in_different_response():
    use = {
        "type": "tool_use",
        "id": "same",
        "name": "mcp__plugin_mnemos-memory_mnemos__mnemos_retrieve",
        "input": {},
    }
    with pytest.raises(ValueError, match="ambiguous"):
        parse_claude_tool_records(
            [_claude_line([use], 1), _claude_line([use], 2)], provenance=PROVENANCE
        )


def test_claude_duplicate_same_response_split_across_lines_rejected():
    use = {
        "type": "tool_use",
        "id": "same",
        "name": "mcp__plugin_mnemos-memory_mnemos__mnemos_retrieve",
        "input": {},
    }
    done = {"type": "tool_result", "tool_use_id": "same", "content": "{}"}
    repeated = _claude_line([use], 3)
    repeated["message"]["id"] = "response-1"
    with pytest.raises(ValueError, match="duplicate"):
        parse_claude_tool_records(
            [_claude_line([use], 1), _claude_line([done], 2), repeated], provenance=PROVENANCE
        )


def test_trace_loader_binds_full_source_and_line_bytes(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"type":"session_meta","payload":{}}\r\n')
    provenance = replace(
        PROVENANCE,
        source_file=str(path),
        source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    records = load_source_records(path, provenance=provenance)
    assert records[0]["line"] == 1
    assert (
        records[0]["line_sha256"] == hashlib.sha256(path.read_bytes().splitlines()[0]).hexdigest()
    )
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="bytes changed"):
        load_source_records(path, provenance=provenance)


def test_codex_later_model_call_cannot_explain_an_earlier_completion():
    completion = _codex_event("end", "same-id", {"query": "x"}, {"facts": []}, line=1)
    model = {
        "type": "response_item",
        "line": 2,
        "line_sha256": "a" * 64,
        "payload": {
            "type": "function_call",
            "name": "mnemos_retrieve",
            "call_id": "same-id",
            "arguments": '{"query":"x"}',
        },
    }
    with pytest.raises(ValueError, match="precede"):
        parse_codex_mcp_records([completion, model], provenance=PROVENANCE)
