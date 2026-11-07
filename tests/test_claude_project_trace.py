"""Project MCP acceptance binds one explicit namespace without rewriting evidence."""

import json

import pytest

from benchmarks.native_hosts.shared_trace import TraceProvenance, parse_claude_tool_records

PREFIX = "mcp__mnemos__"
PLUGIN = "mcp__plugin_mnemos-memory_mnemos__"
PROVENANCE = TraceProvenance("claude-code", 123, "session", "raw.jsonl", "a" * 64)


def records(name=PREFIX + "mnemos_retrieve", *, failed=False):
    return [
        {
            "line": 1,
            "line_sha256": "b" * 64,
            "type": "assistant",
            "message": {
                "id": "response-1",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call-1", "name": name, "input": {"query": "guide"}}
                ],
            },
        },
        {
            "line": 2,
            "line_sha256": "c" * 64,
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "is_error": failed,
                        "content": json.dumps({"facts": [], "schema_version": 1}),
                    }
                ],
            },
        },
    ]


def test_project_namespace_requires_explicit_binding_and_preserves_provenance():
    source = records()
    original = json.dumps(source, sort_keys=True)
    with pytest.raises(ValueError, match="verified Mnemos prefix"):
        parse_claude_tool_records(source, provenance=PROVENANCE)
    calls = parse_claude_tool_records(source, provenance=PROVENANCE, tool_prefix=PREFIX)
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "mnemos_retrieve" and call.call_id == "response-1:call-1"
    assert call.provider_call_id == "call-1" and call.model_call_correlated
    assert call.provenance == PROVENANCE and call.line_sha256 == "c" * 64
    assert call.arguments == {"query": "guide"} and call.result == {
        "facts": [],
        "schema_version": 1,
    }
    assert json.dumps(source, sort_keys=True) == original


@pytest.mark.parametrize("prefix", ["", "mcp__other__", "mnemos_", "mcp__mnemos"])
def test_unrecognized_namespace_binding_is_rejected(prefix):
    with pytest.raises(ValueError, match="prefix"):
        parse_claude_tool_records(records(), provenance=PROVENANCE, tool_prefix=prefix)


@pytest.mark.parametrize(
    "name",
    [
        PLUGIN + "mnemos_retrieve",
        "mcp__other__mnemos_retrieve",
        "mnemos_retrieve",
        PREFIX + "mnemos_retrieve_extra",
        PREFIX + PLUGIN + "mnemos_retrieve",
    ],
)
def test_project_binding_rejects_other_namespaces_and_unknown_tools(name):
    with pytest.raises(ValueError, match="prefix|unsupported native tool"):
        parse_claude_tool_records(records(name), provenance=PROVENANCE, tool_prefix=PREFIX)


def test_project_namespace_does_not_relax_failed_result_or_call_correlation():
    with pytest.raises(ValueError, match="unsuccessful"):
        parse_claude_tool_records(records(failed=True), provenance=PROVENANCE, tool_prefix=PREFIX)
    source = records()
    source[1]["message"]["content"][0]["tool_use_id"] = "different-call"
    with pytest.raises(ValueError, match="preceding unmatched"):
        parse_claude_tool_records(source, provenance=PROVENANCE, tool_prefix=PREFIX)
