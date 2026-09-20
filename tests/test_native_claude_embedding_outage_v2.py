"""Match the real MCP projection without relaxing retrieval acceptance."""

import json
from copy import deepcopy

import pytest

from benchmarks.native_hosts.claude_embedding_outage_v2 import require_mcp_response_match
from tests.test_native_claude_embedding_outage import _response


@pytest.mark.asyncio
async def test_match_accepts_actual_mcp_fact_projection(monkeypatch):
    from mnemos import mcp_server

    response = {"schema_version": 1, **_response()}

    async def request(*args):
        return deepcopy(response)

    monkeypatch.setattr(mcp_server, "_service_context_from_ctx", lambda *args: object())
    monkeypatch.setattr(mcp_server, "_resolved_scope_payload", lambda *args: ("project", {}))
    monkeypatch.setattr(mcp_server, "_service_request", request)
    server = mcp_server.create_mcp_server()
    tool = server._tool_manager.get_tool("mnemos_retrieve")
    actual = json.loads(await tool.fn(query="deployment marker"))
    assert actual != response
    assert actual["facts"][0]["schema_version"] == 1
    require_mcp_response_match(actual, response)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "wrong fact"),
        ("version", 2),
        ("scope_id", "foreign"),
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
    ],
)
def test_match_rejects_changed_fact(field, value):
    service = _response()
    native = deepcopy(service)
    native["facts"][0]["schema_version"] = 1
    native["facts"][0][field] = value
    with pytest.raises(ValueError):
        require_mcp_response_match(native, service)


def test_match_rejects_changed_degraded_status():
    service = _response()
    native = deepcopy(service)
    native["facts"][0]["schema_version"] = 1
    native["status"] = "ok"
    with pytest.raises(ValueError):
        require_mcp_response_match(native, service)
