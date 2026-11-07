"""Inspect identifiers remain unambiguous and bound to the host's scope."""

import json
import sys
from types import SimpleNamespace
from typing import cast

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from mnemos.client import MnemosClient
from mnemos.mcp_server import ServiceMCPContext, _create_service_mcp_server
from mnemos.scopes import ScopeContext
from tests.test_mcp_compatibility import (
    PROFILE_ID,
    ROOT,
    _call_result_text,
    _server_params_from_spec,
    _service_env,
    _start_scoped_service,
)
from tests.test_mcp_tool_annotations import _sqlite_snapshot


@pytest.fixture
def inspector():
    calls = []
    scope = ScopeContext(
        user_profile_id="profile", project_id="project", host="codex", session_id="inspect"
    )

    class Client:
        async def request(self, operation, payload, *, queue_on_failure):
            assert queue_on_failure is False
            calls.append((operation, payload))
            return {"schema_version": 1, "fact": {"id": payload["fact_id"]}, "evidence": []}

    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=ServiceMCPContext(client=cast(MnemosClient, Client()), scope=scope)
        )
    )
    tool = _create_service_mcp_server()._tool_manager.get_tool("mnemos_inspect")
    assert tool is not None
    return tool, context, scope, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"fact_id": "target"},
        {"chunk_id": "target"},
        {"fact_id": "target", "chunk_id": "target"},
        {
            "fact_id": "target",
            "current_scope": "global",
            "scope_id": "foreign",
            "allowed_scopes": ["global"],
        },
    ],
)
async def test_native_fact_id_and_legacy_chunk_id_use_same_scoped_read(inspector, arguments):
    tool, context, scope, calls = inspector
    content, _ = await tool.run(arguments, context=context, convert_result=True)
    assert json.loads(content[0].text)["fact"]["id"] == "target"
    assert calls == [("inspect", {"fact_id": "target", "context": scope.model_dump(mode="json")})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"fact_id": None},
        {"fact_id": ""},
        {"chunk_id": " "},
        {"fact_id": 42},
        {"chunk_id": []},
        {"fact_id": "target", "chunk_id": "other"},
        {"fact_id": "target", "chunk_id": ""},
    ],
)
async def test_invalid_or_conflicting_identifiers_never_reach_service(inspector, arguments):
    tool, context, _, calls = inspector
    with pytest.raises(Exception):
        await tool.run(arguments, context=context, convert_result=True)
    assert calls == []


@pytest.mark.asyncio
async def test_actual_stdio_alias_preserves_evidence_scope_and_read_only_state(tmp_path):
    service, registry_path = _start_scoped_service(tmp_path, "inspect-identifiers")
    try:
        registry = service._dispatcher.registry
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        registry.register_project(foreign, user_profile_id=PROFILE_ID, enabled=True)
        registry.save(registry_path)
        identities = []
        for index, (project, content) in enumerate(
            ((ROOT, "Keep inspection evidence."), (foreign, "Private foreign inspection marker."))
        ):
            scope = registry.resolve(
                project, user_profile_id=PROFILE_ID, host="manual", session_id="inspect-alias"
            )
            payload = scope.model_dump(mode="json")
            service.submit(
                "remember",
                {
                    "event_id": f"inspect-alias-{index}",
                    "content": content,
                    "context": payload,
                    "scope": "project",
                },
            )
            service.submit("maintenance", {})
            facts = service.submit(
                "list", {"context": payload, "allowed_scopes": ["project"], "limit": 10}
            )["facts"]
            assert len(facts) == 1
            identities.append(facts[0]["id"])
        before = _sqlite_snapshot(tmp_path / "inspect-identifiers.db")
        params = _server_params_from_spec(
            {"command": "python", "args": ["-m", "mnemos.mcp_server"]},
            cwd=ROOT,
            env_overrides={
                "MNEMOS_TEST_PYTHON": sys.executable,
                **_service_env(service, registry_path),
            },
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                values = []
                for key in ("fact_id", "chunk_id"):
                    result = await session.call_tool("mnemos_inspect", {key: identities[0]})
                    assert not result.isError, _call_result_text(result)
                    values.append(json.loads(_call_result_text(result)))
                assert values[0] == values[1]
                assert values[0]["fact"]["id"] == identities[0] and values[0]["evidence"]
                denied = await session.call_tool(
                    "mnemos_inspect",
                    {
                        "fact_id": identities[1],
                        "current_scope": "global",
                        "allowed_scopes": ["project", "global"],
                    },
                )
                denied_payload = json.loads(_call_result_text(denied))
                assert denied_payload["status"] == "degraded"
                assert denied_payload["committed"] is False
                assert "fact_not_found" in denied_payload["error"]
                assert "fact" not in denied_payload
                assert "Private foreign inspection marker" not in _call_result_text(denied)
        assert _sqlite_snapshot(tmp_path / "inspect-identifiers.db") == before
    finally:
        service.stop()
