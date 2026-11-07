"""The real MCP transport preserves compact recall and inspectable source evidence."""

import json
import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from tests.test_mcp_compatibility import (
    PROFILE_ID,
    ROOT,
    _call_result_text,
    _server_params_from_spec,
    _service_env,
    _start_scoped_service,
)


@pytest.mark.asyncio
async def test_compact_mcp_retains_scoped_literal_fact_and_source_inspection(tmp_path):
    service, registry_path = _start_scoped_service(tmp_path, "compact-mcp")
    try:
        registry = service._dispatcher.registry
        foreign_root = tmp_path / "foreign"
        foreign_root.mkdir()
        registry.register_project(foreign_root, user_profile_id=PROFILE_ID, enabled=True)
        registry.save(registry_path)
        literal = "Cedar production retains logs for 30 days; staging retains 7. Do not apply this to exports. caf\u00e9"
        for index, (path, content) in enumerate(
            [(ROOT, literal), (foreign_root, "Cedar forbidden marker")]
        ):
            context = registry.resolve(
                path, user_profile_id=PROFILE_ID, session_id="seed", host="manual"
            )
            reply = service.submit(
                "remember",
                {
                    "event_id": f"compact-seed-{index}",
                    "context": context.model_dump(mode="json"),
                    "content": content,
                    "role": "user",
                    "scope": "project",
                    "source_ref": "fixture:source",
                },
            )
            assert reply["status"] == "committed"
        service.submit("maintenance", {})
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
                tools = await session.list_tools()
                tool = next(t for t in tools.tools if t.name == "mnemos_retrieve")
                assert tool.inputSchema["properties"]["response_format"]["enum"] == [
                    "full",
                    "compact",
                ]
                args = {"query": "Cedar", "allowed_scopes": ["project"]}
                full = await session.call_tool("mnemos_retrieve", args)
                compact = await session.call_tool(
                    "mnemos_retrieve", {**args, "response_format": "compact"}
                )
                assert not full.isError and not compact.isError
                full_text, compact_text = _call_result_text(full), _call_result_text(compact)
                a, b = json.loads(full_text), json.loads(compact_text)
                assert len(b["references"]) == len(a["facts"]) == 1
                assert b["context"] == a["context"] and literal in b["context"]
                assert "forbidden marker" not in compact_text
                assert len(compact_text.encode()) < len(full_text.encode())
                assert "\n" not in compact_text
                assert (
                    b["references"][0]["original_source_status"]
                    == a["facts"][0]["original_source_status"]
                )
                assert b["references"][0]["sources"] == [
                    {k: v for k, v in source.items() if k != "excerpt"}
                    for source in a["facts"][0]["sources"]
                ]
                inspected = await session.call_tool(
                    "mnemos_inspect", {"chunk_id": b["references"][0]["id"]}
                )
                evidence = json.loads(_call_result_text(inspected))
                assert not inspected.isError
                assert evidence["fact"]["content"] == literal
                assert evidence["evidence"] == a["facts"][0]["sources"]
                invalid = await session.call_tool(
                    "mnemos_retrieve", {**args, "response_format": "summary"}
                )
                assert invalid.isError
    finally:
        service.stop()
