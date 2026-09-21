"""A tool caller cannot file capture evidence under a host's own channel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from tests.test_mcp_compatibility import (
    PROJECT_ID,
    ROOT,
    _call_result_text,
    _server_params_from_spec,
    _service_env,
    _start_scoped_service,
)


@pytest.mark.asyncio
async def test_reserved_source_kinds_are_refused_at_the_tool_boundary(tmp_path: Path) -> None:
    service, registry_path = _start_scoped_service(tmp_path, "capture-provenance")
    try:
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

                async def capture(source_kind: str) -> tuple[bool, str]:
                    result = await session.call_tool(
                        "mnemos_capture",
                        {
                            "content": "The deploy window is Thursday.",
                            "source_kind": source_kind,
                            "scope": "project",
                            "scope_id": PROJECT_ID,
                        },
                    )
                    return result.isError, _call_result_text(result)

                for claimed in ("claude_code_transcript", "Codex_Hook", " explicit_remember "):
                    refused, text = await capture(claimed)
                    assert refused, text
                    assert "cannot claim" in text

                refused, text = await capture("meeting-notes")
                assert not refused, text
                assert json.loads(text)["event_committed"] is True
    finally:
        service.stop()
