"""Native-host scope argument forms through the actual stdio MCP boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
@pytest.mark.parametrize(
    ("allowed_scopes", "valid"),
    [
        ("project", True),
        (["project"], True),
        ([], False),
        (["project", "unknown"], False),
        (["project", 1], False),
    ],
    ids=["legacy-string", "structured-list", "empty-list", "unknown-scope", "non-string"],
)
async def test_retrieve_scope_arguments_preserve_project_boundary(
    tmp_path: Path, allowed_scopes: object, valid: bool
) -> None:
    service, registry_path = _start_scoped_service(tmp_path, "scope-arguments")
    try:
        registry = service._dispatcher.registry
        foreign_project = tmp_path / "foreign-project"
        foreign_project.mkdir()
        registry.register_project(foreign_project, user_profile_id=PROFILE_ID, enabled=True)
        registry.save(registry_path)
        positive = "Project retrieval scope marker alpha."
        foreign = "Project retrieval scope marker beta."
        for index, (root, content) in enumerate(((ROOT, positive), (foreign_project, foreign))):
            context = registry.resolve(
                root, user_profile_id=PROFILE_ID, session_id=f"scope-seed-{index}", host="manual"
            )
            result = service.submit(
                "remember",
                {
                    "event_id": f"scope-argument-seed-{index}",
                    "context": context.model_dump(mode="json"),
                    "content": content,
                    "role": "user",
                    "scope": "project",
                },
            )
            assert result["status"] == "committed"
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
                result = await session.call_tool(
                    "mnemos_retrieve",
                    {"query": positive, "top_k": 20, "allowed_scopes": allowed_scopes},
                )
                text = _call_result_text(result)
                if not valid:
                    assert result.isError, text
                    return
                assert not result.isError, text
                payload = json.loads(text)
                assert positive in [fact["content"] for fact in payload["facts"]]
                assert foreign not in text
                assert all(fact["scope_id"] == "project-compat" for fact in payload["facts"])
    finally:
        service.stop()
