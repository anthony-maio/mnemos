import asyncio
import json
from typing import Any

from mnemos.adapters.hermes_tools import dispatch_tool, get_tool_schemas
from mnemos.antigravity import build_antigravity_artifact
from mnemos.scopes import ScopeContext


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    async def request(
        self, operation: str, payload: dict[str, Any], *, queue_on_failure: bool
    ) -> dict[str, Any]:
        self.calls.append((operation, payload, queue_on_failure))
        return {"status": "ok", "facts": []}


def _context() -> ScopeContext:
    return ScopeContext(
        project_id="project-a",
        workspace_id="workspace-a",
        user_profile_id="profile-a",
        session_id="hermes-guidance-test",
        host="hermes",
    )


def test_generated_guidance_is_conditional_for_shared_host_schemas() -> None:
    artifact = build_antigravity_artifact("codex", "codex-agents")

    assert "registered project identity" in artifact
    assert "For MCP tools whose schema exposes `current_scope` and `allowed_scopes`" in artifact
    assert "provider-owned tools without those fields, omit them" in artifact
    assert "`mnemos_store` or `mnemos_remember`" in artifact
    assert "only when the host exposes it and its schema supports the call" in artifact
    assert "`mnemos_consolidate`" in artifact
    assert "Use `current_scope=project` and `allowed_scopes=project`." not in artifact
    assert "call `mnemos_store` only for durable facts" not in artifact
    assert "call `mnemos_consolidate`." not in artifact


def test_hermes_schema_rejects_scope_override_and_forwards_query_only() -> None:
    schemas = {schema["name"]: schema for schema in get_tool_schemas()}
    retrieve_properties = schemas["mnemos_retrieve"]["parameters"]["properties"]
    assert "current_scope" not in retrieve_properties
    assert "allowed_scopes" not in retrieve_properties
    assert "mnemos_remember" in schemas
    assert "mnemos_store" not in schemas
    assert "mnemos_consolidate" not in schemas

    client = RecordingClient()
    rejected = json.loads(
        asyncio.run(
            dispatch_tool(
                client,
                _context(),
                "mnemos_retrieve",
                {
                    "query": "release labels",
                    "current_scope": "project",
                    "allowed_scopes": "project",
                },
            )
        )
    )
    assert rejected["status"] == "rejected"
    assert client.calls == []

    accepted = json.loads(
        asyncio.run(
            dispatch_tool(client, _context(), "mnemos_retrieve", {"query": "release labels"})
        )
    )
    assert accepted == {"status": "ok", "facts": []}
    operation, payload, queued = client.calls[0]
    assert operation == "retrieve"
    assert queued is False
    assert payload["query"] == "release labels"
    assert payload["allowed_scopes"] == ["project"]
    assert payload["context"] == _context().model_dump(mode="json")
    assert "current_scope" not in payload
