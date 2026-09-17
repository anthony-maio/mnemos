"""Transport contracts for reviewed related-forget operations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mnemos.control_plane import ControlPlaneService
import mnemos.mcp_server as mcp_module
from mnemos.mcp_server import ServiceMCPContext, _register_service_tools
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.ui_server import MnemosUiRouter


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def request(self, operation: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.calls.append((operation, payload))
        return {"status": "preview" if operation == "forget_preview" else "committed"}


class _FailingClient(_RecordingClient):
    async def request(self, operation: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        _ = payload
        raise RuntimeError(f"stale {operation}")


def _context(project_id: str | None = "project-a") -> ScopeContext:
    return ScopeContext(
        user_profile_id="profile-a",
        project_id=project_id,
        session_id="transport-session",
        host="manual",
    )


def test_mcp_related_forget_tools_validate_and_inject_context() -> None:
    class FakeMcp:
        def __init__(self) -> None:
            self.tools: dict[str, Any] = {}

        def tool(self, **_options: Any) -> Any:
            return lambda function: self.tools.setdefault(function.__name__, function)

        def resource(self, _uri: str) -> Any:
            return lambda function: function

    mcp = FakeMcp()
    _register_service_tools(mcp)
    client = _RecordingClient()
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=ServiceMCPContext(cast(Any, client), _context())
        )
    )

    preview_text = asyncio.run(mcp.tools["mnemos_forget_preview"]("fact-a", 2, 25, ctx=ctx))
    assert json.loads(preview_text)["status"] == "preview"
    assert client.calls[0][0] == "forget_preview"
    assert client.calls[0][1]["context"]["project_id"] == "project-a"
    digest = "a" * 64
    related_text = asyncio.run(
        mcp.tools["mnemos_forget_related"](
            "fact-a", 2, digest, 25, ["copy-a"], ["event-a"], ctx=ctx
        )
    )
    assert json.loads(related_text)["status"] == "committed"
    assert client.calls[1][1]["selected_fact_ids"] == ["copy-a"]

    with pytest.raises(ValueError, match="integer"):
        asyncio.run(mcp.tools["mnemos_forget_preview"]("fact-a", True, 25, ctx=ctx))
    with pytest.raises(ValueError, match="list"):
        asyncio.run(
            mcp.tools["mnemos_forget_related"]("fact-a", 2, digest, 25, ("copy-a",), [], ctx=ctx)
        )
    no_project = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=ServiceMCPContext(cast(Any, client), _context(project_id=None))
        )
    )
    with pytest.raises(ValueError, match="project context"):
        asyncio.run(mcp.tools["mnemos_forget_preview"]("fact-a", 2, 25, ctx=no_project))


def test_fastmcp_call_tool_keeps_expected_version_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastMCP's real argument model must reject bool before service dispatch."""
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.fastmcp.exceptions import ToolError

    monkeypatch.setattr(mcp_module, "Context", Context, raising=False)
    server = FastMCP("related-forget-transport-test")
    _register_service_tools(server)
    client = _RecordingClient()
    service_context = ServiceMCPContext(cast(Any, client), _context())
    monkeypatch.setattr(
        mcp_module,
        "_service_context_from_ctx",
        lambda *_args: service_context,
    )

    async def invoke() -> None:
        with pytest.raises(ToolError, match="valid integer"):
            await server._tool_manager.call_tool(
                "mnemos_forget_preview",
                {"fact_id": "fact-a", "expected_version": True},
            )
        await server._tool_manager.call_tool(
            "mnemos_forget_preview",
            {
                "fact_id": "fact-a",
                "expected_version": 2,
                "context": {"project_id": "foreign-project"},
            },
        )

    asyncio.run(invoke())
    assert len(client.calls) == 1
    assert client.calls[0][1]["context"]["project_id"] == "project-a"


def test_control_plane_related_forget_uses_typed_payloads(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="profile-a", enabled=True)
    registry_path = tmp_path / "scopes.json"
    registry.save(registry_path)
    client = _RecordingClient()
    service = ControlPlaneService(
        cwd=project,
        env={
            "MNEMOS_SCOPE_REGISTRY": str(registry_path),
            "MNEMOS_USER_PROFILE_ID": "profile-a",
        },
        client=client,
    )

    preview = service.preview_forget_memory("fact-a", 2, 20)
    assert preview["status"] == "preview"
    assert client.calls[0][1]["context"]["project_id"]
    service.forget_related_memory("fact-a", 2, "b" * 64, 20, ["copy-a"], ["event-a"])
    assert client.calls[1][0] == "forget_related"
    assert client.calls[1][1]["selected_source_event_ids"] == ["event-a"]
    with pytest.raises(ValueError, match="integer"):
        service.preview_forget_memory("fact-a", True)
    with pytest.raises(ValueError, match="list"):
        service.forget_related_memory("fact-a", 2, "b" * 64, selected_fact_ids=("copy-a",))  # type: ignore[arg-type]

    failing = ControlPlaneService(
        cwd=project,
        env={
            "MNEMOS_SCOPE_REGISTRY": str(registry_path),
            "MNEMOS_USER_PROFILE_ID": "profile-a",
        },
        client=_FailingClient(),
    )
    with pytest.raises(RuntimeError, match="stale forget_preview"):
        failing.preview_forget_memory("fact-a", 2)


def test_ui_related_forget_routes_reject_coercion_and_forward_payload() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        def preview_forget_memory(self, *args: Any) -> dict[str, Any]:
            self.calls.append(("preview", args))
            return {"status": "preview"}

        def forget_memory(self, *args: Any) -> dict[str, Any]:
            if not isinstance(args[1], int) or isinstance(args[1], bool):
                raise ValueError("expected_version must be an integer")
            self.calls.append(("forget", args))
            return {"status": "forgotten"}

        def forget_related_memory(self, *args: Any) -> dict[str, Any]:
            if isinstance(args[1], bool):
                raise ValueError("expected_version must be an integer")
            self.calls.append(("related", args))
            return {"status": "committed"}

    service = FakeService()
    router = MnemosUiRouter(service)  # type: ignore[arg-type]
    preview = router.handle(
        "POST", "/api/memory/fact-a/forget-preview", b'{"expected_version": 2, "limit": 20}'
    )
    assert preview.status == 200
    assert service.calls[0] == ("preview", ("fact-a", 2, 20))
    related = router.handle(
        "POST",
        "/api/memory/fact-a/forget-related",
        json.dumps(
            {
                "expected_version": 2,
                "preview_digest": "c" * 64,
                "preview_limit": 20,
                "selected_fact_ids": ["copy-a"],
                "selected_source_event_ids": ["event-a"],
            }
        ).encode(),
    )
    assert related.status == 200
    assert service.calls[1] == (
        "related",
        ("fact-a", 2, "c" * 64, 20, ["copy-a"], ["event-a"]),
    )
    invalid = router.handle(
        "POST",
        "/api/memory/fact-a/forget-related",
        b'{"expected_version": true, "preview_digest": "c"}',
    )
    assert invalid.status == 400
    assert len(service.calls) == 2

    legacy = router.handle("POST", "/api/memory/fact-a/forget", b'{"expected_version": "2"}')
    assert legacy.status == 200
    assert service.calls[2] == ("forget", ("fact-a", 2))
