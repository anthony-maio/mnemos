from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import mnemos.mcp_server as mcp_server
from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.adapters.hermes_tools import dispatch_tool, get_tool_schemas
from mnemos.client import MnemosClient
from mnemos.config import MnemosConfig, ModuleConfig
from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.mcp_server import ServiceMCPContext
from mnemos.service import MnemosService
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.utils.embeddings import SimpleEmbeddingProvider
from mnemos.utils.llm import UnavailableLLMProvider
from mnemos.utils.storage import SQLiteStore


def context() -> ScopeContext:
    return ScopeContext(
        project_id="project-a",
        workspace_id="workspace-a",
        user_profile_id="profile-a",
        session_id="session-a",
        host="hermes",
    )


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    async def request(
        self, operation: str, payload: dict[str, Any], *, queue_on_failure: bool = True
    ) -> dict[str, Any]:
        self.calls.append((operation, payload, queue_on_failure))
        return {
            "schema_version": 1,
            "view": payload.get("view", "memory"),
            "facts": [],
            "captured_evidence": [],
            "context": "",
            "evidence_context": "",
            "token_count_method": "utf8_bytes_upper_bound",
            "context_token_upper_bound": 0,
        }


def test_hermes_retrieve_schema_and_dispatch_support_evidence_view() -> None:
    schema = next(item for item in get_tool_schemas() if item["name"] == "mnemos_retrieve")
    assert schema["parameters"]["properties"]["view"]["enum"] == [
        "memory",
        "captured_evidence",
    ]
    client = RecordingClient()
    result = json.loads(
        asyncio.run(
            dispatch_tool(
                client,
                context(),
                "mnemos_retrieve",
                {"query": "raw transcript", "view": "captured_evidence"},
            )
        )
    )
    assert result["view"] == "captured_evidence"
    assert client.calls[-1][1]["view"] == "captured_evidence"
    assert client.calls[-1][1]["context"] == context().model_dump(mode="json")


@pytest.mark.parametrize("invalid_view", ["bad", 7, None, []])
def test_hermes_retrieve_defaults_to_memory_and_rejects_invalid_view(
    tmp_path: Path, invalid_view: Any
) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    client = RecordingClient()
    provider = MnemosHermesProvider(client=client, registry=registry, user_profile_id="profile-a")
    provider.initialize("session-a", hermes_home=str(tmp_path), cwd=str(tmp_path))
    default = json.loads(provider.handle_tool_call("mnemos_retrieve", {"query": "current"}))
    assert default["view"] == "memory"
    assert client.calls[-1][1]["view"] == "memory"
    call_count = len(client.calls)
    rejected = json.loads(
        provider.handle_tool_call("mnemos_retrieve", {"query": "current", "view": invalid_view})
    )
    assert rejected["status"] == "rejected"
    assert len(client.calls) == call_count


@pytest.mark.asyncio
async def test_mcp_retrieve_forwards_view_without_scope_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_request(
        service_context: ServiceMCPContext, operation: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured.update(operation=operation, payload=payload)
        return {"schema_version": 1, "view": payload["view"], "facts": []}

    class FakeMCP:
        def tool(self, **_: Any) -> Any:
            def decorate(function: Any) -> Any:
                setattr(self, function.__name__, function)
                return function

            return decorate

        def resource(self, *_: Any, **__: Any) -> Any:
            return lambda function: function

    monkeypatch.setattr(mcp_server, "_service_request", fake_request)
    fake: Any = FakeMCP()
    mcp_server._register_service_tools(fake)
    scope = context()
    service_context = ServiceMCPContext(client=RecordingClient(), scope=scope)  # type: ignore[arg-type]
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=service_context))
    result = json.loads(
        await fake.mnemos_retrieve("raw transcript", view="captured_evidence", ctx=ctx)
    )
    assert result["view"] == "captured_evidence"
    assert captured["operation"] == "retrieve"
    assert captured["payload"]["view"] == "captured_evidence"
    assert captured["payload"]["context"] == scope.model_dump(mode="json")


def test_authenticated_hosts_separate_memory_and_captured_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="profile-a", enabled=True)
    resolved = registry.resolve(
        tmp_path, user_profile_id="profile-a", session_id="session-a", host="hermes"
    )
    config = MnemosConfig(
        modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
    )
    service = MnemosService(
        dispatcher_factory=lambda: LocalMemoryService(
            MnemosEngine(
                config=config,
                llm=UnavailableLLMProvider(),
                embedder=SimpleEmbeddingProvider(),
                store=SQLiteStore(str(tmp_path / "memory.sqlite")),
            ),
            registry,
        ),
        descriptor_path=tmp_path / "runtime.json",
    )
    service.start()
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    try:
        remembered = json.loads(
            asyncio.run(
                dispatch_tool(
                    client,
                    resolved,
                    "mnemos_remember",
                    {
                        "content": "Durable release marker is blue.",
                        "request_id": "view-explicit-1",
                    },
                )
            )
        )
        assert remembered["status"] == "committed"
        captured = service.submit(
            "capture",
            {
                "event_id": "view-captured-1",
                "context": resolved.model_dump(mode="json"),
                "scope": "project",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "role": "assistant",
                "content": "Raw captured release marker is red.",
                "source_kind": "fixture_transcript",
                "source_ref": "fixture:view-captured-1",
            },
        )
        assert captured["status"] == "committed"
        asyncio.run(client.request("maintenance", {}, queue_on_failure=False))

        def call(view: str) -> dict[str, Any]:
            return cast(
                dict[str, Any],
                json.loads(
                    asyncio.run(
                        dispatch_tool(
                            client,
                            resolved,
                            "mnemos_retrieve",
                            {"query": "release marker", "top_k": 20, "view": view},
                        )
                    )
                ),
            )

        memory = call("memory")
        evidence = call("captured_evidence")
        assert memory["view"] == "memory"
        assert any("Durable release marker" in row["content"] for row in memory["facts"])
        assert all("Raw captured release marker" not in row["content"] for row in memory["facts"])
        assert evidence["view"] == "captured_evidence"
        assert evidence["facts"] == []
        assert any(
            "Raw captured release marker" in row["content"]
            and row["sources"][0]["source_ref"] == "fixture:view-captured-1"
            and row["sources"][0]["source_event_id"]
            for row in evidence["captured_evidence"]
        )
        assert evidence["context"] == ""
        assert evidence["evidence_context"]

        captured_calls: dict[str, Any] = {}

        async def record_request(
            service_context: ServiceMCPContext,
            operation: str,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            captured_calls["operation"] = operation
            captured_calls["payload"] = payload
            return await service_context.client.request(operation, payload, queue_on_failure=False)

        class FakeMCP:
            def tool(self, **_: Any) -> Any:
                def decorate(function: Any) -> Any:
                    setattr(self, function.__name__, function)
                    return function

                return decorate

            def resource(self, *_: Any, **__: Any) -> Any:
                return lambda function: function

        monkeypatch.setattr(mcp_server, "_service_request", record_request)
        fake: Any = FakeMCP()
        mcp_server._register_service_tools(fake)
        ctx = SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=ServiceMCPContext(client=client, scope=resolved)
            )
        )
        mcp_evidence = json.loads(
            asyncio.run(
                fake.mnemos_retrieve("release marker", top_k=20, view="captured_evidence", ctx=ctx)
            )
        )
        assert captured_calls["operation"] == "retrieve"
        assert captured_calls["payload"]["context"] == resolved.model_dump(mode="json")
        assert mcp_evidence["view"] == "captured_evidence"
        assert any(
            "Raw captured release marker is red." in row["content"]
            for row in mcp_evidence["captured_evidence"]
        )
    finally:
        asyncio.run(client.close())
        service.stop()
