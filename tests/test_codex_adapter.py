from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.codex import (
    CAPTURE_EVENTS,
    RECALL_EVENTS,
    SUPPORTED_EVENTS,
    CodexAdapter,
    hook_output,
    installed_capabilities,
)
from mnemos.scopes import ScopeRegistry


class RecordingClient:
    def __init__(self) -> None:
        self.captures: list[dict[str, Any]] = []
        self.retrieves: list[dict[str, Any]] = []
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.captures.append(payload)
        return {"status": "committed", "committed": True, "event_id": payload["event_id"]}

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.retrieves.append(payload)
        return {"context": "1. local project convention"}

    async def request(self, operation: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.requests.append((operation, payload))
        return {"status": "committed", "completed": True}


class DegradedClient(RecordingClient):
    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.retrieves.append(payload)
        return {"status": "degraded", "committed": False, "degraded": True}


def make_adapter(tmp_path: Path, client: Any | None = None) -> tuple[CodexAdapter, dict[str, Any]]:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    adapter = CodexAdapter(
        client=client or RecordingClient(), registry=registry, user_profile_id="alice"
    )
    payload = {"cwd": str(tmp_path), "session_id": "session-1", "turn_id": "turn-1"}
    return adapter, payload


@pytest.mark.asyncio
async def test_user_prompt_retrieves_and_captures_stable_event(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    event = {
        **payload,
        "prompt": "Use the local SQLite convention",
        "hook_event_name": "UserPromptSubmit",
    }

    first = await adapter.handle("UserPromptSubmit", event)
    second = await adapter.handle("UserPromptSubmit", event)

    assert first.captured == second.captured == 1
    assert first.event_ids == second.event_ids
    assert len(client.captures) == 2
    assert client.captures[0]["role"] == "user"
    assert client.captures[0]["source_kind"] == "codex_hook"
    assert client.retrieves[0]["context"]["host"] == "codex"
    assert client.retrieves[0]["max_context_tokens"] == 1200


@pytest.mark.asyncio
async def test_distinct_native_turns_preserve_identical_prompt_events(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    first = await adapter.handle(
        "UserPromptSubmit",
        {**payload, "turn_id": "turn-1", "prompt": "Repeat this exact prompt"},
    )
    second = await adapter.handle(
        "UserPromptSubmit",
        {**payload, "turn_id": "turn-2", "prompt": "Repeat this exact prompt"},
    )

    assert first.captured == second.captured == 1
    assert first.event_ids[0] != second.event_ids[0]
    assert [item["event_id"] for item in client.captures] == [
        first.event_ids[0],
        second.event_ids[0],
    ]


@pytest.mark.asyncio
async def test_capture_events_require_native_turn_identity(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    for event, content in (
        ("UserPromptSubmit", {"prompt": "remember this"}),
        ("Stop", {"last_assistant_message": "done"}),
    ):
        result = await adapter.handle(event, {**payload, "turn_id": None, **content})
        assert result.status == "manual_mcp"
        assert result.degraded is True
    assert client.captures == []


@pytest.mark.asyncio
async def test_stop_captures_assistant_without_continuation(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    result = await adapter.handle(
        "Stop", {**payload, "last_assistant_message": "The change is ready."}
    )

    assert result.captured == 1
    assert client.captures[0]["role"] == "assistant"
    assert hook_output("Stop", result) == {"continue": True}
    assert "decision" not in hook_output("Stop", result)


@pytest.mark.asyncio
async def test_stop_hook_reentry_does_not_capture_again(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    result = await adapter.handle(
        "Stop",
        {**payload, "stop_hook_active": True, "last_assistant_message": "Continuation text"},
    )
    assert result.status == "skipped"
    assert client.captures == []


@pytest.mark.asyncio
async def test_missing_profile_or_unregistered_project_degrades_without_transport(
    tmp_path: Path,
) -> None:
    client = RecordingClient()
    registry = ScopeRegistry()
    adapter = CodexAdapter(client=client, registry=registry, user_profile_id="alice")
    result = await adapter.handle(
        "UserPromptSubmit",
        {"cwd": str(tmp_path), "session_id": "session-1", "prompt": "remember this"},
    )
    assert result.status == "manual_mcp"
    assert result.degraded is True
    assert client.captures == []
    assert client.retrieves == []


@pytest.mark.asyncio
async def test_service_retrieval_degradation_is_reported_without_blocking(tmp_path: Path) -> None:
    client = DegradedClient()
    adapter, payload = make_adapter(tmp_path, client)
    result = await adapter.handle(
        "UserPromptSubmit", {**payload, "prompt": "Find local project context"}
    )
    assert result.degraded is True
    assert result.status == "degraded"
    assert result.context == ""


@pytest.mark.asyncio
async def test_post_compact_recall_and_session_end_maintenance(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    restored = await adapter.handle("PostCompact", payload)
    compacted = await adapter.handle("PreCompact", payload)
    ended = await adapter.handle("SessionEnd", payload)
    assert restored.context == "1. local project convention"
    assert compacted.status == "ok"
    assert ended.status == "ok"
    assert [item[0] for item in client.requests] == ["maintenance", "maintenance"]


def test_supported_event_contract_and_capability_probe_are_local() -> None:
    assert {
        "SessionStart",
        "UserPromptSubmit",
        "PreCompact",
        "PostCompact",
        "Stop",
        "SessionEnd",
    } <= set(SUPPORTED_EVENTS)
    assert CAPTURE_EVENTS == {"UserPromptSubmit", "Stop"}
    assert RECALL_EVENTS == {"SessionStart", "UserPromptSubmit", "PostCompact"}
    capabilities = installed_capabilities()
    assert set(capabilities["events"]) == set(SUPPORTED_EVENTS)
    assert capabilities["fallback"] == "manual_mcp"


def test_hook_output_does_not_emit_context_for_capture_only() -> None:
    from mnemos.adapters.codex import CodexHookResult

    assert hook_output("PreCompact", CodexHookResult(status="ok")) == {}


def test_post_compact_output_uses_only_published_fields() -> None:
    from mnemos.adapters.codex import CodexHookResult

    result = CodexHookResult(status="ok", context="restored context")
    assert hook_output("PostCompact", result) == {"continue": True}
