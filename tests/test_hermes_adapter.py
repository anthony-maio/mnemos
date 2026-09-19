from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import mnemos.adapters.hermes as hermes_adapter
from mnemos.adapters.hermes import (
    HERMES_API_AVAILABLE,
    HermesMemoryProvider,
    MnemosHermesProvider,
    capabilities,
)
from mnemos.client import Outbox
from mnemos.scopes import ScopeRegistry


class ProviderCollector:
    def __init__(self) -> None:
        self.provider: Any | None = None

    def register_memory_provider(self, provider: Any) -> None:
        self.provider = provider


class FakeClient:
    def __init__(self, status: str = "committed") -> None:
        self.status = status
        self.captures: list[dict[str, Any]] = []
        self.retrieves: list[dict[str, Any]] = []

    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.captures.append(payload)
        return {"status": self.status, "committed": self.status == "committed"}

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.retrieves.append(payload)
        return {"context": "1. remembered local convention"}


def test_native_tools_route_through_initialized_provider_scope(tmp_path: Path) -> None:
    import json

    class ToolClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def request(
            self, operation: str, payload: dict[str, Any], *, queue_on_failure: bool
        ) -> dict[str, Any]:
            assert queue_on_failure is False
            self.calls.append((operation, payload))
            return {"status": "committed", "processing_status": "pending", "job_id": "job-1"}

    client = ToolClient()
    instance = make_provider(tmp_path, client)
    assert "mnemos_correct" in {item["name"] for item in instance.get_tool_schemas()}
    result = json.loads(
        instance.handle_tool_call(
            "mnemos_remember", {"content": "Use SQLite.", "request_id": "remember-sqlite"}
        )
    )
    assert result["processing_status"] == "pending"
    assert client.calls[0][0] == "remember"
    assert client.calls[0][1]["context"] == instance._current_context().model_dump(mode="json")
    rejected = json.loads(
        instance.handle_tool_call(
            "mnemos_remember",
            {"content": "Use SQLite.", "request_id": "remember-sqlite", "context": {}},
        )
    )
    assert rejected["status"] == "rejected"
    assert len(client.calls) == 1


def test_native_tool_before_initialization_fails_closed(tmp_path: Path) -> None:
    import json

    instance = MnemosHermesProvider(client=FakeClient(), registry=ScopeRegistry())
    result = json.loads(instance.handle_tool_call("mnemos_inspect", {"fact_id": "any"}))
    assert result["status"] == "rejected"


def make_provider(tmp_path: Path, client: FakeClient | None = None) -> MnemosHermesProvider:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    result = MnemosHermesProvider(
        client=client or FakeClient(),
        registry=registry,
        user_profile_id="alice",
    )
    result.initialize("hermes-session", hermes_home=str(tmp_path / "hermes"), cwd=str(tmp_path))
    return result


def test_optional_contract_is_honest_when_hermes_is_absent() -> None:
    assert capabilities().fallback == "manual_mcp"
    assert capabilities().api_available is HERMES_API_AVAILABLE
    assert HermesMemoryProvider is MnemosHermesProvider


def test_register_uses_valid_environment_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "scopes.json"
    ScopeRegistry(profiles=["alice"]).save(registry_path)
    monkeypatch.setattr(hermes_adapter, "HERMES_API_AVAILABLE", True)
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(registry_path))
    monkeypatch.setenv("MNEMOS_USER_PROFILE_ID", "alice")

    collector = ProviderCollector()
    assert hermes_adapter.register(collector) is True
    assert isinstance(collector.provider, MnemosHermesProvider)
    assert collector.provider.registry_path == registry_path
    assert collector.provider.is_available() is True


def test_registry_path_explicit_value_precedes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "environment.json"
    explicit_path = tmp_path / "explicit.json"
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(env_path))
    provider = MnemosHermesProvider(
        registry_path=explicit_path,
        user_profile_id="alice",
    )

    assert provider.registry_path == explicit_path


@pytest.mark.parametrize(
    ("registry_value", "expected_reason"),
    [
        (None, "Mnemos scope registry is not configured"),
        ("directory", "Mnemos scope registry is not configured"),
        ("missing", "Mnemos scope registry is not configured"),
        ("malformed", "Mnemos scope registry is invalid"),
    ],
)
def test_provider_availability_rejects_unusable_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_value: str | None,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(hermes_adapter, "HERMES_API_AVAILABLE", True)
    monkeypatch.setenv("MNEMOS_USER_PROFILE_ID", "alice")
    if registry_value == "directory":
        target = tmp_path / "scopes"
        target.mkdir()
        monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(target))
    elif registry_value == "missing":
        monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(tmp_path / "missing.json"))
    elif registry_value == "malformed":
        target = tmp_path / "scopes.json"
        target.write_text("not-json", encoding="utf-8")
        monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(target))
    else:
        monkeypatch.delenv("MNEMOS_SCOPE_REGISTRY", raising=False)
        monkeypatch.setattr(
            hermes_adapter,
            "default_registry_path",
            lambda: tmp_path / "default-scopes.json",
        )

    provider = MnemosHermesProvider()
    assert provider.is_available() is False
    assert provider.unavailable_reason() == expected_reason


def test_injected_registry_does_not_require_registry_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hermes_adapter, "HERMES_API_AVAILABLE", True)
    provider = MnemosHermesProvider(
        registry=ScopeRegistry(profiles=["alice"]),
        user_profile_id="alice",
    )

    assert provider.is_available() is True


def test_provider_availability_rejects_missing_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "scopes.json"
    ScopeRegistry().save(registry_path)
    monkeypatch.setattr(hermes_adapter, "HERMES_API_AVAILABLE", True)
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(registry_path))
    monkeypatch.delenv("MNEMOS_USER_PROFILE_ID", raising=False)

    provider = MnemosHermesProvider()
    assert provider.is_available() is False
    assert provider.unavailable_reason() == (
        "MNEMOS_USER_PROFILE_ID or Hermes agent_identity is required"
    )


def test_provider_availability_rejects_missing_hermes_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "scopes.json"
    ScopeRegistry(profiles=["alice"]).save(registry_path)
    monkeypatch.setattr(hermes_adapter, "HERMES_API_AVAILABLE", False)
    monkeypatch.setenv("MNEMOS_SCOPE_REGISTRY", str(registry_path))
    monkeypatch.setenv("MNEMOS_USER_PROFILE_ID", "alice")

    provider = MnemosHermesProvider()
    assert provider.is_available() is False
    assert provider.unavailable_reason() == ("Hermes Agent memory_provider API is not installed")


def test_initialize_uses_session_cwd_and_profile_scope(tmp_path: Path) -> None:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="Alice", enabled=True)
    provider = MnemosHermesProvider(client=FakeClient(), registry=registry)
    provider.initialize(
        "s1",
        hermes_home=str(tmp_path / "profile"),
        cwd=str(tmp_path),
        agent_identity="Alice",
    )
    assert provider._context is not None
    assert provider._context.user_profile_id == "Alice"
    assert provider._context.host == "hermes"
    assert provider._context.session_id == "s1"


def test_session_switch_rebinds_context_and_reads_current_service(tmp_path: Path) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)
    instance.prefetch("a useful project query")
    instance.on_session_switch("hermes-session-2", reset=True)
    assert instance.session_id == "hermes-session-2"
    assert instance._context is not None
    assert instance._context.session_id == "hermes-session-2"
    instance.prefetch("a useful project query")
    assert len(client.retrieves) == 2
    assert client.retrieves[-1]["context"]["session_id"] == "hermes-session-2"


def test_prefetch_reads_current_service_each_time(tmp_path: Path) -> None:
    class ChangingClient(FakeClient):
        result = "1. remembered local convention"

        async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.retrieves.append(payload)
            return {"context": self.result}

    client = ChangingClient()
    instance = make_provider(tmp_path, client)
    assert instance.prefetch("What local convention applies?") == "1. remembered local convention"
    client.result = "2. corrected current convention"
    assert instance.prefetch("What local convention applies?") == "2. corrected current convention"
    assert len(client.retrieves) == 2
    assert client.retrieves[0]["context"]["user_profile_id"] == "alice"
    assert client.retrieves[0]["max_context_tokens"] == 1200


def test_sync_turn_preserves_roles_and_is_idempotent_source_identity(tmp_path: Path) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)
    instance.sync_turn(
        "Use SQLite locally",
        "I will use SQLite locally.",
        messages=[
            {"role": "user", "content": "Use SQLite locally"},
            {"role": "assistant", "content": "I will use SQLite locally."},
        ],
    )
    instance.sync_turn(
        "Use SQLite locally",
        "I will use SQLite locally.",
        messages=[
            {"role": "user", "content": "Use SQLite locally"},
            {"role": "assistant", "content": "I will use SQLite locally."},
        ],
    )
    instance.shutdown()
    assert len(client.captures) == 4
    assert sorted(item["role"] for item in client.captures) == [
        "assistant",
        "assistant",
        "user",
        "user",
    ]
    assert len({item["event_id"] for item in client.captures if item["role"] == "user"}) == 1
    assert len({item["event_id"] for item in client.captures if item["role"] == "assistant"}) == 1


def test_outbox_ack_status_counts_as_durable_enqueue(tmp_path: Path) -> None:
    client = FakeClient(status="queued")
    instance = make_provider(tmp_path, client)
    result = instance.on_pre_compress(
        [{"role": "user", "content": "checkpoint this evidence"}],
        require_checkpoint=True,
    )
    assert "checkpoint captured 1" in result


def test_required_checkpoint_raises_when_transport_cannot_accept(tmp_path: Path) -> None:
    class BrokenClient(FakeClient):
        async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise OSError("bridge unavailable")

    instance = make_provider(tmp_path, BrokenClient())
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(
            [{"role": "assistant", "content": "must survive compaction"}],
            require_checkpoint=True,
        )


def test_memory_write_mirroring_deduplicates_repeated_callbacks(tmp_path: Path) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)
    instance.on_memory_write(
        "add", "user", "Prefers local processing", {"session_id": "hermes-session"}
    )
    instance.on_memory_write(
        "add", "user", "Prefers local processing", {"session_id": "hermes-session"}
    )
    instance.shutdown()
    assert len(client.captures) == 1
    assert client.captures[0]["source_kind"] == "hermes_memory_provider"


def test_repeated_same_event_reuses_occurrence_and_changed_content_gets_new_identity(
    tmp_path: Path,
) -> None:
    outbox = Outbox(tmp_path / "outbox.json")

    class OutboxClient:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(payload)
            outbox.append(payload)
            return {"status": "queued", "committed": False}

    client = OutboxClient()
    instance = make_provider(tmp_path, client)  # type: ignore[arg-type]
    messages = [{"role": "user", "content": "Use the stable release label."}]
    instance.sync_turn("", "", messages=messages)
    instance.on_session_end(messages)
    instance.sync_turn("", "", messages=[{"role": "user", "content": "Use a new label."}])

    instance.shutdown()
    pending = outbox.pending()
    assert len(client.payloads) == 3
    assert client.payloads[0]["event_id"] == client.payloads[1]["event_id"]
    assert client.payloads[0]["occurred_at"] == client.payloads[1]["occurred_at"]
    assert client.payloads[0]["event_id"] != client.payloads[2]["event_id"]
    assert len(pending) == 2


def test_memory_write_lifecycle_actions_do_not_reingest_content(tmp_path: Path) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)

    for action in ("remove", "replace"):
        instance.on_memory_write(
            action,
            "hermes-memory-42",
            "The removed project convention",
            {"session_id": "hermes-session"},
        )

    instance.shutdown()
    assert client.captures == []
    diagnostics = instance.get_lifecycle_diagnostics()
    assert [item["action"] for item in diagnostics] == ["remove", "replace"]
    assert all(item["status"] == "unresolved" for item in diagnostics)
    assert all(item["target_digest"] != "hermes-memory-42" for item in diagnostics)
    assert all("content" not in item for item in diagnostics)


def test_failed_add_capture_is_reported_and_can_retry(tmp_path: Path) -> None:
    client = FakeClient(status="rejected")
    instance = make_provider(tmp_path, client)
    payload = {"session_id": "hermes-session"}

    instance.on_memory_write("add", "hermes-memory-42", "Keep this", payload)
    assert instance._flush() is False
    assert instance.get_lifecycle_diagnostics()[0]["status"] == "capture_failed"

    client.status = "committed"
    instance.on_memory_write("add", "hermes-memory-42", "Keep this", payload)
    assert instance._flush() is True
    assert len(client.captures) == 2


def test_async_turn_capture_failure_is_reported_by_flush(tmp_path: Path) -> None:
    client = FakeClient(status="rejected")
    instance = make_provider(tmp_path, client)

    instance.sync_turn("Remember this", "Acknowledged")
    assert instance._flush() is False
    assert any(row["action"] == "capture" for row in instance.get_lifecycle_diagnostics())

    client.status = "committed"
    instance.sync_turn("Remember this", "Acknowledged")
    assert instance._flush() is True


def test_empty_remove_reports_unresolved_lifecycle_action(tmp_path: Path) -> None:
    instance = make_provider(tmp_path, FakeClient())

    instance.on_memory_write("remove", "memory", "", {"session_id": "hermes-session"})

    instance.shutdown()
    diagnostics = instance.get_lifecycle_diagnostics()
    assert diagnostics[0]["status"] == "unresolved"
    assert diagnostics[0]["action"] == "remove"


def test_failed_initialize_clears_scope_and_blocks_later_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path / "routing-review"))
    assert instance.prefetch("same query")
    client.retrieves.clear()

    with pytest.raises(ValueError, match="session cwd"):
        instance.initialize("hermes-session", cwd="")

    instance.sync_turn("must not capture", "must not capture")
    instance.on_memory_write("add", "user", "must not capture")
    assert instance.prefetch("same query") == ""
    instance.shutdown()
    assert client.captures == []
    assert client.retrieves == []


def test_reinitialize_does_not_reuse_other_projects_prefetch(tmp_path: Path) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)
    assert instance.prefetch("same query")
    first = client.retrieves[-1]["context"]["project_id"]
    other = tmp_path / "other"
    other.mkdir()
    assert instance.registry is not None
    second = instance.registry.register_project(other, user_profile_id="alice", enabled=True)
    instance.initialize("hermes-session", cwd=str(other))

    assert instance.prefetch("same query")
    assert len(client.retrieves) == 2
    assert client.retrieves[-1]["context"]["project_id"] == second.id
    assert second.id != first


def test_lifecycle_diagnostics_are_bounded_and_return_detached_values(tmp_path: Path) -> None:
    instance = make_provider(tmp_path)
    for index in range(40):
        instance.on_memory_write("remove", f"category-{index}", "private content")
    diagnostics = instance.get_lifecycle_diagnostics()
    assert len(diagnostics) == 32
    assert all(set(item) == {"action", "target_digest", "status"} for item in diagnostics)
    diagnostics[0]["status"] = "changed"
    assert instance.get_lifecycle_diagnostics()[0]["status"] == "unresolved"


def test_async_capture_exception_remains_a_reported_failure(tmp_path: Path) -> None:
    class RaisingClient(FakeClient):
        async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Unavailable")

    instance = make_provider(tmp_path, RaisingClient())
    instance.on_memory_write("add", "memory", "Keep this")
    assert instance._flush() is False
    assert instance.get_lifecycle_diagnostics()[-1]["status"] == "capture_failed"


def test_prefetch_observes_external_correction_and_forget(tmp_path: Path) -> None:
    import asyncio

    from mnemos.client import MnemosClient
    from mnemos.config import MnemosConfig, ModuleConfig
    from mnemos.engine import MnemosEngine
    from mnemos.memory_service import LocalMemoryService
    from mnemos.service import MnemosService
    from mnemos.utils.embeddings import SimpleEmbeddingProvider
    from mnemos.utils.llm import UnavailableLLMProvider
    from mnemos.utils.storage import SQLiteStore

    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    context = registry.resolve(
        tmp_path, user_profile_id="alice", session_id="external-session", host="codex"
    ).model_dump(mode="json")

    def factory() -> LocalMemoryService:
        engine = MnemosEngine(
            config=MnemosConfig(
                modules=ModuleConfig(**{name: False for name in ModuleConfig.model_fields})
            ),
            llm=UnavailableLLMProvider(),
            embedder=SimpleEmbeddingProvider(),
            store=SQLiteStore(str(tmp_path / "memory.sqlite")),
        )
        return LocalMemoryService(engine, registry)

    service = MnemosService(dispatcher_factory=factory, descriptor_path=tmp_path / "runtime.json")
    service.start()
    client = MnemosClient(descriptor_path=service.descriptor_path, auto_start=False)
    provider = MnemosHermesProvider(client=client, registry=registry, user_profile_id="alice")
    provider.initialize("hermes-session", cwd=str(tmp_path))

    def request(operation: str, **payload: Any) -> dict[str, Any]:
        return asyncio.run(
            client.request(operation, {"context": context, **payload}, queue_on_failure=False)
        )

    try:
        assert (
            request(
                "remember",
                event_id="seed",
                role="user",
                content="The project release label is cedar.",
                scope="project",
            )["status"]
            == "committed"
        )
        request("maintenance")
        fact = request("list", allowed_scopes=["project"])["facts"][0]
        query = "What is the project release label?"
        assert "cedar" in provider.prefetch(query)
        assert (
            request(
                "correct",
                fact_id=fact["id"],
                expected_version=1,
                event_id="correction",
                content="The project release label is maple.",
            )["status"]
            == "committed"
        )
        request("maintenance")
        corrected = provider.prefetch(query)
        assert "maple" in corrected
        assert "cedar" not in corrected
        assert request("forget", fact_id=fact["id"], expected_version=2)["forgotten"] is True
        assert provider.prefetch(query) == ""
    finally:
        provider.shutdown()
        asyncio.run(client.close())
        service.stop()
