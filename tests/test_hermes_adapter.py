from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.hermes import (
    HERMES_API_AVAILABLE,
    HermesMemoryProvider,
    MnemosHermesProvider,
    capabilities,
)
from mnemos.scopes import ScopeRegistry


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


def test_session_switch_rebinds_context_and_cache(tmp_path: Path) -> None:
    instance = make_provider(tmp_path)
    instance.prefetch("a useful project query")
    instance.on_session_switch("hermes-session-2", reset=True)
    assert instance.session_id == "hermes-session-2"
    assert instance._context is not None
    assert instance._context.session_id == "hermes-session-2"
    assert instance._prefetch_cache == {}


def test_prefetch_is_bounded_cached_and_scoped(tmp_path: Path) -> None:
    client = FakeClient()
    instance = make_provider(tmp_path, client)
    assert instance.prefetch("What local convention applies?") == "1. remembered local convention"
    assert instance.prefetch("What local convention applies?") == "1. remembered local convention"
    assert len(client.retrieves) == 1
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
