"""Hermes turn capture sends each row once, on one worker, with one ledger write."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

import mnemos.adapters.hermes as hermes_adapter
from mnemos.adapters.hermes import MnemosHermesProvider
from mnemos.errors import ScopeUnavailable
from mnemos.scopes import ScopeRegistry


class SlowClient:
    """Records captures and how many were ever in flight at once."""

    def __init__(self) -> None:
        self.captures: list[dict[str, Any]] = []
        self.in_flight = 0
        self.most_in_flight = 0
        self.lock = threading.Lock()

    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        await asyncio.sleep(0.005)
        with self.lock:
            self.in_flight -= 1
            self.captures.append(payload)
        return {"status": "committed", "committed": True}

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"facts": []}


def _provider(tmp_path: Path, client: SlowClient) -> MnemosHermesProvider:
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    provider = MnemosHermesProvider(client=client, registry=registry, user_profile_id="alice")
    provider.initialize("session", hermes_home=str(tmp_path / "hermes"), cwd=str(tmp_path))
    return provider


def _history(rows: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "user" if row % 2 else "assistant",
            "content": f"Message {row}.",
            "_row_id": row,
        }
        for row in range(1, rows + 1)
    ]


def test_each_turn_sends_only_rows_not_yet_delivered(tmp_path: Path) -> None:
    client = SlowClient()
    provider = _provider(tmp_path, client)

    provider.sync_turn("", "", messages=_history(4))
    assert provider._flush()
    provider.sync_turn("", "", messages=_history(6))
    assert provider._flush()
    provider.on_session_end(_history(6))

    refs = [payload["source_ref"] for payload in client.captures]
    assert refs == [f"transcript:v2:{row}" for row in range(1, 7)]


def test_one_worker_delivers_turns_in_order(tmp_path: Path) -> None:
    client = SlowClient()
    provider = _provider(tmp_path, client)

    # Back to back turns used to start a thread each, all resending history.
    for rows in range(2, 21, 2):
        provider.sync_turn("", "", messages=_history(rows))
    assert provider._flush(timeout=10)

    assert client.most_in_flight == 1
    assert [payload["source_ref"] for payload in client.captures] == [
        f"transcript:v2:{row}" for row in range(1, 21)
    ]


def test_a_long_unseen_history_is_recorded_in_one_ledger_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = hermes_adapter.occurrence_times

    def counting(home: str, observed: dict[str, str]) -> dict[str, str]:
        calls.append(len(observed))
        return real(home, observed)

    monkeypatch.setattr(hermes_adapter, "occurrence_times", counting)
    client = SlowClient()
    provider = _provider(tmp_path, client)

    provider.sync_turn("", "", messages=_history(60))
    assert provider._flush(timeout=10)

    assert calls == [60]
    times = [payload["occurred_at"] for payload in client.captures]
    assert times == sorted(times) and len(set(times)) == 60


def test_pre_compress_does_not_resend_delivered_rows(tmp_path: Path) -> None:
    client = SlowClient()
    provider = _provider(tmp_path, client)
    provider.sync_turn("", "", messages=_history(4))
    assert provider._flush()

    result = provider.on_pre_compress(_history(5), require_checkpoint=True)

    assert "captured 5" in result
    assert [payload["source_ref"] for payload in client.captures][-1] == "transcript:v2:5"
    assert len(client.captures) == 5


def test_reinitializing_for_another_profile_does_not_keep_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MNEMOS_USER_PROFILE_ID", raising=False)
    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path / "routing-review"))
    registry = ScopeRegistry()
    registry.register_project(tmp_path, user_profile_id="alice", enabled=True)
    client = SlowClient()
    provider = MnemosHermesProvider(client=client, registry=registry)
    provider.initialize(
        "s-alice", hermes_home=str(tmp_path / "alice"), cwd=str(tmp_path), agent_identity="alice"
    )
    assert provider.user_profile_id == "alice"

    # The project belongs to alice, so bob's session must fail closed rather
    # than be routed into alice's memory.
    with pytest.raises(ScopeUnavailable):
        provider.initialize(
            "s-bob", hermes_home=str(tmp_path / "bob"), cwd=str(tmp_path), agent_identity="bob"
        )
    assert provider.user_profile_id == "bob"
    provider.sync_turn("bob's question", "bob's answer", session_id="s-bob")
    provider.shutdown()
    assert client.captures == []
