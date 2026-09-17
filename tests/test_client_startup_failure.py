"""Deterministic client startup diagnostics for exited and stalled children."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mnemos.client as client_module
from mnemos.client import MnemosClient


class FakeProcess:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.poll_calls = 0
        self.terminate_calls = 0

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode if self.returncode is not None else 0


class VirtualLoop:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


def patch_virtual_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[VirtualLoop, list[float]]:
    loop = VirtualLoop()
    sleep_calls: list[float] = []

    async def virtual_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        loop.now += delay

    monkeypatch.setattr(
        client_module,
        "asyncio",
        SimpleNamespace(get_running_loop=lambda: loop, sleep=virtual_sleep),
    )
    return loop, sleep_calls


def descriptor() -> dict[str, Any]:
    return {"host": "127.0.0.1", "port": 43123, "token": "test-token"}


@pytest.mark.asyncio
async def test_exited_autostart_child_reports_exit_code_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(returncode=17)
    monkeypatch.setattr(client_module.subprocess, "Popen", lambda *a, **k: child)
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        auto_start=True,
        startup_timeout=0.15,
        request_timeout=1,
    )
    monkeypatch.setattr(client, "_descriptor", lambda: None)
    loop, sleep_calls = patch_virtual_startup(monkeypatch)

    result = await client.retrieve({"query": "startup failure"})

    assert result["status"] == "degraded"
    assert "exited" in str(result["error"])
    assert "17" in str(result["error"])
    assert sleep_calls == []
    assert loop.now < 0.15
    assert child.terminate_calls == 0


@pytest.mark.asyncio
async def test_exited_autostart_child_capture_still_queues_with_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(returncode=23)
    monkeypatch.setattr(client_module.subprocess, "Popen", lambda *a, **k: child)
    outbox = tmp_path / "outbox.json"
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        outbox_path=outbox,
        auto_start=True,
        startup_timeout=0.15,
        request_timeout=1,
    )
    monkeypatch.setattr(client, "_descriptor", lambda: None)
    _loop, sleep_calls = patch_virtual_startup(monkeypatch)

    result = await client.capture(
        {"event_id": "startup-failure-capture", "content": "durable startup capture"}
    )

    assert result["status"] == "queued"
    assert result["committed"] is False
    assert "exited" in str(result["error"])
    assert "23" in str(result["error"])
    assert len(client.outbox.pending()) == 1
    assert sleep_calls == []
    assert child.terminate_calls == 0


@pytest.mark.asyncio
async def test_live_child_without_descriptor_reports_startup_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(returncode=None)
    monkeypatch.setattr(client_module.subprocess, "Popen", lambda *a, **k: child)
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        auto_start=True,
        startup_timeout=0.15,
        request_timeout=1,
    )
    monkeypatch.setattr(client, "_descriptor", lambda: None)
    loop, sleep_calls = patch_virtual_startup(monkeypatch)

    result = await client.retrieve({"query": "startup timeout"})

    assert result["status"] == "degraded"
    assert "timed out" in str(result["error"])
    assert sleep_calls
    assert loop.now >= 0.15
    assert child.terminate_calls == 0


@pytest.mark.asyncio
async def test_existing_owner_descriptor_wins_over_exited_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(returncode=29)
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        auto_start=True,
        startup_timeout=0.15,
        request_timeout=1,
    )
    client._child = child
    client._autostart_attempted = True
    monkeypatch.setattr(client, "_descriptor", descriptor)

    async def dispatch(
        operation: str, payload: dict[str, Any], owner: dict[str, Any]
    ) -> dict[str, Any]:
        assert operation == "retrieve"
        assert payload == {"query": "owner"}
        assert owner == descriptor()
        return {"status": "committed", "committed": True}

    monkeypatch.setattr(client, "_dispatch_http", dispatch)
    client._outbox_drained = True

    result = await client.retrieve({"query": "owner"})

    assert result["committed"] is True
    assert child.terminate_calls == 0


@pytest.mark.asyncio
async def test_live_child_becoming_ready_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(returncode=None)
    monkeypatch.setattr(client_module.subprocess, "Popen", lambda *a, **k: child)
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        auto_start=True,
        startup_timeout=0.15,
        request_timeout=1,
    )
    client._child = child
    descriptors: list[dict[str, Any] | None] = [None, descriptor()]

    def becomes_ready() -> dict[str, Any] | None:
        return descriptors.pop(0) if descriptors else descriptor()

    monkeypatch.setattr(client, "_descriptor", becomes_ready)
    _loop, sleep_calls = patch_virtual_startup(monkeypatch)

    result = await client._wait_for_descriptor()

    assert result == descriptor()
    assert child.terminate_calls == 0
    assert sleep_calls == [0.05]


@pytest.mark.asyncio
async def test_descriptor_appearing_during_terminal_poll_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, dict[str, Any] | None] = {"descriptor": None}

    class RaceProcess(FakeProcess):
        def poll(self) -> int | None:
            self.poll_calls += 1
            state["descriptor"] = descriptor()
            return 31

    child = RaceProcess(returncode=31)
    client = MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        auto_start=True,
        startup_timeout=0.15,
        request_timeout=1,
    )
    client._child = child

    def read_descriptor() -> dict[str, Any] | None:
        return state["descriptor"]

    monkeypatch.setattr(client, "_descriptor", read_descriptor)
    loop = VirtualLoop()

    async def virtual_sleep(delay: float) -> None:
        loop.now += delay
        state["descriptor"] = descriptor()

    monkeypatch.setattr(
        client_module,
        "asyncio",
        SimpleNamespace(get_running_loop=lambda: loop, sleep=virtual_sleep),
    )

    result = await client._wait_for_descriptor()

    assert result == descriptor()
    assert child.terminate_calls == 0
