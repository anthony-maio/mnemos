"""Competing launchers must wait for an authenticated, ready owner."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
import socket
import threading
import time

import pytest

from mnemos.service import MnemosService, _ProcessLock, _write_private_json


class Dispatcher:
    async def dispatch(self, operation: str, payload: dict) -> dict:
        raise AssertionError("Startup readiness must not dispatch memory operations")


def test_competing_launcher_waits_past_dead_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    initializing = threading.Event()
    release = threading.Event()

    def delayed_dispatcher() -> Dispatcher:
        initializing.set()
        assert release.wait(5), "Test did not release owner initialization"
        return Dispatcher()

    winner = MnemosService(
        dispatcher_factory=delayed_dispatcher, descriptor_path=path, startup_timeout=3
    )
    loser = MnemosService(dispatcher=Dispatcher(), descriptor_path=path, startup_timeout=2)
    with socket.socket() as dead_port, ThreadPoolExecutor(max_workers=2) as pool:
        dead_port.bind(("127.0.0.1", 0))
        stale = {
            "schema_version": 1,
            "host": "127.0.0.1",
            "port": dead_port.getsockname()[1],
            "pid": 123456,
            "token": "dead-test-owner",
        }
        _write_private_json(path, stale)
        first = pool.submit(winner.start)
        second = None
        try:
            assert initializing.wait(2)
            second = pool.submit(loser.start)
            with pytest.raises(FutureTimeout):
                second.result(timeout=0.2)
            release.set()
            owner = first.result(timeout=3)
            assert second.result(timeout=3) == owner
            assert owner != stale
            assert winner.owner_thread_id is not None
            assert loser.owner_thread_id is None
        finally:
            release.set()
            first.result(timeout=4)
            if second is not None:
                second.result(timeout=4)
            loser.stop()
            winner.stop()


def test_dead_descriptor_with_held_lock_times_out_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    lock = _ProcessLock(path.with_suffix(".lock"))
    assert lock.acquire()
    loser = MnemosService(dispatcher=Dispatcher(), descriptor_path=path, startup_timeout=0.15)
    with socket.socket() as dead_port:
        dead_port.bind(("127.0.0.1", 0))
        _write_private_json(
            path,
            {
                "schema_version": 1,
                "host": "127.0.0.1",
                "port": dead_port.getsockname()[1],
                "pid": 123456,
                "token": "dead-test-owner",
            },
        )
        original = path.read_bytes()
        started = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="not ready before startup timeout"):
                loser.start()
            assert 0.1 <= time.monotonic() - started < 1
            assert path.read_bytes() == original
            assert loser.descriptor is None and loser.owner_thread_id is None
        finally:
            loser.stop()
            lock.release()
        assert path.read_bytes() == original


@pytest.mark.parametrize("invalid", ["token", "schema", "status"])
def test_competing_launcher_rejects_unverified_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    path = tmp_path / "runtime.json"
    winner = MnemosService(dispatcher=Dispatcher(), descriptor_path=path)
    loser = MnemosService(dispatcher=Dispatcher(), descriptor_path=path, startup_timeout=0.5)
    owner = winner.start()
    if invalid == "token":
        _write_private_json(path, {**owner, "token": "wrong-token"})
    else:
        monkeypatch.setattr(
            winner,
            "health",
            lambda: {
                "schema_version": True if invalid == "schema" else 1,
                "status": "starting" if invalid == "status" else "ok",
            },
        )
    original = path.read_bytes()
    try:
        with pytest.raises(RuntimeError, match="readiness"):
            loser.start()
        assert loser.descriptor is None and loser.owner_thread_id is None
        assert path.read_bytes() == original
        loser.stop()
        assert path.read_bytes() == original
    finally:
        # Restore the owner's descriptor so normal owner cleanup can remove it.
        _write_private_json(path, owner)
        loser.stop()
        winner.stop()
