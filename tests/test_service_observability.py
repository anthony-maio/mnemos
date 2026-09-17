"""Focused tests for the managed service's private structured log."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sqlite3
import time

import pytest

from mnemos.maintenance import MaintenanceScheduler
from mnemos import observability
from mnemos.observability import log_event
from mnemos.service import MnemosService
from mnemos.client import MnemosClient
from mnemos.scopes import ScopeRegistry


class _Dispatcher:
    async def dispatch(self, operation: str, payload: dict) -> dict:
        return {"status": "committed", "operation": operation}


class _BrokenDispatcher:
    async def dispatch(self, operation: str, payload: dict) -> dict:
        raise sqlite3.OperationalError("database-secret-must-not-be-written")


class _Maintenance:
    def __init__(self) -> None:
        self.errors: list[BaseException] = []
        self.on_error = self.errors.append


class _MaintenanceDispatcher(_Dispatcher):
    def __init__(self) -> None:
        self.maintenance = _Maintenance()


class _FailingMaintenanceStore:
    def job_summary(self, *, now: object) -> dict[str, int]:
        raise sqlite3.OperationalError("maintenance-store-secret")


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _start(tmp_path: Path, log_path: Path | None = None) -> MnemosService:
    service = MnemosService(
        dispatcher=_Dispatcher(),
        descriptor_path=tmp_path / "runtime.json",
        service_log_path=log_path,
        startup_timeout=2,
        dispatch_timeout=2,
    )
    service.start()
    return service


def test_private_log_starts_after_descriptor_and_whitelists_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime" / "service.log"
    logger = logging.getLogger("mnemos")
    prior_level = logger.level
    service = _start(tmp_path, log_path)
    try:
        assert service.descriptor is not None
        log_event(
            "mnemos.provider_failure",
            provider="sqlite",
            operation="dispatch",
            error_type="OperationalError",
            error="secret-content",
            content="secret-content",
        )
        log_event(
            "mnemos.retrieval_latency",
            latency_ms=3.5,
            result_count=1,
            top_k=5,
            scope="project",
            scope_id="project-a",
            allowed_scopes="project,global",
            store_backend="SQLiteStore",
            query="secret-query",
            path="secret-path",
        )
    finally:
        service.stop()
    records = _read_log(log_path)
    if os.name != "nt":
        assert log_path.stat().st_mode & 0o777 == 0o600
    assert records[0]["event"] == "mnemos.startup"
    assert records[0]["status"] == "running"
    assert "token" not in records[0]
    failure = next(record for record in records if record["event"] == "mnemos.provider_failure")
    assert failure["error_type"] == "OperationalError"
    assert failure["event"] == "mnemos.provider_failure"
    assert failure["operation"] == "dispatch"
    assert failure["provider"] == "sqlite"
    assert set(failure) == {"timestamp", "event", "provider", "operation", "error_type"}
    latency = next(record for record in records if record["event"] == "mnemos.retrieval_latency")
    assert "query" not in latency and "path" not in latency and "scope_id" not in latency
    assert logger.level == prior_level


def test_injected_dispatcher_does_not_create_log_by_default(tmp_path: Path) -> None:
    service = _start(tmp_path)
    try:
        assert not (tmp_path / "service.log").exists()
    finally:
        service.stop()


def test_dispatch_sqlite_failure_is_logged_without_exception_text(tmp_path: Path) -> None:
    log_path = tmp_path / "service.log"
    service = MnemosService(
        dispatcher=_BrokenDispatcher(),
        descriptor_path=tmp_path / "runtime.json",
        service_log_path=log_path,
        startup_timeout=2,
        dispatch_timeout=2,
    )
    service.start()
    try:
        with pytest.raises(sqlite3.OperationalError, match="database-secret"):
            service.submit("retrieve", {"query": "secret-query"})
    finally:
        service.stop()
    text = log_path.read_text(encoding="utf-8")
    assert "database-secret" not in text
    failure = next(
        record for record in _read_log(log_path) if record["event"] == "mnemos.provider_failure"
    )
    assert failure["provider"] == "sqlite"
    assert failure["operation"] == "dispatch"
    assert failure["error_type"] == "OperationalError"


def test_private_log_rotation_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "SERVICE_LOG_MAX_BYTES", 256)
    log_path = tmp_path / "service.log"
    service = _start(tmp_path, log_path)
    try:
        for _ in range(100):
            log_event(
                "mnemos.startup",
                status="running",
                backend="SQLiteStore",
                provider="MockLLMProvider",
            )
    finally:
        service.stop()
    logs = sorted(tmp_path.glob("service.log*"))
    assert len(logs) <= observability.SERVICE_LOG_BACKUPS + 1
    assert all(path.stat().st_size > 0 for path in logs)


def test_protection_failure_releases_lock_for_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mnemos.service as service_module

    original = service_module._protect_private_path

    def fail_log(path: Path) -> None:
        if path.name == "service.log":
            raise PermissionError("injected protection failure")
        original(path)

    monkeypatch.setattr(service_module, "_protect_private_path", fail_log)
    failed = MnemosService(
        dispatcher=_Dispatcher(),
        descriptor_path=tmp_path / "runtime.json",
        service_log_path=tmp_path / "service.log",
        startup_timeout=2,
    )
    with pytest.raises(PermissionError, match="injected protection"):
        failed.start()
    assert failed._owner_lock._handle is None

    monkeypatch.setattr(service_module, "_protect_private_path", original)
    replacement = _start(tmp_path, tmp_path / "service.log")
    replacement.stop()


def test_failed_start_retry_waits_for_new_owner_initialization(tmp_path: Path) -> None:
    calls = 0

    def factory() -> _Dispatcher:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first initialization failure")
        time.sleep(0.1)
        return _Dispatcher()

    service = MnemosService(
        dispatcher_factory=factory,
        descriptor_path=tmp_path / "runtime.json",
        startup_timeout=0.02,
        shutdown_timeout=1,
    )
    with pytest.raises(RuntimeError, match="failed to initialize"):
        service.start()
    with pytest.raises(TimeoutError, match="did not initialize"):
        service.start()
    assert calls == 2


def test_only_one_explicit_service_log_binding_is_active(tmp_path: Path) -> None:
    first = _start(tmp_path / "first", tmp_path / "first" / "service.log")
    second = MnemosService(
        dispatcher=_Dispatcher(),
        descriptor_path=tmp_path / "second" / "runtime.json",
        service_log_path=tmp_path / "second" / "service.log",
        startup_timeout=2,
    )
    try:
        with pytest.raises(RuntimeError, match="service log is already active"):
            second.start()
        log_event("mnemos.startup", status="running", backend="SQLiteStore", provider="test")
        assert len(_read_log(tmp_path / "first" / "service.log")) >= 2
    finally:
        second.stop()
        first.stop()


def test_unlogged_injected_service_cannot_write_another_service_log(tmp_path: Path) -> None:
    first = _start(tmp_path / "first", tmp_path / "first" / "service.log")
    second = _start(tmp_path / "second")
    try:
        first_records = _read_log(tmp_path / "first" / "service.log")
        assert [record["event"] for record in first_records] == ["mnemos.startup"]
    finally:
        second.stop()
        first.stop()


def test_maintenance_observer_logs_and_restores_previous_callback(tmp_path: Path) -> None:
    dispatcher = _MaintenanceDispatcher()
    previous = dispatcher.maintenance.on_error
    log_path = tmp_path / "service.log"
    service = MnemosService(
        dispatcher=dispatcher,
        descriptor_path=tmp_path / "runtime.json",
        service_log_path=log_path,
        startup_timeout=2,
    )
    service.start()
    try:
        assert dispatcher.maintenance.on_error is not previous
        error = sqlite3.OperationalError("maintenance-secret")
        dispatcher.maintenance.on_error(error)
        assert dispatcher.maintenance.errors == [error]
    finally:
        service.stop()
    assert dispatcher.maintenance.on_error is previous
    records = _read_log(log_path)
    failure = next(record for record in records if record["event"] == "mnemos.provider_failure")
    assert failure["provider"] == "sqlite"
    assert failure["operation"] == "maintenance"
    assert failure["error_type"] == "OperationalError"
    assert "maintenance-secret" not in log_path.read_text(encoding="utf-8")


def test_scheduler_outer_failure_reaches_private_log(tmp_path: Path) -> None:
    dispatcher = _Dispatcher()
    dispatcher.maintenance = MaintenanceScheduler(
        _FailingMaintenanceStore(),
        lambda job: None,
        interval_seconds=0.01,
        idle_seconds=0,
    )
    log_path = tmp_path / "service.log"
    service = MnemosService(
        dispatcher=dispatcher,
        descriptor_path=tmp_path / "runtime.json",
        service_log_path=log_path,
        startup_timeout=2,
    )
    service.start()

    async def exercise() -> None:
        task = asyncio.create_task(dispatcher.maintenance.run())
        await asyncio.sleep(0.03)
        dispatcher.maintenance.stop()
        await task

    try:
        asyncio.run(exercise())
    finally:
        service.stop()
    records = _read_log(log_path)
    failure = next(record for record in records if record["event"] == "mnemos.provider_failure")
    assert failure["provider"] == "sqlite"
    assert failure["operation"] == "maintenance"
    assert failure["error_type"] == "OperationalError"
    assert "maintenance-store-secret" not in log_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_autostarted_service_logs_real_retrieval_without_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in list(os.environ):
        if key.startswith("MNEMOS_"):
            monkeypatch.delenv(key)
    project = tmp_path / "project"
    project.mkdir()
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="log-fixture", enabled=True)
    registry_path = tmp_path / "scopes.json"
    registry.save(registry_path)
    context = registry.resolve(
        project, user_profile_id="log-fixture", session_id="log-session", host="manual"
    ).model_dump(mode="json")
    config = tmp_path / "config.toml"
    config.write_text(
        '[embedding]\nprovider = "simple"\ndim = 64\n'
        "[modules]\nsurprisal = false\nmutable_rag = false\n"
        "affective = false\nsleep = false\nspreading = false\n",
        encoding="utf-8",
    )
    for key, value in {
        "MNEMOS_CONFIG_PATH": str(config),
        "MNEMOS_SQLITE_PATH": str(tmp_path / "memory.sqlite"),
        "MNEMOS_STORE_TYPE": "sqlite",
        "MNEMOS_SCOPE_REGISTRY": str(registry_path),
        "MNEMOS_RUNTIME_DIR": str(tmp_path / "runtime"),
    }.items():
        monkeypatch.setenv(key, value)
    descriptor = tmp_path / "runtime" / "descriptor.json"
    client = MnemosClient(descriptor_path=descriptor, startup_timeout=10)
    content = "The project logging fixture uses cobalt notebooks."
    try:
        health = await client.request("health", {})
        assert health["status"] == "running"
        assert client._child is not None and client._child.poll() is None
        assert client._child.stdout is None and client._child.stderr is None
        await client.request(
            "remember", {"context": context, "content": content, "event_id": "log-fact"}
        )
        await client.request("maintenance", {})
        result = await client.request(
            "retrieve", {"context": context, "query": content, "allowed_scopes": ["project"]}
        )
        assert any(fact["content"] == content for fact in result["facts"])
    finally:
        client.auto_start = False
        try:
            if descriptor.exists():
                await client.request("shutdown", {})
        finally:
            spawned = client._child
            await client.close()
            if spawned is not None:
                for _ in range(100):
                    if spawned.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                if spawned.poll() is None:
                    spawned.terminate()
                    spawned.wait(timeout=5)
    assert spawned is not None and spawned.returncode == 0
    assert not descriptor.exists()
    log_path = descriptor.with_name("service.log")
    records = _read_log(log_path)
    startups = [record for record in records if record["event"] == "mnemos.startup"]
    assert len(startups) == 1
    assert startups[0]["backend"] == "SQLiteStore"
    assert startups[0]["provider"] == "UnavailableLLMProvider"
    assert any(
        record["event"] == "mnemos.retrieval_latency" and record["result_count"] == 1
        for record in records
    )
    text = log_path.read_text(encoding="utf-8")
    assert content not in text and context["project_id"] not in text
    assert str(tmp_path) not in text
