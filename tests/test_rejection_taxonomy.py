"""Requests the service will never accept come back as rejections, not outages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pydantic import ValidationError

from mnemos.client import MnemosClient
from mnemos.contracts import CaptureEvent
from mnemos.errors import (
    REQUEST_REJECTIONS,
    EventIdentityConflict,
    MnemosRejection,
    ScopeUnavailable,
    VersionConflict,
)
from mnemos.service import MnemosService
from tests.test_suppression_ordering import _remember, _service, _settle

CAPTURE = {"event_id": "unacceptable-capture", "content": "A capture the service refuses"}


def _schema_error() -> ValidationError:
    try:
        CaptureEvent.model_validate({"content": "x" * 100_001})
    except ValidationError as exc:
        return exc
    raise AssertionError("an oversized capture must fail validation")


class ScriptedDispatcher:
    def __init__(self, outcome: BaseException | dict[str, Any]) -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(operation)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return dict(self.outcome)


def _running(tmp_path: Path, dispatcher: ScriptedDispatcher) -> MnemosService:
    service = MnemosService(
        dispatcher=dispatcher,
        descriptor_path=tmp_path / "runtime.json",
        lock_path=tmp_path / "runtime.lock",
        startup_timeout=2,
        dispatch_timeout=3,
    )
    service.start()
    return service


def _client(tmp_path: Path) -> MnemosClient:
    return MnemosClient(
        descriptor_path=tmp_path / "runtime.json",
        outbox_path=tmp_path / "outbox.json",
        auto_start=False,
    )


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (_schema_error(), "invalid_request"),
        (MnemosRejection("internal detail that must not be relayed"), "invalid_request"),
        (VersionConflict("Correction version conflict"), "version_conflict"),
        (EventIdentityConflict("identity reused"), "event_identity_conflict"),
    ],
)
async def test_terminal_rejection_is_returned_and_never_queued(
    tmp_path: Path, error: Exception, code: str
) -> None:
    service = _running(tmp_path, ScriptedDispatcher(error))
    client = _client(tmp_path)
    try:
        result = await client.request("capture", CAPTURE)
        assert (result["status"], result["committed"], result["error"]) == (
            "rejected",
            False,
            code,
        )
        assert result["message"] == REQUEST_REJECTIONS[code]
        assert len(client.outbox) == 0
    finally:
        await client.close()
        service.stop()


async def test_version_conflict_is_distinct_from_an_outage(tmp_path: Path) -> None:
    service = _running(tmp_path, ScriptedDispatcher(VersionConflict("Forget version conflict")))
    client = _client(tmp_path)
    try:
        result = await client.request("forget", {"fact_id": "f", "expected_version": 1})
        assert (result["status"], result["error"]) == ("rejected", "version_conflict")
        assert "degraded" not in result
    finally:
        await client.close()
        service.stop()


async def test_unavailable_scope_keeps_capture_queued_for_later(tmp_path: Path) -> None:
    service = _running(
        tmp_path, ScriptedDispatcher(ScopeUnavailable("Project is not enabled for this user"))
    )
    client = _client(tmp_path)
    try:
        result = await client.request("capture", CAPTURE)
        assert result["status"] == "queued"
        assert "scope_unavailable" in result["error"]
        assert len(client.outbox) == 1
        replay = await client.replay_outbox()
        assert replay["pending"] == 1
    finally:
        await client.close()
        service.stop()


@pytest.mark.parametrize(
    "error", [RuntimeError("database is busy"), ValueError("unexpected internal failure")]
)
async def test_unexpected_failure_still_queues_capture(tmp_path: Path, error: Exception) -> None:
    service = _running(tmp_path, ScriptedDispatcher(error))
    client = _client(tmp_path)
    try:
        result = await client.request("capture", CAPTURE)
        assert result["status"] == "queued"
        assert len(client.outbox) == 1
    finally:
        await client.close()
        service.stop()


@pytest.mark.parametrize(
    "outcome",
    [
        {"schema_version": 1, "status": "rejected", "reason": "Capture channel is disabled"},
        EventIdentityConflict("identity reused"),
    ],
)
async def test_replay_acknowledges_rejected_entries(
    tmp_path: Path, outcome: BaseException | dict[str, Any]
) -> None:
    client = _client(tmp_path)
    client.outbox.append(dict(CAPTURE), operation="capture")
    service = _running(tmp_path, ScriptedDispatcher(outcome))
    try:
        replay = await client.replay_outbox()
        assert replay["pending"] == 0
    finally:
        await client.close()
        service.stop()


async def test_stale_versions_raise_version_conflict(tmp_path: Path) -> None:
    memory, context = _service(tmp_path)
    store = memory.engine.store
    try:
        await _remember(memory, context, "npm", "Use npm")
        await _settle(memory)
        fact = store.get_all()[0]
        with pytest.raises(VersionConflict):
            await memory.dispatch(
                "correct",
                {
                    "context": context,
                    "event_id": "stale-correction",
                    "fact_id": fact.id,
                    "expected_version": fact.version + 1,
                    "content": "Use pnpm",
                },
            )
        with pytest.raises(VersionConflict):
            await memory.dispatch(
                "forget",
                {"context": context, "fact_id": fact.id, "expected_version": fact.version + 1},
            )
    finally:
        store.close()
