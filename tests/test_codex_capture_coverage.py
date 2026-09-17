"""Missing host text must not be reported as successful capture coverage."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from mnemos.adapters import codex
from tests.test_codex_adapter import RecordingClient, make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,field", [("Stop", "last_assistant_message"), ("UserPromptSubmit", "prompt")]
)
@pytest.mark.parametrize("fields", [{}, {"value": ""}, {"value": " \n "}, {"value": []}])
async def test_missing_capture_text_reports_degraded_without_writing(
    tmp_path: Path, event: str, field: str, fields: dict
) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)
    if "value" in fields:
        payload[field] = fields["value"]

    result = await adapter.handle(event, payload)

    assert result.status == "degraded"
    assert result.degraded is True
    assert result.captured == 0
    assert result.event_ids == ()
    assert client.captures == []
    if event == "Stop":
        assert codex.hook_output(event, result) == {"continue": True}


def test_missing_stop_text_emits_diagnostic_and_stays_nonblocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter, payload = make_adapter(tmp_path)
    payload["hook_event_name"] = "Stop"
    monkeypatch.setattr(codex, "CodexAdapter", lambda: adapter)
    monkeypatch.setattr(codex.sys, "stdin", io.StringIO(json.dumps(payload)))

    assert codex.main([]) == 0

    result = capsys.readouterr()
    assert json.loads(result.out) == {"continue": True}
    assert "host=codex event=Stop status=degraded" in result.err
    assert str(tmp_path) not in result.err
    assert "session-1" not in result.err


@pytest.mark.asyncio
async def test_stop_reentry_without_text_remains_intentionally_skipped(tmp_path: Path) -> None:
    client = RecordingClient()
    adapter, payload = make_adapter(tmp_path, client)

    result = await adapter.handle("Stop", {**payload, "stop_hook_active": True})

    assert result.status == "skipped"
    assert result.degraded is False
    assert client.captures == []
