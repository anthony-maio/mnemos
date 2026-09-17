"""Exercise UTF-8 hook JSON through a Windows-style text stdin wrapper."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from mnemos.adapters import claude_code, codex
from tests.test_claude_adapter import FakeClient, registry_for
from tests.test_codex_adapter import RecordingClient, make_adapter


@pytest.mark.parametrize("host", ["codex", "claude-code"])
@pytest.mark.parametrize("event", ["UserPromptSubmit", "Stop"])
def test_hook_capture_preserves_utf8_text_under_legacy_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, event: str
) -> None:
    content = "\u201cShip it\u201d in \u6771\u4eac \U0001f680"
    if host == "codex":
        client = RecordingClient()
        adapter, payload = make_adapter(tmp_path, client)
        module = codex
        monkeypatch.setattr(codex, "CodexAdapter", lambda: adapter)
    else:
        client = FakeClient()
        adapter = claude_code.ClaudeCodeAdapter(
            client=client,
            registry=registry_for(tmp_path),
            user_profile_id="profile-a",
            checkpoint_directory=tmp_path / "checkpoints",
        )
        payload = {"cwd": str(tmp_path), "session_id": "session-1", "prompt_id": "turn-1"}
        module = claude_code
        monkeypatch.setattr(claude_code, "ClaudeCodeAdapter", lambda: adapter)
    payload.update(
        hook_event_name=event,
        **{"prompt" if event == "UserPromptSubmit" else "last_assistant_message": content},
    )
    wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    stream = io.TextIOWrapper(io.BytesIO(wire), encoding="cp1252", errors="surrogateescape")
    monkeypatch.setattr(module.sys, "stdin", stream)

    assert module.main([]) == 0

    assert len(client.captures) == 1
    assert client.captures[0]["content"] == content
    assert client.captures[0]["role"] == ("user" if event == "UserPromptSubmit" else "assistant")


def test_parsed_stop_error_still_returns_nonblocking_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fail(*args, **kwargs):
        raise ValueError("private payload must not appear")

    monkeypatch.setattr(codex, "run_hook", fail)
    monkeypatch.setattr(codex.sys, "stdin", io.StringIO('{"hook_event_name":"Stop"}'))

    assert codex.main([]) == 0

    result = capsys.readouterr()
    assert json.loads(result.out) == {"continue": True}
    assert "event=Stop status=error exception=ValueError" in result.err
    assert "private payload" not in result.err
