"""One record the write policy refuses must not stall Claude transcript capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters import claude_code
from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.scopes import ProjectIdentity, ScopeRegistry, canonical_project_root

_MARKER = "sk-live-0000"


class _Outbox:
    def __init__(self, *, redact: bool) -> None:
        self.redact = redact

    def sanitize(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.redact:
            # Mirrors the real sanitizer rewriting secret-shaped text, which
            # makes a content-derived identity unsafe to replay.
            return {**payload, "content": payload["content"].replace(_MARKER, "[REDACTED_KEY]")}
        return dict(payload)


class _Client:
    def __init__(self, refusal: str) -> None:
        self.outbox = _Outbox(redact=refusal == "redacted")
        self.reject = refusal == "rejected"
        self.accepted: list[str] = []

    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.reject and _MARKER in payload["content"]:
            return {"schema_version": 1, "status": "rejected", "reason": "Write policy"}
        self.accepted.append(payload["source_ref"])
        return {"status": "committed", "committed": True}


def _adapter(root: Path, client: _Client) -> ClaudeCodeAdapter:
    registry = ScopeRegistry(
        projects={
            "project-a": ProjectIdentity(
                id="project-a",
                root=canonical_project_root(root),
                user_profile_id="profile-a",
                enabled=True,
            )
        }
    )
    return ClaudeCodeAdapter(
        client=client,
        registry=registry,
        user_profile_id="profile-a",
        checkpoint_directory=root / "checkpoints",
    )


def _append(transcript: Path, uuid: str, role: str, text: str, prompt_id: str) -> None:
    record = {
        "type": role,
        "uuid": uuid,
        "promptId": prompt_id,
        "message": {"role": role, "content": text},
    }
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _stop(root: Path, transcript: Path, prompt_id: str, **extra: str) -> dict[str, Any]:
    return {
        "hook_event_name": "Stop",
        "session_id": "session-a",
        "cwd": str(root),
        "transcript_path": str(transcript),
        "prompt_id": prompt_id,
        **extra,
    }


def _offset(root: Path) -> object:
    (checkpoint,) = (root / "checkpoints").glob("*.json")
    return json.loads(checkpoint.read_text(encoding="utf-8"))["offset"]


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["redacted", "rejected"])
async def test_a_refused_record_is_skipped_and_later_turns_are_captured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], refusal: str
) -> None:
    client = _Client(refusal)
    adapter = _adapter(tmp_path, client)
    transcript = tmp_path / "session.jsonl"
    _append(transcript, "u-1", "user", f"The build broke after {_MARKER} rotated, why?", "p1")
    _append(transcript, "a-1", "assistant", "Because the cache expired.", "p1")

    first = await adapter.handle("Stop", _stop(tmp_path, transcript, "p1"))

    assert (first.status, first.captured) == ("ok", 1)
    assert _offset(tmp_path) == transcript.stat().st_size
    assert "status=skipped reason=record_refused" in capsys.readouterr().err

    _append(transcript, "u-2", "user", "Benign follow-up question.", "p2")
    _append(transcript, "a-2", "assistant", "Benign answer.", "p2")
    second = await adapter.handle("Stop", _stop(tmp_path, transcript, "p2"))

    assert (second.status, second.captured) == ("ok", 2)
    assert _offset(tmp_path) == transcript.stat().st_size
    assert client.accepted == ["a-1", "u-2", "a-2"]
    assert "record_refused" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_refused_final_reply_degrades_one_stop_without_holding_the_checkpoint(
    tmp_path: Path,
) -> None:
    client = _Client("rejected")
    adapter = _adapter(tmp_path, client)
    transcript = tmp_path / "session.jsonl"
    refused_reply = f"Rotate {_MARKER} before the deploy."
    _append(transcript, "u-1", "user", "What should happen before the deploy?", "p1")
    _append(transcript, "a-1", "assistant", refused_reply, "p1")

    # The Stop fallback asks once more for the same reply and gets the same answer.
    first = await adapter.handle(
        "Stop", _stop(tmp_path, transcript, "p1", last_assistant_message=refused_reply)
    )

    assert (first.status, first.error) == ("degraded", "capture_refused")
    assert _offset(tmp_path) == transcript.stat().st_size

    _append(transcript, "u-2", "user", "And after the deploy?", "p2")
    _append(transcript, "a-2", "assistant", "Watch the error rate.", "p2")
    second = await adapter.handle(
        "Stop", _stop(tmp_path, transcript, "p2", last_assistant_message="Watch the error rate.")
    )

    assert (second.status, second.captured) == ("ok", 2)
    assert client.accepted == ["u-1", "u-2", "a-2"]


@pytest.mark.asyncio
async def test_a_first_hook_succeeds_where_hardening_a_missing_path_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def posix_protect(path: Path) -> None:
        # POSIX chmod on a missing file surfaces as PermissionError here.
        if not Path(path).exists():
            raise PermissionError(f"could not protect private path: {path}")

    monkeypatch.setattr(claude_code, "_protect_private_path", posix_protect)
    client = _Client("none")
    adapter = _adapter(tmp_path, client)
    transcript = tmp_path / "session.jsonl"
    _append(transcript, "u-1", "user", "Use the bounded test command.", "p1")

    result = await adapter.handle("Stop", _stop(tmp_path, transcript, "p1"))

    assert (result.status, result.captured) == ("ok", 1)
    assert _offset(tmp_path) == transcript.stat().st_size
