"""Claude transcript capture keeps moving: refusals, backlogs, rewrites, new sessions."""

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
    def __init__(self, refusal: str, *, fail_once: frozenset[str] = frozenset()) -> None:
        self.outbox = _Outbox(redact=refusal == "redacted")
        self.reject = refusal == "rejected"
        self.fail_once = set(fail_once)
        self.accepted: list[str] = []

    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.reject and _MARKER in payload["content"]:
            return {"schema_version": 1, "status": "rejected", "reason": "Write policy"}
        if payload["source_ref"] in self.fail_once:
            self.fail_once.discard(payload["source_ref"])
            return {"status": "degraded", "committed": False}
        self.accepted.append(payload["source_ref"])
        return {"status": "committed", "committed": True}

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"facts": []}


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
async def test_a_backlog_past_the_time_budget_is_finished_by_later_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With no budget, each hook captures exactly one record and saves its place.
    monkeypatch.setattr(claude_code, "TRANSCRIPT_CAPTURE_BUDGET_SECONDS", 0.0)
    client = _Client("none")
    adapter = _adapter(tmp_path, client)
    transcript = tmp_path / "session.jsonl"
    _append(transcript, "u-1", "user", "First question.", "p1")
    _append(transcript, "a-1", "assistant", "First answer.", "p1")
    _append(transcript, "u-2", "user", "Second question.", "p2")
    payload = _stop(tmp_path, transcript, "p2", last_assistant_message="Second answer.")

    results = [await adapter.handle("Stop", payload) for _ in range(3)]

    assert [(result.status, result.error, result.captured) for result in results] == [
        ("degraded", "backlog_deferred", 1),
        ("degraded", "backlog_deferred", 1),
        ("ok", None, 2),
    ]
    # Only the hook that reaches the end falls back to the host's copy of the
    # reply, which the transcript does not hold yet.
    assert client.accepted == ["u-1", "a-1", "u-2", "p2"]
    assert _offset(tmp_path) == transcript.stat().st_size


@pytest.mark.asyncio
async def test_a_hook_that_fails_midway_keeps_the_progress_it_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_code, "CHECKPOINT_INTERVAL_SECONDS", 0.0)
    client = _Client("none", fail_once=frozenset({"u-2"}))
    adapter = _adapter(tmp_path, client)
    transcript = tmp_path / "session.jsonl"
    _append(transcript, "u-1", "user", "First question.", "p1")
    _append(transcript, "a-1", "assistant", "First answer.", "p1")
    _append(transcript, "u-2", "user", "Second question.", "p2")
    _append(transcript, "a-2", "assistant", "Second answer.", "p2")

    first = await adapter.handle("Stop", _stop(tmp_path, transcript, "p2"))
    second = await adapter.handle("Stop", _stop(tmp_path, transcript, "p2"))

    assert (first.status, first.error, first.captured) == (
        "degraded",
        "capture_not_acknowledged",
        2,
    )
    assert (second.status, second.captured) == ("ok", 2)
    # The retry starts at the failed record, not at the top of the batch.
    assert client.accepted == ["u-1", "a-1", "u-2", "a-2"]


@pytest.mark.asyncio
async def test_a_transcript_rewritten_shorter_in_place_keeps_capturing(tmp_path: Path) -> None:
    client = _Client("none")
    adapter = _adapter(tmp_path, client)
    transcript = tmp_path / "session.jsonl"
    header = json.dumps({"type": "summary", "summary": "session header", "leafUuid": "x"}) + "\n"
    transcript.write_text(header, encoding="utf-8")
    _append(transcript, "u-1", "user", "First question " + "x" * 400, "p1")
    _append(transcript, "a-1", "assistant", "First answer " + "y" * 400, "p1")
    first = await adapter.handle("Stop", _stop(tmp_path, transcript, "p1"))
    assert (first.status, first.captured) == ("ok", 2)

    # Same file and first line, so the same transcript identity, but shorter.
    with transcript.open("r+", encoding="utf-8") as handle:
        handle.write(header)
        handle.truncate()
    _append(transcript, "u-2", "user", "Retry.", "p2")
    second = await adapter.handle("Stop", _stop(tmp_path, transcript, "p2"))
    _append(transcript, "u-3", "user", "Next question.", "p3")
    prompt = await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "Next question.",
            "prompt_id": "p3",
        },
    )
    third = await adapter.handle("Stop", _stop(tmp_path, transcript, "p3"))

    assert (second.status, second.captured) == ("ok", 1)
    assert (prompt.status, prompt.captured) == ("ok", 1)
    assert third.status == "ok"
    assert _offset(tmp_path) == transcript.stat().st_size
    assert client.accepted == ["u-1", "a-1", "u-2", "p3"]


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
