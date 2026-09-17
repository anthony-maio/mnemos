from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter, _native_output
from mnemos.scopes import ProjectIdentity, ScopeRegistry, canonical_project_root


class FakeClient:
    def __init__(self, *, capture_status: str = "committed") -> None:
        self.capture_status = capture_status
        self.captures: list[dict] = []
        self.retrievals: list[dict] = []

    async def capture(self, payload: dict) -> dict:
        self.captures.append(payload)
        return {
            "status": self.capture_status,
            "committed": self.capture_status == "committed",
        }

    async def retrieve(self, payload: dict) -> dict:
        self.retrievals.append(payload)
        return {"facts": [{"content": "Use the repository's bounded test command."}]}


@pytest.mark.asyncio
async def test_prompt_is_captured_before_failing_recall(tmp_path: Path) -> None:
    class FailedRecall(FakeClient):
        async def retrieve(self, payload: dict) -> dict:
            assert len(self.captures) == 1
            raise TimeoutError("embedding unavailable")

    client = FailedRecall()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    result = await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "Keep this durable fact",
        },
    )
    assert result.captured == 1
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_disabled_recall_keeps_prompt_capture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MNEMOS_AUTO_RETRIEVE", "false")
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    result = await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "Keep this durable fact",
        },
    )
    assert result.captured == 1
    assert client.retrievals == []


@pytest.mark.asyncio
async def test_rejected_prompt_capture_is_degraded_but_keeps_recall_context(
    tmp_path: Path,
) -> None:
    client = FakeClient(capture_status="degraded")
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    result = await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "Keep this durable fact",
        },
    )

    assert result.status == "degraded"
    assert result.committed is False
    assert result.captured == 0
    assert result.error == "capture_not_acknowledged"
    assert "bounded test command" in result.context


def registry_for(path: Path) -> ScopeRegistry:
    return ScopeRegistry(
        projects={
            "project-a": ProjectIdentity(
                id="project-a",
                root=canonical_project_root(path),
                user_profile_id="profile-a",
                enabled=True,
            )
        }
    )


def hook_payload(tmp_path: Path, transcript: Path) -> dict:
    return {
        "session_id": "session-a",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path),
        "hook_event_name": "Stop",
        "last_assistant_message": "The work is complete.",
    }


@pytest.mark.asyncio
async def test_transcript_deltas_are_role_tagged_checkpointed_and_idempotent(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "u1",
                        "message": {"role": "user", "content": "Please use SQLite."},
                    }
                ),
                "malformed",
                json.dumps({"type": "system", "message": {"role": "system", "content": "ignore"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "a1",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "I will use SQLite."}],
                        },
                    }
                ),
                json.dumps(
                    {"type": "tool_result", "message": {"role": "tool", "content": "ignore"}}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    result = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    assert result.captured == 2
    assert [item["role"] for item in client.captures] == ["user", "assistant"]
    assert (tmp_path / "checkpoints").exists()

    again = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    assert again.captured == 0
    assert len(client.captures) == 2

    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u2",
                    "message": {"role": "user", "content": "Add the migration note."},
                }
            )
            + "\n"
        )
    appended = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    assert appended.captured == 1
    assert len(client.captures) == 3


@pytest.mark.asyncio
async def test_checkpoint_waits_for_durable_or_outbox_ack(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "u1",
                "message": {"role": "user", "content": "A durable project convention."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient(capture_status="degraded")
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    first = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    assert first.captured == 0
    assert not (tmp_path / "checkpoints").exists()

    client.capture_status = "queued"
    second = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    assert second.captured == 1
    assert (tmp_path / "checkpoints").exists()


@pytest.mark.asyncio
async def test_rotation_restarts_offset_and_stop_deduplicates_final_message(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    rotated = tmp_path / "rotated.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "a1",
                "message": {"role": "assistant", "content": "First response."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    rotated.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "a2",
                "message": {"role": "assistant", "content": "Second response."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rotated.replace(transcript)
    payload = hook_payload(tmp_path, transcript)
    payload["last_assistant_message"] = "Second response."
    result = await adapter.handle("Stop", payload)
    assert result.captured == 1
    assert [item["content"] for item in client.captures] == ["First response.", "Second response."]


@pytest.mark.asyncio
async def test_stop_falls_back_to_last_assistant_message_when_transcript_is_missing(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    payload = hook_payload(tmp_path, tmp_path / "missing-session.jsonl")

    result = await adapter.handle("Stop", payload)

    assert result.status == "degraded"
    assert result.committed is True
    assert result.captured == 1
    assert client.captures[0]["role"] == "assistant"
    assert client.captures[0]["content"] == "The work is complete."
    checkpoint = json.loads(
        (tmp_path / "checkpoints" / adapter._checkpoint_path("session-a").name).read_text(
            encoding="utf-8"
        )
    )
    assert "path" not in checkpoint
    assert "offset" not in checkpoint


@pytest.mark.asyncio
async def test_stop_transcript_read_failure_keeps_fallback_and_does_not_advance_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnemos.adapters import claude_code

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("not read", encoding="utf-8")

    def fail_read(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("transcript unavailable")

    monkeypatch.setattr(claude_code, "read_transcript_deltas", fail_read)
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    result = await adapter.handle("Stop", hook_payload(tmp_path, transcript))

    assert result.status == "degraded"
    assert result.committed is True
    assert result.captured == 1
    checkpoint = json.loads(
        (tmp_path / "checkpoints" / adapter._checkpoint_path("session-a").name).read_text(
            encoding="utf-8"
        )
    )
    assert "path" not in checkpoint
    assert "offset" not in checkpoint


@pytest.mark.asyncio
async def test_missing_transcript_fallback_is_idempotent_and_later_transcript_deduplicates(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    payload = hook_payload(tmp_path, transcript)

    first = await adapter.handle("Stop", payload)
    replay = await adapter.handle("Stop", payload)
    assert first.captured == 1
    assert replay.captured == 0
    assert len(client.captures) == 1

    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"role": "assistant", "content": "The work is complete."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    later = await adapter.handle("Stop", payload)

    assert later.captured == 0
    assert len(client.captures) == 1
    checkpoint = json.loads(
        (tmp_path / "checkpoints" / adapter._checkpoint_path("session-a").name).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["offset"] == transcript.stat().st_size
    assert checkpoint["pending_dedup"] == []


@pytest.mark.asyncio
async def test_missing_transcript_distinct_prompt_ids_preserve_identical_messages(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "missing-session.jsonl"
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    base = hook_payload(tmp_path, transcript)

    first = await adapter.handle("Stop", {**base, "prompt_id": "turn-1"})
    retry = await adapter.handle("Stop", {**base, "prompt_id": "turn-1"})
    second = await adapter.handle("Stop", {**base, "prompt_id": "turn-2"})

    assert first.captured == 1
    assert retry.captured == 0
    assert second.captured == 1
    assert len(client.captures) == 2
    assert [item["source_ref"] for item in client.captures] == ["turn-1", "turn-2"]

    transcript.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": uuid,
                    "message": {"role": "assistant", "content": "The work is complete."},
                }
            )
            for uuid in ("assistant-1", "assistant-2")
        )
        + "\n",
        encoding="utf-8",
    )
    later = await adapter.handle("Stop", {**base, "prompt_id": "turn-2"})

    assert later.captured == 0
    assert len(client.captures) == 2
    checkpoint = json.loads(
        (tmp_path / "checkpoints" / adapter._checkpoint_path("session-a").name).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["pending_dedup"] == []


@pytest.mark.asyncio
async def test_missing_transcript_without_fallback_is_degraded_and_uncommitted(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    result = await adapter.handle(
        "Stop",
        {
            "session_id": "session-a",
            "transcript_path": str(tmp_path / "missing-session.jsonl"),
            "cwd": str(tmp_path),
        },
    )

    assert result.status == "degraded"
    assert result.committed is False
    assert result.captured == 0
    assert client.captures == []


@pytest.mark.asyncio
async def test_corrupt_checkpoint_is_distinct_from_unavailable_transcript(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    checkpoint = adapter._checkpoint_path("session-a")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        await adapter.handle("Stop", hook_payload(tmp_path, tmp_path / "missing.jsonl"))

    assert client.captures == []


@pytest.mark.asyncio
async def test_checkpoint_write_failure_is_not_reported_as_transcript_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"role": "assistant", "content": "A transcript response."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(adapter, "_write_checkpoint", fail_write)
    with pytest.raises(OSError, match="checkpoint write failed"):
        await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))

    assert [item["content"] for item in client.captures] == ["A transcript response."]


@pytest.mark.asyncio
async def test_readable_transcript_capture_rejection_is_degraded_and_uncommitted(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"role": "assistant", "content": "A transcript response."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient(capture_status="degraded")
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    result = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))

    assert result.status == "degraded"
    assert result.committed is False
    assert result.captured == 0
    assert result.error == "capture_not_acknowledged"
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.asyncio
async def test_fallback_capture_rejection_is_degraded_and_uncommitted(tmp_path: Path) -> None:
    client = FakeClient(capture_status="degraded")
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    result = await adapter.handle(
        "Stop", hook_payload(tmp_path, tmp_path / "missing-session.jsonl")
    )

    assert result.status == "degraded"
    assert result.committed is False
    assert result.captured == 0
    assert result.error == "capture_not_acknowledged"
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.asyncio
async def test_non_object_checkpoint_is_corrupt(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    checkpoint = adapter._checkpoint_path("session-a")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        await adapter.handle("Stop", hook_payload(tmp_path, tmp_path / "missing.jsonl"))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_offset", [True, "1", -1])
async def test_malformed_checkpoint_offsets_are_corrupt(tmp_path: Path, bad_offset: Any) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    checkpoint = adapter._checkpoint_path("session-a")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"offset": bad_offset}), encoding="utf-8")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        await adapter.handle("Stop", hook_payload(tmp_path, transcript))


@pytest.mark.asyncio
async def test_malformed_offset_is_rejected_before_missing_transcript_fallback(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    checkpoint = adapter._checkpoint_path("session-a")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"offset": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        await adapter.handle("Stop", hook_payload(tmp_path, tmp_path / "missing.jsonl"))

    assert client.captures == []


@pytest.mark.asyncio
async def test_session_start_and_prompt_return_native_context_and_scope(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    payload = {
        "session_id": "session-a",
        "cwd": str(tmp_path),
        "hook_event_name": "SessionStart",
        "source": "startup",
    }
    result = await adapter.handle("SessionStart", payload)
    output = _native_output("SessionStart", result)
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "bounded test command" in output["hookSpecificOutput"]["additionalContext"]
    assert client.retrievals[0]["context"]["project_id"] == "project-a"

    prompt = dict(
        payload,
        hook_event_name="UserPromptSubmit",
        prompt="Please preserve this project convention.",
    )
    prompt_result = await adapter.handle("UserPromptSubmit", prompt)
    assert prompt_result.captured == 1


@pytest.mark.asyncio
async def test_distinct_native_prompt_ids_preserve_identical_prompt_events(tmp_path: Path) -> None:
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    base = {
        "session_id": "session-a",
        "cwd": str(tmp_path),
        "prompt": "Repeat this exact prompt",
    }
    first = await adapter.handle("UserPromptSubmit", {**base, "prompt_id": "prompt-1"})
    second = await adapter.handle("UserPromptSubmit", {**base, "prompt_id": "prompt-2"})

    assert first.captured == second.captured == 1
    assert len(client.captures) == 2
    assert client.captures[0]["event_id"] != client.captures[1]["event_id"]
    assert [item["source_ref"] for item in client.captures] == ["prompt-1", "prompt-2"]


@pytest.mark.asyncio
async def test_missing_profile_fails_closed_without_shared_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MNEMOS_USER_PROFILE_ID", raising=False)
    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path / "routing-review"))
    adapter = ClaudeCodeAdapter(client=FakeClient(), registry=registry_for(tmp_path))

    result = await adapter.handle("SessionStart", {"session_id": "session-a", "cwd": str(tmp_path)})
    assert result.status == "degraded"
    assert result.committed is False
    assert result.error == "missing_user_profile"


@pytest.mark.asyncio
async def test_unknown_or_disabled_project_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path / "routing-review"))
    registry = ScopeRegistry(
        projects={
            "project-a": ProjectIdentity(
                id="project-a",
                root=canonical_project_root(tmp_path),
                user_profile_id="profile-a",
                enabled=False,
            )
        }
    )
    adapter = ClaudeCodeAdapter(client=FakeClient(), registry=registry, user_profile_id="profile-a")
    result = await adapter.handle("SessionStart", {"session_id": "s", "cwd": str(tmp_path)})
    assert result.status == "degraded"
    assert result.committed is False
    assert result.error == "project_not_enabled"


def test_malformed_hook_process_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mnemos.adapters import claude_code

    monkeypatch.setattr(
        claude_code.sys, "stdin", type("Input", (), {"read": lambda self: "not-json"})()
    )
    assert claude_code.main([]) == 0
    assert capsys.readouterr().out == ""


def test_hook_exception_reports_sanitized_diagnostic_without_blocking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mnemos.adapters import claude_code

    async def fail(*args, **kwargs):
        raise RuntimeError("password=super-secret C:/private/transcript.jsonl")

    monkeypatch.setattr(claude_code, "run_hook", fail)
    monkeypatch.setattr(
        claude_code.sys,
        "stdin",
        type("Input", (), {"read": lambda self: json.dumps({"hook_event_name": "Stop"})})(),
    )

    assert claude_code.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "host=claude-code" in captured.err
    assert "event=Stop" in captured.err
    assert "status=error" in captured.err
    assert "exception=RuntimeError" in captured.err
    assert "super-secret" not in captured.err
    assert "private/transcript" not in captured.err


def test_degraded_hook_reports_status_without_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mnemos.adapters import claude_code

    async def degraded(*args, **kwargs):
        from mnemos.adapters.claude_code import HookResult

        return HookResult(status="degraded", committed=False, error="token=secret")

    monkeypatch.setattr(claude_code, "run_hook", degraded)
    monkeypatch.setattr(
        claude_code.sys,
        "stdin",
        type("Input", (), {"read": lambda self: json.dumps({"hook_event_name": "SessionStart"})})(),
    )

    assert claude_code.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "host=claude-code" in captured.err
    assert "event=SessionStart" in captured.err
    assert "status=degraded" in captured.err
    assert "secret" not in captured.err
