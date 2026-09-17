from __future__ import annotations

import json
from pathlib import Path

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
    adapter = ClaudeCodeAdapter(client=FakeClient(), registry=registry_for(tmp_path))

    with pytest.raises(ValueError, match="PROFILE_ID"):
        await adapter.handle("SessionStart", {"session_id": "session-a", "cwd": str(tmp_path)})


@pytest.mark.asyncio
async def test_unknown_or_disabled_project_fails_closed(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="Project is not enabled"):
        await adapter.handle("SessionStart", {"session_id": "s", "cwd": str(tmp_path)})


def test_malformed_hook_process_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mnemos.adapters import claude_code

    monkeypatch.setattr(
        claude_code.sys, "stdin", type("Input", (), {"read": lambda self: "not-json"})()
    )
    assert claude_code.main([]) == 0
    assert capsys.readouterr().out == ""
