from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter, _native_output
from mnemos.scopes import (
    ProjectIdentity,
    ScopeContext,
    ScopeRegistry,
    canonical_project_root,
)


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


def scoped_checkpoint_path(adapter: ClaudeCodeAdapter) -> Path:
    return adapter._checkpoint_path(scoped_test_context())


def scoped_test_context() -> ScopeContext:
    return ScopeContext(
        user_profile_id="profile-a",
        project_id="project-a",
        session_id="session-a",
        host="claude-code",
    )


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
async def test_stop_replay_after_restart_uses_assistant_checkpoint_marker(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    assistant = "The durable assistant response."
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "user-1",
                        "message": {"role": "user", "content": "Remember this."},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "assistant-1",
                        "message": {"role": "assistant", "content": assistant},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient()
    checkpoint_directory = tmp_path / "checkpoints"
    first_adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=checkpoint_directory,
    )
    payload = {**hook_payload(tmp_path, transcript), "last_assistant_message": assistant}

    first = await first_adapter.handle("Stop", payload)
    checkpoint = json.loads(scoped_checkpoint_path(first_adapter).read_text(encoding="utf-8"))
    assert first.captured == 2
    assert checkpoint["last_assistant"]["source_ref"] == "assistant-1"
    assert "content" not in checkpoint["last_assistant"]

    restarted = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=checkpoint_directory,
    )
    replay = await restarted.handle("Stop", payload)

    assert replay.captured == 0
    assert replay.status == "degraded"
    assert replay.error == "identity_reference_unavailable"
    assert len(client.captures) == 2


@pytest.mark.asyncio
async def test_precompact_then_stop_does_not_recapture_assistant(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    assistant = "The compacted assistant response."
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-compact-1",
                "message": {"role": "assistant", "content": assistant},
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
    payload = {**hook_payload(tmp_path, transcript), "last_assistant_message": assistant}

    compacted = await adapter.handle("PreCompact", payload)
    stopped = await adapter.handle("Stop", payload)

    assert compacted.captured == 1
    assert stopped.captured == 0
    assert stopped.status == "degraded"
    assert stopped.error == "identity_reference_unavailable"
    assert len(client.captures) == 1


@pytest.mark.asyncio
async def test_identical_assistant_text_with_new_transcript_uuid_is_captured(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    assistant = "The same words can be a new response."
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {"role": "assistant", "content": assistant},
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
    payload = {**hook_payload(tmp_path, transcript), "last_assistant_message": assistant}
    await adapter.handle("PreCompact", payload)
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "assistant-2",
                    "message": {"role": "assistant", "content": assistant},
                }
            )
            + "\n"
        )

    result = await adapter.handle("Stop", payload)

    assert result.captured == 1
    assert [item["source_ref"] for item in client.captures] == [
        "assistant-1",
        "assistant-2",
    ]


@pytest.mark.asyncio
async def test_delayed_distinct_assistant_transcript_is_not_lost(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "user-1",
                "message": {"role": "user", "content": "Start the response."},
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
    first_payload = {
        **hook_payload(tmp_path, transcript),
        "last_assistant_message": "The first response arrived through the fallback.",
    }
    first = await adapter.handle("Stop", first_payload)
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "assistant-delayed-2",
                    "message": {
                        "role": "assistant",
                        "content": "A distinct response arrived later.",
                    },
                }
            )
            + "\n"
        )

    second = await adapter.handle(
        "Stop",
        {**first_payload, "last_assistant_message": "A distinct response arrived later."},
    )

    assert first.captured == 2
    assert second.captured == 1
    assert client.captures[-1]["source_ref"] == "assistant-delayed-2"


@pytest.mark.asyncio
async def test_new_turn_same_text_uses_prompt_id_before_transcript_flush(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    assistant = "A repeated response from a later turn."
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-turn-1",
                "message": {"role": "assistant", "content": assistant},
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
    first_payload = {
        **hook_payload(tmp_path, transcript),
        "prompt_id": "turn-1",
        "last_assistant_message": assistant,
    }
    first = await adapter.handle("PreCompact", first_payload)

    second = await adapter.handle(
        "Stop",
        {**first_payload, "prompt_id": "turn-2"},
    )

    assert first.captured == 1
    assert second.captured == 1
    assert client.captures[-1]["source_ref"] == "turn-2"


@pytest.mark.asyncio
async def test_user_only_checkpoint_keeps_unflushed_assistant_fallback_available(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "user-only-1",
                "message": {"role": "user", "content": "The response is still flushing."},
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
    payload = {
        **hook_payload(tmp_path, transcript),
        "last_assistant_message": "The response is still flushing.",
    }

    compacted = await adapter.handle("PreCompact", payload)
    stopped = await adapter.handle("Stop", payload)

    assert compacted.captured == 1
    assert stopped.captured == 1
    assert stopped.status == "degraded"
    assert client.captures[-1]["source_ref"] == "last_assistant_message"
    checkpoint = json.loads(scoped_checkpoint_path(adapter).read_text(encoding="utf-8"))
    assert checkpoint["last_assistant"] is None


@pytest.mark.asyncio
async def test_eof_checkpoint_without_assistant_marker_degrades_without_capture(
    tmp_path: Path,
) -> None:
    from mnemos.adapters.claude_code import _current_transcript_identity

    transcript = tmp_path / "session.jsonl"
    assistant = "An assistant response waiting for a trustworthy marker."
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "assistant-eof-1",
                "message": {"role": "assistant", "content": assistant},
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
    context = scoped_test_context()
    checkpoint = scoped_checkpoint_path(adapter)
    adapter._write_checkpoint(
        context,
        {
            "path": str(transcript.resolve()),
            "identity": _current_transcript_identity(transcript),
            "offset": transcript.stat().st_size,
            "pending_dedup": [],
        },
    )

    result = await adapter.handle(
        "Stop",
        {**hook_payload(tmp_path, transcript), "last_assistant_message": assistant},
    )

    assert result.captured == 0
    assert result.status == "degraded"
    assert result.error == "ambiguous_transcript_replay"
    assert client.captures == []
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "last_assistant" not in saved


@pytest.mark.asyncio
async def test_malformed_assistant_marker_fails_closed(
    tmp_path: Path,
) -> None:
    from mnemos.adapters.claude_code import _current_transcript_identity

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    adapter._write_checkpoint(
        scoped_test_context(),
        {
            "path": str(transcript.resolve()),
            "identity": _current_transcript_identity(transcript),
            "offset": 0,
            "pending_dedup": [],
            "last_assistant": {"fingerprint": "missing-required-marker-fields"},
        },
    )

    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        await adapter.handle("Stop", hook_payload(tmp_path, transcript))
    assert client.captures == []


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
@pytest.mark.parametrize("hook_event", ["PreCompact", "Stop"])
async def test_partial_transcript_capture_retries_acknowledged_rows_safely(
    tmp_path: Path,
    hook_event: str,
) -> None:
    class FailOnce(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = True

        async def capture(self, payload: dict) -> dict:
            if self.fail_next and payload["role"] == "assistant":
                self.fail_next = False
                self.captures.append(payload)
                return {"status": "degraded", "committed": False}
            return await super().capture(payload)

    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "retry-user",
                        "message": {"role": "user", "content": "Retry this batch."},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "retry-assistant",
                        "message": {"role": "assistant", "content": "The batch is retried."},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = FailOnce()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    payload = {
        **hook_payload(tmp_path, transcript),
        "last_assistant_message": "The batch is retried.",
    }
    first = await adapter.handle(hook_event, payload)
    assert first.captured == 1
    assert first.error == "capture_not_acknowledged"
    assert not (tmp_path / "checkpoints").exists()

    # A batch checkpoint is written only after every row is accepted. The retry
    # may replay an acknowledged row, so the service must deduplicate its ID.
    second = await adapter.handle(hook_event, payload)

    assert second.captured == 2
    assert [item["source_ref"] for item in client.captures] == [
        "retry-user",
        "retry-assistant",
        "retry-user",
        "retry-assistant",
    ]
    assert (
        json.loads(scoped_checkpoint_path(adapter).read_text(encoding="utf-8"))["last_assistant"][
            "source_ref"
        ]
        == "retry-assistant"
    )


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
    checkpoint = json.loads(scoped_checkpoint_path(adapter).read_text(encoding="utf-8"))
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
    checkpoint = json.loads(scoped_checkpoint_path(adapter).read_text(encoding="utf-8"))
    assert "path" not in checkpoint
    assert "offset" not in checkpoint


@pytest.mark.asyncio
async def test_missing_transcript_fallback_keeps_later_native_observation(
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

    assert later.captured == 1
    assert len(client.captures) == 2
    assert client.captures[-1]["source_ref"] == "assistant-1"
    assert later.status == "degraded"
    assert later.error == "ambiguous_transcript_association"
    checkpoint = json.loads(scoped_checkpoint_path(adapter).read_text(encoding="utf-8"))
    assert checkpoint["offset"] == transcript.stat().st_size
    assert len(checkpoint["pending_dedup"]) == 1
    assert checkpoint["pending_dedup"][0]["turn_ref"] is None


@pytest.mark.asyncio
async def test_missing_transcript_prompt_association_keeps_distinct_native_observation(
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

    assert later.captured == 2
    assert len(client.captures) == 4
    assert later.status == "degraded"
    assert later.error == "ambiguous_transcript_association"
    assert [item["source_ref"] for item in client.captures[-2:]] == [
        "assistant-1",
        "assistant-2",
    ]
    checkpoint = json.loads(scoped_checkpoint_path(adapter).read_text(encoding="utf-8"))
    assert len(checkpoint["pending_dedup"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_field", ["prompt_id", "promptId", "turn_id"])
async def test_fallback_prompt_turn_can_deduplicate_its_later_transcript_flush(
    tmp_path: Path,
    reference_field: str,
) -> None:
    transcript = tmp_path / "session.jsonl"
    assistant = "The response was captured before the transcript flushed."
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    base = hook_payload(tmp_path, transcript)

    first = await adapter.handle(
        "Stop",
        {**base, "prompt_id": "turn-1", "last_assistant_message": assistant},
    )
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "native-assistant-1",
                reference_field: "turn-1",
                "message": {"role": "assistant", "content": assistant},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    flushed = await adapter.handle(
        "Stop",
        {**base, "prompt_id": "turn-1", "last_assistant_message": assistant},
    )

    assert first.captured == 1
    if reference_field == "turn_id":
        assert flushed.captured == 1
        assert len(client.captures) == 2
        assert flushed.error == "ambiguous_transcript_association"
    else:
        assert flushed.captured == 0
        assert len(client.captures) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_event", ["PreCompact", "Stop"])
async def test_transcript_without_native_id_fails_closed_degraded(
    tmp_path: Path,
    hook_event: str,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "Offset identity only."},
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

    result = await adapter.handle(hook_event, hook_payload(tmp_path, transcript))

    assert result.captured == 0
    assert result.status == "degraded"
    assert result.error == "identity_reference_unavailable"
    assert client.captures == []
    assert not scoped_checkpoint_path(adapter).exists()


@pytest.mark.asyncio
async def test_native_prompt_reference_associates_user_capture_with_transcript(
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
    payload = {
        **hook_payload(tmp_path, transcript),
        "prompt_id": "native-prompt-1",
        "prompt": "Keep this native prompt association.",
    }
    first = await adapter.handle("UserPromptSubmit", payload, include_retrieval=False)
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "native-user-record-1",
                "prompt_id": "native-prompt-1",
                "message": {"role": "user", "content": payload["prompt"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    later = await adapter.handle("PreCompact", payload, include_retrieval=False)
    assert first.captured == 1
    assert later.captured == 0
    assert later.status == "ok"
    assert len(client.captures) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_prompt_id", [17, "   ", {"id": "not-a-string"}])
async def test_stop_invalid_prompt_id_uses_degraded_fallback(
    tmp_path: Path,
    invalid_prompt_id: Any,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    payload = {**hook_payload(tmp_path, transcript), "prompt_id": invalid_prompt_id}
    first = await adapter.handle("Stop", payload)
    replay = await adapter.handle("Stop", payload)
    assert first.captured == 1
    assert first.status == "degraded"
    assert first.error == "identity_reference_unavailable"
    assert replay.captured == 0
    assert len(client.captures) == 1
    assert client.captures[0]["source_ref"] == "last_assistant_message"


@pytest.mark.asyncio
async def test_missing_id_rotation_with_same_offset_restarts_safely(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    first_record = {
        "type": "assistant",
        "message": {"role": "assistant", "content": "AAAA"},
    }
    second_record = {
        "type": "assistant",
        "message": {"role": "assistant", "content": "BBBB"},
    }
    transcript.write_text(json.dumps(first_record) + "\n", encoding="utf-8")
    client = FakeClient()
    adapter = ClaudeCodeAdapter(
        client=client,
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    first = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))
    transcript.write_text(json.dumps(second_record) + "\n", encoding="utf-8")
    second = await adapter.handle("PreCompact", hook_payload(tmp_path, transcript))

    assert first.captured == 0
    assert second.captured == 0
    assert first.error == "identity_reference_unavailable"
    assert second.error == "identity_reference_unavailable"
    assert client.captures == []


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
    checkpoint = scoped_checkpoint_path(adapter)
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
    checkpoint = scoped_checkpoint_path(adapter)
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
    checkpoint = scoped_checkpoint_path(adapter)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "identity_version": 2,
                "scope": adapter._checkpoint_scope(scoped_test_context()),
                "offset": bad_offset,
            }
        ),
        encoding="utf-8",
    )
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
    checkpoint = scoped_checkpoint_path(adapter)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "identity_version": 2,
                "scope": adapter._checkpoint_scope(scoped_test_context()),
                "offset": True,
            }
        ),
        encoding="utf-8",
    )

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
    assert all(item["identity_version"] == 2 for item in client.captures)
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
