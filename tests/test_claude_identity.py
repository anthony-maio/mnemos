from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.claude_identity import legacy_event_id, scoped_event_id
from mnemos.scopes import ProjectIdentity, ScopeContext, ScopeRegistry, canonical_project_root


def context(*, project_id: str = "project-a", profile: str = "profile-a") -> ScopeContext:
    return ScopeContext(
        user_profile_id=profile,
        project_id=project_id,
        session_id="session-a",
        host="claude-code",
    )


def registry_for(path: Path, *, project_id: str = "project-a") -> ScopeRegistry:
    return ScopeRegistry(
        projects={
            project_id: ProjectIdentity(
                id=project_id,
                root=canonical_project_root(path),
                user_profile_id="profile-a",
                enabled=True,
            )
        }
    )


def test_legacy_helper_reproduces_unscoped_identity() -> None:
    expected = hashlib.sha256(
        "\0".join(("claude-code", "session-a", "native-1", "user", "Remember this")).encode()
    ).hexdigest()
    assert legacy_event_id("session-a", "native-1", "user", "Remember this") == "claude:" + expected


def test_scoped_identity_is_deterministic_and_scope_bound() -> None:
    first = scoped_event_id(context(), "native-1", "user", "Remember this")
    repeat = scoped_event_id(context(), "native-1", "user", "Remember this")
    other_project = scoped_event_id(
        context(project_id="project-b"), "native-1", "user", "Remember this"
    )
    other_profile = scoped_event_id(
        context(profile="profile-b"), "native-1", "user", "Remember this"
    )

    assert first == repeat
    assert first.startswith("claude:v2:")
    assert first != other_project
    assert first != other_profile
    assert scoped_event_id(context(), "native-1", "user", " Remember this ") == first


class _FakeClient:
    async def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "committed", "committed": True}

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"facts": []}


class _RedactingOutbox:
    @staticmethod
    def sanitize(payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "content": "[REDACTED_CREDENTIAL_ASSIGNMENT]"}


class _RedactingClient(_FakeClient):
    outbox = _RedactingOutbox()


@pytest.mark.asyncio
async def test_redaction_before_identity_is_degraded_without_capture(tmp_path: Path) -> None:
    client = _RedactingClient()
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
            "prompt": "Remember this",
            "prompt_id": "native-1",
        },
    )

    assert result.status == "degraded"
    assert result.committed is False
    assert result.captured == 0
    assert result.error == "identity_redaction_ambiguous"


@pytest.mark.asyncio
async def test_missing_native_prompt_reference_is_reported_after_capture(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(
        client=_FakeClient(),
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )

    result = await adapter.handle(
        "UserPromptSubmit",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "prompt": "Remember this",
        },
    )

    assert result.status == "degraded"
    assert result.committed is True
    assert result.captured == 1
    assert result.error == "identity_reference_unavailable"


def test_checkpoint_path_and_payload_are_scope_bound(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(
        client=_FakeClient(),
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    first = context()
    second = context(project_id="project-b")
    adapter._write_checkpoint(first, {"offset": 0, "pending_dedup": []})

    assert adapter._checkpoint_path(first) != adapter._checkpoint_path(second)
    saved = json.loads(adapter._checkpoint_path(first).read_text(encoding="utf-8"))
    assert saved["identity_version"] == 2
    assert saved["scope"] == adapter._checkpoint_scope(first)
    assert adapter._read_checkpoint(first) == saved


def test_checkpoint_scope_mismatch_is_rejected(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(
        client=_FakeClient(),
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    first = context()
    adapter._write_checkpoint(first, {"offset": 0, "pending_dedup": []})
    path = adapter._checkpoint_path(first)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["scope"]["project_id"] = "project-b"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        adapter._read_checkpoint(first)


@pytest.mark.asyncio
async def test_unbound_legacy_checkpoint_degrades_without_mutation(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(
        client=_FakeClient(),
        registry=registry_for(tmp_path),
        user_profile_id="profile-a",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    legacy = adapter._legacy_checkpoint_path("session-a")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"offset": 4}), encoding="utf-8")
    before = legacy.read_bytes()

    result = await adapter.handle(
        "Stop",
        {
            "session_id": "session-a",
            "cwd": str(tmp_path),
            "transcript_path": str(tmp_path / "missing.jsonl"),
        },
    )

    assert result.status == "degraded"
    assert result.committed is False
    assert result.error == "Claude checkpoint requires explicit scope binding"
    assert legacy.read_bytes() == before
