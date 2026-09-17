from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mnemos.adapters.claude_code import ClaudeCodeAdapter
from mnemos.adapters.codex import CodexAdapter
from mnemos.scopes import ScopeRegistry
from mnemos.routing_review import (
    ack_routing_failure,
    enqueue_routing_failure,
    list_routing_failures,
)


def test_queue_is_idempotent_and_stores_only_hashed_identity(tmp_path: Path) -> None:
    first = enqueue_routing_failure(
        host="codex",
        event="UserPromptSubmit",
        reason="missing_user_profile",
        session_id="session-secret",
        project_hint="C:/private/project",
        configured_profile="profile-secret",
        directory=tmp_path,
        timestamp="2026-09-16T12:00:00+00:00",
    )
    second = enqueue_routing_failure(
        host="codex",
        event="UserPromptSubmit",
        reason="missing_user_profile",
        session_id="session-secret",
        project_hint="C:/private/project",
        configured_profile="profile-secret",
        directory=tmp_path,
        timestamp="2026-09-16T12:01:00+00:00",
    )

    assert first["record_id"] == second["record_id"]
    records = list_routing_failures(configured_profile="profile-secret", directory=tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["host"] == "codex"
    assert record["event"] == "UserPromptSubmit"
    assert record["reason"] == "missing_user_profile"
    assert record["timestamp_utc"] == "2026-09-16T12:00:00+00:00"
    assert record["session_id_sha256"]
    assert record["project_hint_sha256"]
    assert record["configured_profile_sha256"]
    raw = (tmp_path / "unresolved-routing.json").read_text(encoding="utf-8")
    assert "session-secret" not in raw
    assert "C:/private/project" not in raw
    assert "profile-secret" not in raw


def test_queue_is_profile_isolated_and_bounded(tmp_path: Path) -> None:
    enqueue_routing_failure(
        host="claude-code",
        event="SessionStart",
        reason="project_not_enabled",
        session_id="a",
        configured_profile="alice",
        directory=tmp_path,
        timestamp="2026-09-16T12:00:00Z",
        max_records=3,
    )
    enqueue_routing_failure(
        host="claude-code",
        event="SessionStart",
        reason="project_not_enabled",
        session_id="b",
        configured_profile="alice",
        directory=tmp_path,
        timestamp="2026-09-16T12:01:00Z",
        max_records=3,
    )
    enqueue_routing_failure(
        host="claude-code",
        event="SessionStart",
        reason="project_not_enabled",
        session_id="c",
        configured_profile="bob",
        directory=tmp_path,
        timestamp="2026-09-16T12:02:00Z",
        max_records=3,
    )

    assert len(list_routing_failures(configured_profile="alice", directory=tmp_path)) == 2
    assert len(list_routing_failures(configured_profile="bob", directory=tmp_path)) == 1
    assert len(list_routing_failures(configured_profile=None, directory=tmp_path)) == 0
    state = json.loads((tmp_path / "unresolved-routing.json").read_text(encoding="utf-8"))
    assert len(state["records"]) == 3
    assert state["dropped_count"] == 0

    bounded = tmp_path / "bounded"
    for session_id in ("one", "two"):
        enqueue_routing_failure(
            host="codex",
            event="SessionStart",
            reason="project_not_enabled",
            session_id=session_id,
            configured_profile="alice",
            directory=bounded,
            max_records=1,
        )
    bounded_state = json.loads((bounded / "unresolved-routing.json").read_text(encoding="utf-8"))
    assert len(bounded_state["records"]) == 1
    assert bounded_state["dropped_count"] == 1


def test_adapter_unresolved_context_is_degraded_and_metadata_only(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path))
    secret = "do-not-store-this-prompt"
    claude = ClaudeCodeAdapter(
        registry=ScopeRegistry(),
        user_profile_id=None,
        checkpoint_directory=tmp_path / "checkpoints",
    )
    claude_result = asyncio.run(
        claude.handle(
            "UserPromptSubmit",
            {"cwd": str(tmp_path), "session_id": "claude-session", "prompt": secret},
        )
    )
    assert claude_result.status == "degraded"
    assert claude_result.committed is False

    codex = CodexAdapter(registry=ScopeRegistry(), user_profile_id=None)
    codex_result = asyncio.run(
        codex.handle(
            "UserPromptSubmit",
            {
                "cwd": str(tmp_path),
                "session_id": "codex-session",
                "turn_id": "turn-1",
                "prompt": secret,
            },
        )
    )
    assert codex_result.status == "manual_mcp"
    assert codex_result.degraded is True

    raw = (tmp_path / "unresolved-routing.json").read_text(encoding="utf-8")
    assert secret not in raw
    assert "claude-session" not in raw
    assert "codex-session" not in raw
    assert "exception" not in raw
    assert len(list_routing_failures(configured_profile=None, directory=tmp_path)) == 2


def test_ack_requires_matching_profile_hash(tmp_path: Path) -> None:
    record = enqueue_routing_failure(
        host="codex",
        event="SessionStart",
        reason="project_not_enabled",
        session_id="s",
        configured_profile="alice",
        directory=tmp_path,
    )
    assert not ack_routing_failure(
        record["record_id"], configured_profile="bob", directory=tmp_path
    )
    assert ack_routing_failure(record["record_id"], configured_profile="alice", directory=tmp_path)
    assert list_routing_failures(configured_profile="alice", directory=tmp_path) == []


def test_queue_reader_rejects_unexpected_content_fields(tmp_path):
    enqueue_routing_failure(host="codex", event="Stop", reason="missing_cwd", directory=tmp_path)
    path = tmp_path / "unresolved-routing.json"
    state = json.loads(path.read_text())
    state["records"][0]["prompt"] = "content must not become a review field"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="record schema"):
        list_routing_failures(configured_profile=None, directory=tmp_path)


def test_hermes_resolution_failure_records_no_raw_identity(tmp_path, monkeypatch):
    from mnemos.adapters.hermes import MnemosHermesProvider

    monkeypatch.setenv("MNEMOS_ROUTING_REVIEW_DIR", str(tmp_path))
    provider = MnemosHermesProvider(registry=ScopeRegistry(), user_profile_id="unknown-profile")
    provider.cwd = str(tmp_path / "private-path")
    Path(provider.cwd).mkdir()
    with pytest.raises(ValueError):
        provider._resolve_context("private-session")
    rows = list_routing_failures(configured_profile="unknown-profile", directory=tmp_path)
    assert len(rows) == 1 and rows[0]["host"] == "hermes"
    raw = (tmp_path / "unresolved-routing.json").read_text()
    assert "private-session" not in raw and "private-path" not in raw
