"""Only turns of the conversation become Claude transcript evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mnemos.adapters.claude_code import read_transcript_deltas


def _user(uuid: str, content: Any, **fields: Any) -> dict[str, Any]:
    return {"type": "user", "uuid": uuid, "message": {"role": "user", "content": content}, **fields}


def _assistant(uuid: str, text: str, **fields: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    message.update(fields.pop("message", {}))
    return {"type": "assistant", "uuid": uuid, "message": message, **fields}


# Record shapes as Claude Code 2.x writes them into the main session file.
RECORDS = [
    _user(
        "typed",
        [{"type": "text", "text": "Use the bounded test command."}],
        origin={"kind": "human"},
        promptSource="sdk",
        turnOrigin="human",
    ),
    _user("older-typed", "Keep diffs small."),
    _user(
        "compaction",
        "This session is being continued from a previous conversation.",
        isCompactSummary=True,
        isVisibleInTranscriptOnly=True,
        turnOrigin="human",
    ),
    _user(
        "skill", [{"type": "text", "text": "Base directory for this skill: C:/skills"}], isMeta=True
    ),
    _user(
        "peer",
        "Another Claude session sent a message: the refactor is done.",
        isMeta=True,
        origin={"kind": "peer", "from": "a123"},
        promptSource="system",
        turnOrigin="peer",
    ),
    _user(
        "notification",
        "<task-notification>The background task finished.</task-notification>",
        origin={"kind": "task-notification"},
        promptSource="system",
        turnOrigin="task_notification",
    ),
    _user("subagent-prompt", "Search the repo for retries.", isSidechain=True),
    _assistant("subagent-reply", "Retries live in the client.", isSidechain=True),
    _assistant("api-error", "API Error: overloaded", isApiErrorMessage=True),
    _assistant("synthetic", "No response requested.", message={"model": "<synthetic>"}),
    _assistant("reply", "The bounded command is make test-fast."),
]


def test_only_typed_prompts_and_real_replies_become_evidence(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in RECORDS), encoding="utf-8"
    )

    deltas, next_offset, _ = read_transcript_deltas(transcript)

    assert [(delta.source_ref, delta.role) for delta in deltas] == [
        ("typed", "user"),
        ("older-typed", "user"),
        ("reply", "assistant"),
    ]
    # Skipped records are still consumed, so they are never read again.
    assert next_offset == transcript.stat().st_size
