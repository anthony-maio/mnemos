"""Reconcile native text identities with service acknowledgements offline.

This proves only coverage of the supplied, independently selected native
records. It does not infer exactly-once callbacks, retained facts, extraction
quality, or recovery from a successful receipt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def _required(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("native identity is missing")
    return value


def _text(value: Any, types: tuple[str, ...] = ("text",)) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    return "".join(
        part["text"]
        for part in value
        if isinstance(part, dict)
        and part.get("type") in types
        and isinstance(part.get("text"), str)
    ).strip()


def _identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        _required(row.get(key)) for key in ("phase", "host", "session_id", "role", "source_ref")
    )


def _capture_ids_valid(payload: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Independently reproduce the bound host and service identity contracts."""
    context = payload["context"]
    scope = [
        context.get(key) for key in ("user_profile_id", "project_id", "workspace_id", "session_id")
    ]
    host = context.get("host")
    role, ref, content = (payload.get(key) for key in ("role", "source_ref", "content"))
    if not all(isinstance(value, str) for value in (role, ref, content)):
        return False
    if host == "claude-code":
        prefix = "claude:v2:"
        material = ["claude-code", 2, *scope, ref, role, content]
    elif host == "hermes" and isinstance(ref, str) and ref.startswith("transcript:v2:"):
        prefix = "hermes:"
        # A durable native row keeps its event ID if its body changes. Text
        # and role are checked separately against the native observation.
        material = [*scope, "", ref, ""]
    elif host == "codex" and isinstance(ref, str) and ":" in ref:
        prefix = "codex:"
        event, turn = ref.split(":", 1)
        material = [*scope, event, turn, role, content]
    else:
        return False
    host_id = (
        prefix
        + hashlib.sha256(
            json.dumps(material, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    durable_id = hashlib.sha256(
        json.dumps([scope[0], host, scope[3], host_id]).encode("utf-8")
    ).hexdigest()
    return payload.get("event_id") == host_id and result.get("event_id") == durable_id


def expected_messages(
    host: str, phase: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Read native text independently of capture receipts and fact storage.

    Codex uses actual user events and assistant response records, excluding
    injected user-role context. Its hook identity is the native turn ID.
    Claude uses transcript UUIDs, not the provider message ID or stdout UUID.
    Hermes uses durable native row IDs, including after a shortened view.
    Repeated snapshots must agree; this bounded verifier does not infer a
    complete transcript by merging partial or changing snapshots.
    """
    kinds = {
        "codex": "codex_hook",
        "claude-code": "claude_code_transcript",
        "hermes": "hermes_memory_provider",
    }
    if host not in kinds:
        raise ValueError("unsupported native host")
    _required(phase)
    found: dict[tuple[str, ...], dict[str, Any]] = {}

    def add(
        session: Any, role: str, ref: Any, content: str, line: int, supported: bool = True
    ) -> None:
        if not content:
            return
        row = {
            "phase": phase,
            "host": host,
            "session_id": _required(session),
            "role": role,
            "source_ref": _required(ref),
            "source_kind": kinds[host],
            "content": content,
            "native_line": line,
            "capture_supported": supported,
        }
        identity = _identity(row)
        previous = found.get(identity)
        if previous is not None and previous["content"] != content:
            raise ValueError("conflicting native text at one identity")
        if previous is None:
            found[identity] = row

    session: str | None = None
    turn: str | None = None
    users = finals = 0
    snapshots: dict[str, str] = {}
    for line, record in enumerate(records, 1):
        if host == "codex":
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("invalid native Codex payload")
            kind = record.get("type")
            event = payload.get("type")
            if kind == "session_meta":
                new_session = _required(payload.get("id"))
                if session is not None and new_session != session:
                    raise ValueError("conflicting Codex session identity")
                session = new_session
            elif kind == "event_msg" and event == "task_started":
                if turn is not None:
                    raise ValueError("previous Codex turn is not complete")
                turn = _required(payload.get("turn_id"))
                users = finals = 0
            elif kind == "event_msg" and event == "user_message":
                _required(turn)
                content = _text(payload.get("message"))
                if not content:
                    raise ValueError("Codex user text is empty")
                add(session, "user", f"UserPromptSubmit:{turn}", content, line)
                users += 1
            elif (
                kind == "response_item"
                and event == "message"
                and payload.get("role") == "assistant"
            ):
                content = _text(payload.get("content"), ("output_text", "text"))
                if content:
                    _required(turn)
                    final = payload.get("phase") == "final_answer"
                    ref = f"Stop:{turn}" if final else f"unsupported-assistant:{turn}:{line}"
                    add(session, "assistant", ref, content, line, supported=final)
                    finals += int(final)
            elif kind == "event_msg" and event == "task_complete":
                if turn is None or payload.get("turn_id") != turn or users != 1 or finals != 1:
                    raise ValueError("Codex turn is not complete with one user and final")
                turn = None
        elif host == "claude-code":
            role = record.get("type")
            message = record.get("message", {})
            if role in {"user", "assistant"} and isinstance(message, dict):
                content = _text(message.get("content"))
                if content:
                    if message.get("role", role) != role:
                        raise ValueError("conflicting native role")
                    add(record.get("sessionId"), str(role), record.get("uuid"), content, line)
        elif record.get("operation") == "transcript":
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError("Hermes transcript messages are missing")
            native_session = _required(record.get("session_id"))
            snapshot = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if snapshots.get(native_session, snapshot) != snapshot:
                raise ValueError("conflicting Hermes transcript snapshots")
            snapshots[native_session] = snapshot
            row_ids: set[int] = set()
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("invalid Hermes transcript message")
                role = message.get("role")
                if role in {"user", "assistant"} and _text(message.get("content")):
                    row_id = message.get("_row_id")
                    if type(row_id) is not int or row_id <= 0 or row_id in row_ids:
                        raise ValueError("Hermes native row identity is missing or duplicated")
                    row_ids.add(row_id)
                    add(
                        record.get("session_id"),
                        str(role),
                        f"transcript:v2:{row_id}",
                        _text(message.get("content")),
                        line,
                    )
    if turn is not None:
        raise ValueError("Codex turn is not complete")
    if not found:
        raise ValueError("native text expectations are empty")
    return list(found.values())


def reconcile_captures(
    expected: Sequence[Mapping[str, Any]],
    dispatches: Sequence[Mapping[str, Any]],
    *,
    scope: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Match scope, native identity and text; keep receipt/job states separate.

    Non-capture operations and phases outside the supplied expectation set
    cannot satisfy coverage. A provisional Claude prompt is reported separately
    and never substitutes for a transcript-identified user message.
    """
    if not expected:
        raise ValueError("native text expectations are empty")
    _required(scope.get("user_profile_id"))
    _required(scope.get("project_id"))
    keys = [_identity(row) for row in expected]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate expected native identity")
    phases = {row["phase"] for row in expected}
    matched: list[list[tuple[int, Mapping[str, Any]]]] = [[] for _ in expected]
    provisional: list[int] = []
    unmatched: list[int] = []
    excluded: list[int] = []
    for number, receipt in enumerate(dispatches):
        if receipt.get("operation") != "capture":
            continue
        if receipt.get("phase") not in phases:
            excluded.append(number)
            continue
        payload = receipt.get("payload")
        result = receipt.get("result")
        if not isinstance(payload, dict) or not isinstance(result, dict):
            unmatched.append(number)
            continue
        context = payload.get("context")
        if (
            not isinstance(context, dict)
            or result.get("status") != "committed"
            or not isinstance(result.get("event_id"), str)
            or not result["event_id"].strip()
        ):
            unmatched.append(number)
            continue
        if not _capture_ids_valid(payload, result):
            unmatched.append(number)
            continue
        candidates = [
            index
            for index, row in enumerate(expected)
            if row["phase"] == receipt.get("phase")
            and row["capture_supported"]
            and row["host"] == context.get("host")
            and row["session_id"] == context.get("session_id")
            and row["role"] == payload.get("role")
            and row["content"] == payload.get("content")
            and row["source_kind"] == payload.get("source_kind")
            and all(
                context.get(key) == scope.get(key)
                for key in ("user_profile_id", "project_id", "workspace_id")
            )
        ]
        normal = [
            index
            for index in candidates
            if expected[index]["source_ref"] == payload.get("source_ref")
        ]
        identity_normal = result.get("identity_status") == "normal" or (
            "identity_status" not in result
            and result.get("processing_status") == "suppressed"
            and payload.get("identity_status") == "normal"
        )
        if len(normal) == 1 and identity_normal:
            matched[normal[0]].append((number, result))
        elif (
            len(candidates) == 1
            and context.get("host") == "claude-code"
            and payload.get("role") == "user"
            and str(payload.get("source_ref", "")).startswith("prompt:")
            and result.get("identity_status") == "provisional_unassociated"
            and result.get("fact_eligible") is False
            and result.get("processing_status") == "deferred"
        ):
            provisional.append(number)
        else:
            unmatched.append(number)

    job_status = {row.get("id"): row.get("status") for row in jobs}
    owners: dict[str, set[int]] = {}
    for index, rows in enumerate(matched):
        for _number, result in rows:
            owners.setdefault(result["event_id"], set()).add(index)
    messages: list[dict[str, Any]] = []
    conflicts: list[str] = []
    duplicate_count = 0
    for index, (row, rows) in enumerate(zip(expected, matched)):
        ids = sorted({result["event_id"] for _, result in rows})
        conflict = len(ids) > 1 or any(len(owners[event_id]) > 1 for event_id in ids)
        label = ":".join(keys[index])
        if conflict:
            conflicts.append(label)
        elif rows:
            duplicate_count += len(rows) - 1
        messages.append(
            {
                "identity": label,
                "host": row["host"],
                "phase": row["phase"],
                "role": row["role"],
                "native_line": row["native_line"],
                "content_sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
                "content_bytes": len(row["content"].encode("utf-8")),
                "capture_supported": row["capture_supported"],
                "message_covered": bool(rows) and not conflict,
                "receipt_indices": [number for number, _ in rows],
                "durable_event_ids": ids,
                "suppressed": any(
                    result.get("processing_status") == "suppressed" for _, result in rows
                ),
                "processing_statuses": sorted(
                    {str(result.get("processing_status")) for _, result in rows}
                ),
                "job_statuses": sorted(
                    {str(job_status.get(result.get("job_id"), "not_present")) for _, result in rows}
                ),
            }
        )
    covered = sum(row["message_covered"] for row in messages)
    return {
        "schema_version": 1,
        "status": (
            "bounded_text_receipts_matched"
            if covered == len(expected) and not unmatched and not conflicts
            else "incomplete"
        ),
        "expected_messages": len(expected),
        "covered_messages": covered,
        "duplicate_receipts": duplicate_count,
        "provisional_receipts": provisional,
        "unmatched_receipts": unmatched,
        "excluded_receipts": excluded,
        "identity_conflicts": conflicts,
        "messages": messages,
        "limits": [
            "Coverage is conditional on the supplied native records being complete.",
            "Committed receipts do not prove exactly-once callbacks or retained facts.",
            "Pending/deferred/suppressed receipts and final job states are separate observations.",
            "No provider wire, outage, Desktop, multi-turn, or daily-use guarantee follows.",
        ],
    }
