"""Private metadata queue for unresolved host identity diagnostics.

The queue records why a host event could not be scoped.  It deliberately does
not provide a conversation backlog or a replay mechanism: host content stays
with the host until an operator resolves identity and submits a new event.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from .service import _ProcessLock, _write_private_json, default_runtime_directory

_HOSTS = {"codex", "claude-code", "hermes"}
_EVENTS = {
    "SessionStart",
    "UserPromptSubmit",
    "PreCompact",
    "PostCompact",
    "Stop",
    "SessionEnd",
    "ContextResolve",
}
_REASONS = {
    "missing_cwd",
    "missing_session_id",
    "missing_user_profile",
    "unknown_user_profile",
    "project_not_enabled",
    "invalid_workspace",
    "registry_missing",
    "identity_resolution_failed",
}
_DEFAULT_MAX_RECORDS = 256
_QUEUE_NAME = "unresolved-routing.json"
_RECORD_FIELDS = {
    "record_id",
    "host",
    "event",
    "reason",
    "timestamp_utc",
    "session_id_sha256",
    "project_hint_sha256",
    "configured_profile_sha256",
}


def _sha256(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _queue_path(directory: str | Path | None = None) -> Path:
    if directory is not None:
        root = Path(directory).expanduser()
    else:
        configured = os.environ.get("MNEMOS_ROUTING_REVIEW_DIR", "").strip()
        root = Path(configured).expanduser() if configured else default_runtime_directory()
    return root / _QUEUE_NAME


def _timestamp_utc(value: str | datetime | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("timestamp must be an ISO timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "dropped_count": 0, "records": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Routing review queue is unreadable") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported routing review queue schema")
    records = raw.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("Routing review queue records are invalid")
    for item in records:
        if (
            set(item) != _RECORD_FIELDS
            or item["host"] not in _HOSTS
            or item["event"] not in _EVENTS
            or item["reason"] not in _REASONS
        ):
            raise ValueError("Routing review record schema is invalid")
        for name in (
            "record_id",
            "session_id_sha256",
            "project_hint_sha256",
            "configured_profile_sha256",
        ):
            value = item[name]
            if value is None and name != "record_id":
                continue
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Routing review identity digest is invalid")
        try:
            _timestamp_utc(item["timestamp_utc"])
        except (ValueError, TypeError):
            raise ValueError("Routing review timestamp is invalid") from None
    return {
        "schema_version": 1,
        "dropped_count": max(0, int(raw.get("dropped_count", 0) or 0)),
        "records": records,
    }


@contextmanager
def _queue_lock(path: Path) -> Iterator[None]:
    lock = _ProcessLock(path.with_suffix(".lock"))
    if not lock.acquire():
        raise OSError("Routing review queue is busy")
    try:
        yield
    finally:
        lock.release()


def _record_id(
    *,
    host: str,
    event: str,
    reason: str,
    session_id_sha256: str | None,
    project_hint_sha256: str | None,
    configured_profile_sha256: str | None,
) -> str:
    material = json.dumps(
        [
            host,
            event,
            reason,
            session_id_sha256,
            project_hint_sha256,
            configured_profile_sha256,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def enqueue_routing_failure(
    *,
    host: str,
    event: str,
    reason: str,
    session_id: str | None = None,
    project_hint: str | None = None,
    configured_profile: str | None = None,
    timestamp: str | datetime | None = None,
    directory: str | Path | None = None,
    max_records: int = _DEFAULT_MAX_RECORDS,
) -> dict[str, Any]:
    """Atomically enqueue one deduplicated, content-free routing diagnostic."""
    if host not in _HOSTS:
        raise ValueError("Unsupported routing review host")
    if event not in _EVENTS:
        raise ValueError("Unsupported routing review event")
    if reason not in _REASONS:
        raise ValueError("Unsupported routing review reason")
    if not 1 <= max_records <= 4096:
        raise ValueError("max_records must be between 1 and 4096")

    session_hash = _sha256(session_id)
    project_hash = _sha256(project_hint)
    profile_hash = _sha256(configured_profile)
    record_id = _record_id(
        host=host,
        event=event,
        reason=reason,
        session_id_sha256=session_hash,
        project_hint_sha256=project_hash,
        configured_profile_sha256=profile_hash,
    )
    record = {
        "record_id": record_id,
        "host": host,
        "event": event,
        "reason": reason,
        "timestamp_utc": _timestamp_utc(timestamp),
        "session_id_sha256": session_hash,
        "project_hint_sha256": project_hash,
        "configured_profile_sha256": profile_hash,
    }
    path = _queue_path(directory)
    with _queue_lock(path):
        state = _read_state(path)
        records = state["records"]
        existing = next((item for item in records if item.get("record_id") == record_id), None)
        if existing is not None:
            return dict(existing)
        records.append(record)
        dropped = max(0, len(records) - max_records)
        if dropped:
            state["records"] = records[dropped:]
            state["dropped_count"] += dropped
        _write_private_json(path, state)
    return record


def record_unresolved_routing_failure(**kwargs: Any) -> dict[str, Any] | None:
    """Best-effort adapter boundary; queue failure must not block the host."""
    try:
        return enqueue_routing_failure(**kwargs)
    except Exception:
        return None


def list_routing_failures(
    *,
    configured_profile: str | None,
    directory: str | Path | None = None,
    limit: int = _DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """List diagnostics visible to one configured profile's local owner."""
    if not 0 <= limit <= 4096:
        raise ValueError("limit must be between 0 and 4096")
    path = _queue_path(directory)
    if not path.exists():
        return []
    if limit == 0:
        return []
    with _queue_lock(path):
        state = _read_state(path)
    profile_hash = _sha256(configured_profile)
    return [
        dict(item)
        for item in state["records"]
        if item.get("configured_profile_sha256") == profile_hash
    ][-limit:]


def ack_routing_failure(
    record_id: str,
    *,
    configured_profile: str | None,
    directory: str | Path | None = None,
) -> bool:
    """Remove one diagnostic only from the matching profile's local queue."""
    if not isinstance(record_id, str) or not record_id.strip():
        return False
    path = _queue_path(directory)
    if not path.exists():
        return False
    profile_hash = _sha256(configured_profile)
    with _queue_lock(path):
        state = _read_state(path)
        records = state["records"]
        retained = [
            item
            for item in records
            if not (
                item.get("record_id") == record_id
                and item.get("configured_profile_sha256") == profile_hash
            )
        ]
        if len(retained) == len(records):
            return False
        state["records"] = retained
        _write_private_json(path, state)
    return True


def reason_for_context_error(error: BaseException) -> str:
    """Map resolver errors to a fixed, non-sensitive diagnostic category."""
    message = str(error).lower()
    if "cwd is required" in message:
        return "missing_cwd"
    if "session_id is required" in message:
        return "missing_session_id"
    if "scope registry" in message and "configured" in message:
        return "registry_missing"
    if "user_profile_id is required" in message:
        return "missing_user_profile"
    if "user profile is not registered" in message:
        return "unknown_user_profile"
    if "workspace" in message:
        return "invalid_workspace"
    if "project" in message and ("enabled" in message or "identity" in message):
        return "project_not_enabled"
    return "identity_resolution_failed"


__all__ = [
    "ack_routing_failure",
    "enqueue_routing_failure",
    "list_routing_failures",
    "reason_for_context_error",
    "record_unresolved_routing_failure",
]
