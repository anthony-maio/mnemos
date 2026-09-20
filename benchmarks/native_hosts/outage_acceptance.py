"""Offline acceptance gates for the successor Claude outage observation.

These helpers inspect captured values only.  They do not start a service or a
native host, and they deliberately keep the v1 persistence checks intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_service_outage_v1 import require_pending_sources as _require_v1_sources


def _database_path(health: Mapping[str, Any]) -> Any:
    runtime = health.get("runtime")
    store = health.get("store")
    if not isinstance(runtime, Mapping) or not isinstance(store, Mapping):
        raise ValueError("dispatcher runtime or store health is missing")
    if runtime.get("store_type") != "SQLiteStore":
        raise ValueError("dispatcher store type is not SQLiteStore")
    if runtime.get("database_path") != store.get("db_path"):
        raise ValueError("dispatcher runtime and store database paths differ")
    return runtime.get("database_path")


def require_health_gate(
    health: Mapping[str, Any],
    *,
    expected_database: str | Path,
    module_config: Mapping[str, bool],
    background_maintenance: bool = False,
) -> None:
    """Check a response obtained through the authenticated Mnemos client."""
    if type(health.get("schema_version")) is not int or health["schema_version"] != 1:
        raise ValueError("dispatcher health schema differs")
    if health.get("status") != "running":
        raise ValueError("dispatcher health is not running")
    actual = _database_path(health)
    if (
        not isinstance(actual, str)
        or not actual
        or Path(actual).resolve() != Path(expected_database).resolve()
    ):
        raise ValueError("dispatcher database path differs from the expected pin")
    required_modules = {"surprisal", "mutable_rag", "affective", "sleep", "spreading"}
    if set(module_config) != required_modules or any(
        value is not False for value in module_config.values()
    ):
        raise ValueError("all five injected module switches must be disabled")
    capabilities = health.get("capabilities")
    maintenance = health.get("maintenance")
    if not isinstance(capabilities, Mapping) or not isinstance(maintenance, Mapping):
        raise ValueError("dispatcher maintenance health is missing")
    if capabilities.get("background_maintenance") is not background_maintenance:
        raise ValueError("background maintenance capability differs from the expected pin")
    if maintenance.get("scheduler_running") is not background_maintenance:
        raise ValueError("background maintenance state differs from the injected pin")


def _contains_tool_block(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("type") in {"tool_use", "tool_result"}:
            return True
        return any(_contains_tool_block(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tool_block(item) for item in value)
    return False


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
        return text
    return ""


def require_native_transcript(
    records: Sequence[Mapping[str, Any]],
    *,
    submitted_marker: str,
    expected_ack: str,
    expected_model: str,
) -> None:
    """Require one exact user marker and acknowledgement from the pinned model."""
    users = [row for row in records if row.get("type") == "user"]
    assistants = [row for row in records if row.get("type") == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError("native transcript does not contain exactly one user and assistant record")
    if _contains_tool_block(records):
        raise ValueError("native transcript contains a tool block")
    if _message_text(users[0].get("message", {})) != submitted_marker:
        raise ValueError("native transcript marker differs from the submitted prompt")
    assistant = assistants[0].get("message")
    if not isinstance(assistant, Mapping):
        raise ValueError("native assistant message is missing")
    if assistant.get("model") != expected_model:
        raise ValueError("native assistant model differs from the expected model")
    if _message_text(assistant) != expected_ack:
        raise ValueError("native assistant acknowledgement differs from the expected text")


def require_pending_sources_v2(state: dict[str, Any], pending: list[dict[str, Any]]) -> None:
    """Apply v1 outbox checks plus physical event scope and identity columns."""
    _require_v1_sources(state, pending)
    events = {row["id"]: row for row in state["events"]}
    for entry in pending:
        payload = entry["payload"]
        context = payload["context"]
        event = events[entry["event_id"]]
        expected_scope = context.get("scope", "project")
        expected_scope_id = context.get("scope_id", context.get("project_id"))
        checks = {
            "scope": expected_scope,
            "scope_id": expected_scope_id,
            "source": context.get("host"),
            "user_profile_id": context.get("user_profile_id"),
            "session_id": context.get("session_id"),
            "source_event_id": payload["event_id"],
        }
        if any(event.get(key) != value for key, value in checks.items()):
            raise ValueError("physical event identity differs from the native payload context")


__all__ = ["require_health_gate", "require_native_transcript", "require_pending_sources_v2"]
