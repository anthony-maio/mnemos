"""
mnemos/hook_autostore.py — Deterministic Claude Code hook payload ingestion.

Builds safe memory candidates from hook events so Claude Code can auto-store
useful context without manual tool calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MemorySafetyConfig
from .memory_safety import MemoryWriteFirewall
from .types import Interaction

SUPPORTED_HOOK_EVENTS: tuple[str, ...] = ("UserPromptSubmit", "PostToolUse")

_LOW_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(ok|okay|thanks|thank you|sounds good|cool|great)\s*[.!]?\s*$", re.I),
    re.compile(r"^\s*(yes|no|yep|nope)\s*[.!]?\s*$", re.I),
)

_TOOL_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\berror\b"),
    re.compile(r"(?i)\bfailed\b"),
    re.compile(r"(?i)\bexception\b"),
    re.compile(r"(?i)\btraceback\b"),
)

_HOOK_FIREWALL = MemoryWriteFirewall(
    MemorySafetyConfig(
        enabled=True,
        secret_action="block",
        pii_action="block",
    )
)


@dataclass(frozen=True)
class AutoStoreDecision:
    """Result of evaluating a hook event for automatic memory storage."""

    should_store: bool
    reason: str
    interaction: Interaction | None
    scope: str
    scope_id: str | None


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def parse_hook_payload(raw: str) -> dict[str, Any]:
    """Parse hook JSON payload text into a dict."""
    stripped = raw.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _iter_strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        values: list[str] = []
        for item in node:
            values.extend(_iter_strings(item))
        return values
    if isinstance(node, dict):
        values = []
        for value in node.values():
            values.extend(_iter_strings(value))
        return values
    return []


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_user_prompt(payload: dict[str, Any]) -> str | None:
    direct = _first_string(
        payload,
        ("prompt", "user_prompt", "input", "text", "content", "message"),
    )
    if direct:
        return _normalize_text(direct)

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role not in {"user", "human"}:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return _normalize_text(content)
            if isinstance(content, list):
                pieces = [piece for piece in _iter_strings(content) if piece.strip()]
                if pieces:
                    return _normalize_text(" ".join(pieces))
    return None


def _extract_tool_name(payload: dict[str, Any]) -> str:
    for key in ("tool_name", "toolName", "tool", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "tool"


def _extract_tool_output(payload: dict[str, Any]) -> str | None:
    for key in ("output", "result", "stderr", "stdout", "response", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_text(value)

    result_obj = payload.get("result")
    if isinstance(result_obj, dict):
        strings = [value for value in _iter_strings(result_obj) if value.strip()]
        if strings:
            return _normalize_text(" ".join(strings))
    return None


def _is_low_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    if len(normalized) < 20:
        return True
    return any(pattern.match(normalized) for pattern in _LOW_SIGNAL_PATTERNS)


def _is_failure_signal(text: str) -> bool:
    return any(pattern.search(text) for pattern in _TOOL_FAILURE_PATTERNS)


def _scope_id_from_payload(payload: dict[str, Any]) -> str | None:
    for key in (
        "cwd",
        "working_directory",
        "workspace",
        "workspace_path",
        "project_path",
        "repo_path",
        "root",
    ):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                return Path(raw).resolve().name or Path(raw).name
            except OSError:
                return Path(raw).name or raw.strip()
    return None


def decide_autostore(
    *,
    event: str,
    payload: dict[str, Any],
    default_scope: str = "project",
    default_scope_id: str | None = None,
    max_chars: int = 1200,
) -> AutoStoreDecision:
    """Create an auto-store decision from hook event and payload."""
    if event not in SUPPORTED_HOOK_EVENTS:
        return AutoStoreDecision(
            should_store=False,
            reason=f"Unsupported hook event: {event!r}",
            interaction=None,
            scope=default_scope,
            scope_id=default_scope_id,
        )

    scope = default_scope
    payload_scope_id = _scope_id_from_payload(payload)
    scope_id = default_scope_id or payload_scope_id or "default"

    if event == "UserPromptSubmit":
        prompt = _extract_user_prompt(payload)
        if not prompt:
            return AutoStoreDecision(
                should_store=False,
                reason="No user prompt found in hook payload.",
                interaction=None,
                scope=scope,
                scope_id=scope_id,
            )
        if _is_low_signal(prompt):
            return AutoStoreDecision(
                should_store=False,
                reason="Skipped low-signal prompt.",
                interaction=None,
                scope=scope,
                scope_id=scope_id,
            )
        safety = _HOOK_FIREWALL.apply(prompt)
        if not safety.allowed:
            return AutoStoreDecision(
                should_store=False,
                reason=f"Skipped by safety policy: {safety.reason}",
                interaction=None,
                scope=scope,
                scope_id=scope_id,
            )

        content = safety.content[:max_chars]
        return AutoStoreDecision(
            should_store=True,
            reason="Store user prompt memory.",
            interaction=Interaction(
                role="user",
                content=content,
                metadata={
                    "source": "claude_hook",
                    "hook_event": event,
                },
            ),
            scope=scope,
            scope_id=scope_id,
        )

    tool_name = _extract_tool_name(payload)
    tool_output = _extract_tool_output(payload)
    if not tool_output:
        return AutoStoreDecision(
            should_store=False,
            reason="No tool output found in hook payload.",
            interaction=None,
            scope=scope,
            scope_id=scope_id,
        )
    safety = _HOOK_FIREWALL.apply(tool_output)
    if not safety.allowed:
        return AutoStoreDecision(
            should_store=False,
            reason=f"Skipped by safety policy: {safety.reason}",
            interaction=None,
            scope=scope,
            scope_id=scope_id,
        )
    if not _is_failure_signal(tool_output):
        return AutoStoreDecision(
            should_store=False,
            reason="Skipped non-failure tool output.",
            interaction=None,
            scope=scope,
            scope_id=scope_id,
        )

    content = f"Tool failure [{tool_name}]: {safety.content[:max_chars]}"
    return AutoStoreDecision(
        should_store=True,
        reason="Store high-signal tool failure memory.",
        interaction=Interaction(
            role="assistant",
            content=content,
            metadata={
                "source": "claude_hook",
                "hook_event": event,
                "tool_name": tool_name,
            },
        ),
        scope=scope,
        scope_id=scope_id,
    )
