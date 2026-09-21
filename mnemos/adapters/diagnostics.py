"""Sanitized, fail-open diagnostics for native host hooks."""

from __future__ import annotations

import re
import sys
from typing import Any, TextIO

_EVENTS = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreCompact",
        "PostCompact",
        "Stop",
        "SessionEnd",
    }
)
_STATUSES = frozenset({"error", "degraded", "manual_mcp", "unsupported", "skipped"})
# Exact matches only: adapters may also carry arbitrary exception text in errors.
_REASONS = frozenset(
    {
        "transcript_unavailable",
        "ambiguous_transcript_replay",
        "ambiguous_transcript_association",
        "capture_not_acknowledged",
        "identity_redaction_ambiguous",
        "identity_reference_unavailable",
        "record_refused",
        "backlog_deferred",
        "missing_cwd",
        "missing_session_id",
        "missing_user_profile",
        "unknown_user_profile",
        "project_not_enabled",
        "invalid_workspace",
        "registry_missing",
        "identity_resolution_failed",
    }
)
_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def emit_hook_diagnostic(
    *,
    host: str,
    event: Any,
    status: str,
    reason: str | None = None,
    exception: BaseException | None = None,
    stream: TextIO | None = None,
) -> None:
    """Write fixed metadata only, keeping hook failures nonblocking and private."""

    safe_event = event if isinstance(event, str) and event in _EVENTS else "unknown"
    safe_status = status if status in _STATUSES else "error"
    safe_host = host if host in {"claude-code", "codex"} else "unknown"
    fields = [f"host={safe_host}", f"event={safe_event}", f"status={safe_status}"]
    if reason is not None:
        safe_reason = reason if reason in _REASONS else "unknown"
        fields.append(f"reason={safe_reason}")
    if exception is not None:
        exception_type = type(exception).__name__
        if not _TYPE_PATTERN.fullmatch(exception_type):
            exception_type = "Exception"
        fields.append(f"exception={exception_type}")
    try:
        print("mnemos hook diagnostic " + " ".join(fields), file=stream or sys.stderr)
    except (OSError, ValueError):
        # A disconnected diagnostic stream must not turn capture into a host failure.
        pass
