"""Native Codex lifecycle adapter for the local Mnemos service.

Codex command hooks receive one JSON object on stdin.  This adapter keeps the
hook boundary small: prompts and completed assistant messages become durable
episodic events, while start and post-compaction hooks request bounded recall.
It intentionally does not read Codex transcript files because their format is
not a stable hook contract.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..client import MnemosClient
from ..scopes import ScopeContext, ScopeRegistry, default_registry_path as shared_registry_path

SUPPORTED_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreCompact",
    "PostCompact",
    "Stop",
    "SessionEnd",
)
RECALL_EVENTS = {"SessionStart", "UserPromptSubmit", "PostCompact"}
CAPTURE_EVENTS = {"UserPromptSubmit", "Stop"}
MAX_CONTEXT_CHARS = 8_000
MAX_QUERY_CHARS = 10_000


@dataclass(frozen=True)
class CodexHookResult:
    """Safe result returned by one command-hook invocation."""

    status: str
    captured: int = 0
    context: str = ""
    event_ids: tuple[str, ...] = ()
    degraded: bool = False
    error: str | None = None


def default_registry_path() -> Path:
    return shared_registry_path()


def load_registry(path: str | Path | None = None) -> ScopeRegistry:
    target = Path(path).expanduser() if path is not None else default_registry_path()
    if not target.exists():
        raise ValueError("Mnemos scope registry is not configured")
    return ScopeRegistry.load(target)


def installed_capabilities() -> dict[str, Any]:
    """Report local CLI evidence without changing Codex configuration."""
    executable = shutil.which("codex")
    version = None
    if executable:
        try:
            completed = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            version = (completed.stdout or completed.stderr).strip() or None
        except (OSError, subprocess.TimeoutExpired):
            version = None
    return {
        "executable": executable,
        "version": version,
        "events": list(SUPPORTED_EVENTS),
        "fallback": "manual_mcp",
    }


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp(payload: Mapping[str, Any]) -> datetime:
    value = payload.get("timestamp") or payload.get("occurred_at")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class CodexAdapter:
    """Scope-aware, transport-only Codex hook adapter."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        registry: ScopeRegistry | None = None,
        registry_path: str | Path | None = None,
        user_profile_id: str | None = None,
        top_k: int = 5,
        max_context_tokens: int = 1200,
    ) -> None:
        self.client = client or MnemosClient()
        self.registry = registry or load_registry(registry_path)
        # A default profile would silently merge unrelated users.  Require an
        # explicit identity and let a missing environment be degraded below.
        self.user_profile_id = user_profile_id or os.environ.get("MNEMOS_USER_PROFILE_ID")
        self.top_k = max(0, min(int(top_k), 100))
        self.max_context_tokens = max(0, min(int(max_context_tokens), 32_000))

    def _context(self, payload: Mapping[str, Any]) -> ScopeContext:
        cwd = _text(payload.get("cwd"))
        session_id = _text(payload.get("session_id"))
        if not cwd:
            raise ValueError("Codex hook cwd is required")
        if not session_id:
            raise ValueError("Codex hook session_id is required")
        if not self.user_profile_id:
            raise ValueError("MNEMOS_USER_PROFILE_ID is required")
        workspace_id = payload.get("workspace_id")
        if workspace_id is not None and not isinstance(workspace_id, str):
            raise ValueError("workspace_id must be a string")
        return self.registry.resolve(
            cwd,
            user_profile_id=self.user_profile_id,
            session_id=session_id,
            host="codex",
            workspace_id=workspace_id,
        )

    @staticmethod
    def _event_id(
        context: ScopeContext,
        *,
        event: str,
        turn_id: str,
        role: str,
        content: str,
    ) -> str:
        identity = [
            context.user_profile_id,
            context.project_id,
            context.workspace_id,
            context.session_id,
            event,
            turn_id,
            role,
            content,
        ]
        material = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        return "codex:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _query(event: str, payload: Mapping[str, Any]) -> str:
        if event == "UserPromptSubmit":
            return _text(payload.get("prompt"))[:MAX_QUERY_CHARS]
        return "recent project context and decisions"

    async def _capture(
        self,
        context: ScopeContext,
        *,
        event: str,
        role: str,
        content: str,
        turn_id: str,
        occurred_at: datetime,
    ) -> tuple[bool, str, bool]:
        event_id = self._event_id(context, event=event, turn_id=turn_id, role=role, content=content)
        payload = {
            "event_id": event_id,
            "context": context.model_dump(mode="json"),
            "occurred_at": occurred_at.astimezone(timezone.utc).isoformat(),
            "role": role,
            "content": content,
            "source_kind": "codex_hook",
            "source_ref": f"{event}:{turn_id or 'session'}",
        }
        try:
            result = await self.client.capture(payload)
        except Exception:
            return False, event_id, True
        if not isinstance(result, Mapping):
            return False, event_id, True
        accepted = result.get("committed") is True or result.get("status") in {
            "committed",
            "queued",
        }
        return bool(accepted), event_id, result.get("status") == "queued"

    async def _retrieve(
        self, context: ScopeContext, event: str, payload: Mapping[str, Any]
    ) -> tuple[str, bool]:
        if os.environ.get("MNEMOS_AUTO_RETRIEVE", "true").lower() in {"0", "false", "no", "off"}:
            return "", False
        query = self._query(event, payload)
        if len(query) < 3:
            return "", False
        result = await self.client.retrieve(
            {
                "query": query,
                "context": context.model_dump(mode="json"),
                "allowed_scopes": ["project", "workspace", "global"],
                "top_k": self.top_k,
                "max_context_tokens": self.max_context_tokens,
                "profile": "baseline",
            }
        )
        if not isinstance(result, Mapping):
            return "", True
        degraded = result.get("degraded") is True or result.get("status") == "degraded"
        rendered = result.get("context")
        if isinstance(rendered, str) and rendered:
            return rendered[:MAX_CONTEXT_CHARS], degraded
        facts = result.get("facts")
        if not isinstance(facts, list):
            return "", degraded
        lines: list[str] = []
        for index, fact in enumerate(facts, 1):
            if isinstance(fact, Mapping):
                content = _text(fact.get("content"))
                if content:
                    lines.append(f"{index}. {' '.join(content.split())[:800]}")
        return "\n".join(lines)[:MAX_CONTEXT_CHARS], degraded

    async def _maintenance(self, context: ScopeContext) -> bool:
        request = getattr(self.client, "request", None)
        if not callable(request):
            return False
        try:
            result = await request(
                "maintenance",
                {"context": context.model_dump(mode="json"), "host": "codex"},
                queue_on_failure=False,
            )
        except Exception:
            return False
        return isinstance(result, Mapping) and result.get("degraded") is not True

    async def handle(self, event: str, payload: Mapping[str, Any]) -> CodexHookResult:
        if event not in SUPPORTED_EVENTS:
            return CodexHookResult(status="unsupported", error=f"unsupported event: {event}")
        # Codex's UserPromptSubmit and Stop input contracts require turn_id.
        # A content hash or session fallback would merge separate native turns
        # and make a repeated prompt disappear from durable capture.
        if event in CAPTURE_EVENTS and not _text(payload.get("turn_id")):
            return CodexHookResult(
                status="manual_mcp",
                degraded=True,
                error="Codex hook turn_id is required for capture",
            )
        try:
            context = self._context(payload)
        except Exception as exc:
            return CodexHookResult(status="manual_mcp", degraded=True, error=str(exc))

        captured = 0
        ids: list[str] = []
        degraded = False
        context_text = ""

        if event in CAPTURE_EVENTS:
            if event == "Stop" and payload.get("stop_hook_active") is True:
                return CodexHookResult(status="skipped", context=context_text)
            role = "user" if event == "UserPromptSubmit" else "assistant"
            content = _text(payload.get("prompt" if role == "user" else "last_assistant_message"))
            if content:
                accepted, event_id, queued = await self._capture(
                    context,
                    event=event,
                    role=role,
                    content=content,
                    turn_id=_text(payload.get("turn_id")),
                    occurred_at=_timestamp(payload),
                )
                if accepted:
                    captured = 1
                    ids.append(event_id)
                degraded = degraded or queued or not accepted
        elif event in {"PreCompact", "SessionEnd"}:
            degraded = not await self._maintenance(context)

        if event in RECALL_EVENTS:
            try:
                context_text, retrieve_degraded = await self._retrieve(context, event, payload)
                degraded = degraded or retrieve_degraded
            except Exception:
                degraded = True

        return CodexHookResult(
            status="degraded" if degraded else "ok",
            captured=captured,
            context=context_text,
            event_ids=tuple(ids),
            degraded=degraded,
        )


# Keep the longer name available to callers that treat host adapters uniformly.
CodexHookAdapter = CodexAdapter
HookResult = CodexHookResult


def hook_output(event: str, result: CodexHookResult) -> dict[str, Any]:
    """Render only documented, non-blocking Codex output fields."""
    if event == "Stop":
        # Stop requires JSON on stdout.  Never return decision:block or a
        # continuation reason: memory capture must not create a new turn.
        return {"continue": True}
    # The published PostCompact output schema does not define
    # hookSpecificOutput/additionalContext.  It is a lifecycle notification,
    # so keep the response valid and let the next native prompt/session-start
    # boundary request context through its supported output contract.
    if event == "PostCompact":
        return {"continue": True} if result.context else {}
    if event in {"SessionStart", "UserPromptSubmit"} and result.context:
        return {
            "continue": True,
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": result.context,
            },
        }
    return {}


async def run_hook(
    event: str,
    payload: Mapping[str, Any],
    *,
    adapter: CodexAdapter | None = None,
) -> CodexHookResult:
    return await (adapter or CodexAdapter()).handle(event, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mnemos Codex lifecycle hook")
    parser.add_argument("event", nargs="?", default="")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return 0
        event = _text(payload.get("hook_event_name")) or args.event
        if not event:
            return 0
        result = asyncio.run(run_hook(event, payload))
        output = hook_output(event, result)
        if output:
            print(json.dumps(output, separators=(",", ":")))
    except Exception:
        # Hook failures must not block coding or expose captured content.
        if args.event == "Stop":
            print(json.dumps({"continue": True}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
