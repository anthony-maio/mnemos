"""Claude Code command-hook adapter for the local Mnemos service.

The adapter is deliberately transport-only.  It resolves a configured scope,
sanitizes through the service client's outbox, and records a checkpoint only
after the client reports a committed or queued event.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..client import MnemosClient
from .diagnostics import emit_hook_diagnostic
from ..scopes import ScopeContext, ScopeRegistry, default_registry_path as shared_registry_path
from ..routing_review import reason_for_context_error, record_unresolved_routing_failure
from ..service import _protect_private_path, default_runtime_directory

MAX_TRANSCRIPT_LINE = 200_000
MAX_TRANSCRIPT_BYTES_PER_HOOK = 2_000_000
MAX_CONTEXT_CHARS = 8_000
TRANSCRIPT_UNAVAILABLE = "transcript_unavailable"
CAPTURE_NOT_ACKNOWLEDGED = "capture_not_acknowledged"


@dataclass(frozen=True)
class TranscriptDelta:
    role: str
    content: str
    source_ref: str
    occurred_at: datetime
    next_offset: int


@dataclass(frozen=True)
class HookResult:
    """Adapter result, including whether a host event is durably accepted."""

    status: str
    committed: bool
    queued: bool = False
    captured: int = 0
    context: str = ""
    error: str | None = None


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def default_registry_path() -> Path:
    return shared_registry_path()


def default_checkpoint_directory() -> Path:
    return _env_path(
        "MNEMOS_CLAUDE_CHECKPOINT_DIR",
        default_runtime_directory() / "claude-checkpoints",
    )


def load_registry(path: str | Path | None = None) -> ScopeRegistry:
    """Load the persisted setup registry, failing closed when it is absent."""
    target = Path(path).expanduser() if path is not None else default_registry_path()
    if not target.exists():
        raise ValueError("Mnemos scope registry is not configured")
    return ScopeRegistry.load(target)


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    pieces: list[str] = []
    for block in value:
        if isinstance(block, str):
            pieces.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "".join(pieces)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _transcript_identity(path: Path, prefix: bytes) -> str:
    """Identify the underlying transcript file while allowing normal appends."""
    stat = path.stat()
    if stat.st_ino:
        return f"{path.resolve()}:{stat.st_dev}:{stat.st_ino}"
    digest = hashlib.sha256(prefix).hexdigest()
    return f"{path.resolve()}:prefix:{digest}"


def _current_transcript_identity(path: Path) -> str:
    with path.open("rb") as handle:
        return _transcript_identity(path, handle.read(4096))


def read_transcript_deltas(
    path: str | Path,
    *,
    offset: int = 0,
    max_bytes: int = MAX_TRANSCRIPT_BYTES_PER_HOOK,
) -> tuple[list[TranscriptDelta], int, str]:
    """Read complete new JSONL records and return the next safe byte offset."""
    target = Path(path).expanduser()
    with target.open("rb") as handle:
        handle.seek(0)
        prefix = handle.read(4096)
        identity = _transcript_identity(target, prefix)
        size = target.stat().st_size
        start = offset if 0 <= offset <= size else 0
        handle.seek(start)
        consumed = 0
        next_offset = start
        deltas: list[TranscriptDelta] = []
        while consumed < max_bytes:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            next_offset = handle.tell()
            consumed += len(line)
            if len(line) > MAX_TRANSCRIPT_LINE:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            message = record.get("message")
            role = record.get("role")
            if isinstance(message, dict):
                role = message.get("role", role)
                content = _text_content(message.get("content"))
            else:
                content = _text_content(record.get("content"))
            if role not in {"user", "assistant"} and record.get("type") in {
                "user",
                "assistant",
            }:
                role = record["type"]
            if role not in {"user", "assistant"} or not content.strip():
                continue
            source_ref = str(record.get("uuid") or record.get("id") or line_start)
            deltas.append(
                TranscriptDelta(
                    role=role,
                    content=content.strip(),
                    source_ref=source_ref,
                    occurred_at=_parse_timestamp(record.get("timestamp")),
                    next_offset=next_offset,
                )
            )
        return deltas, next_offset, identity


class ClaudeCodeAdapter:
    """Scope-aware Claude Code adapter using only the Mnemos client API."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        registry: ScopeRegistry | None = None,
        registry_path: str | Path | None = None,
        user_profile_id: str | None = None,
        checkpoint_directory: str | Path | None = None,
        top_k: int = 5,
    ) -> None:
        self.client = client or MnemosClient()
        self.registry = registry or load_registry(registry_path)
        # Do not invent a shared profile when the host did not provide one.
        # Scope resolution must fail closed rather than merging users.
        self.user_profile_id = user_profile_id or os.environ.get("MNEMOS_USER_PROFILE_ID")
        self.checkpoint_directory = Path(
            checkpoint_directory or default_checkpoint_directory()
        ).expanduser()
        self.top_k = max(0, min(int(top_k), 100))

    def _context(self, payload: Mapping[str, Any]) -> ScopeContext:
        cwd = payload.get("cwd")
        session_id = payload.get("session_id")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError("Claude hook cwd is required")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Claude hook session_id is required")
        if not self.user_profile_id:
            raise ValueError("MNEMOS_USER_PROFILE_ID is required")
        workspace_id = payload.get("workspace_id")
        if workspace_id is not None and not isinstance(workspace_id, str):
            raise ValueError("workspace_id must be a string")
        return self.registry.resolve(
            cwd,
            user_profile_id=self.user_profile_id,
            session_id=session_id,
            host="claude-code",
            workspace_id=workspace_id,
        )

    @staticmethod
    def _event_id(session_id: str, source_ref: str, role: str, content: str) -> str:
        material = "\0".join(("claude-code", session_id, source_ref, role, content))
        return "claude:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _checkpoint_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.checkpoint_directory / f"{digest}.json"

    @staticmethod
    def _content_fingerprint(role: str, content: str) -> str:
        return hashlib.sha256(f"{role}\0{content}".encode("utf-8")).hexdigest()

    @staticmethod
    def _pending_records(checkpoint: Mapping[str, Any]) -> list[dict[str, str | None]]:
        raw = checkpoint.get("pending_dedup", [])
        if not isinstance(raw, list):
            return []
        records: list[dict[str, str | None]] = []
        for item in raw:
            if isinstance(item, str) and item:
                records.append({"fingerprint": item, "source_ref": None})
            elif isinstance(item, Mapping):
                fingerprint = item.get("fingerprint")
                source_ref = item.get("source_ref")
                if isinstance(fingerprint, str) and fingerprint:
                    records.append(
                        {
                            "fingerprint": fingerprint,
                            "source_ref": source_ref if isinstance(source_ref, str) else None,
                        }
                    )
        return records

    def _read_checkpoint(self, session_id: str) -> dict[str, Any]:
        target = self._checkpoint_path(session_id)
        try:
            _protect_private_path(target)
            value = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            raise ValueError("Claude checkpoint is corrupt") from exc
        except OSError as exc:
            raise RuntimeError("Claude checkpoint could not be read") from exc
        if not isinstance(value, dict):
            raise ValueError("Claude checkpoint is corrupt")
        if "offset" in value:
            offset = value["offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError("Claude checkpoint is corrupt")
        return value

    def _write_checkpoint(self, session_id: str, value: Mapping[str, Any]) -> None:
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        _protect_private_path(self.checkpoint_directory)
        target = self._checkpoint_path(session_id)
        descriptor_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=self.checkpoint_directory
        )
        os.close(descriptor_fd)
        temporary = Path(temporary_name)
        try:
            _protect_private_path(temporary)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(dict(value), handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _protect_private_path(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _remember_pending_duplicate(
        self, session_id: str, role: str, content: str, *, source_ref: str
    ) -> None:
        checkpoint = self._read_checkpoint(session_id)
        pending = self._pending_records(checkpoint)
        fingerprint = self._content_fingerprint(role, content)
        if not any(
            item["fingerprint"] == fingerprint and item["source_ref"] == source_ref
            for item in pending
        ):
            pending.append({"fingerprint": fingerprint, "source_ref": source_ref})
        checkpoint["pending_dedup"] = pending[-16:]
        self._write_checkpoint(session_id, checkpoint)

    async def _capture(
        self,
        context: ScopeContext,
        *,
        role: str,
        content: str,
        source_ref: str,
        occurred_at: datetime,
    ) -> bool:
        result = await self.client.capture(
            {
                "event_id": self._event_id(context.session_id, source_ref, role, content),
                "context": context.model_dump(mode="json"),
                "occurred_at": occurred_at.isoformat(),
                "role": role,
                "content": content,
                "source_kind": "claude_code_transcript",
                "source_ref": source_ref,
            }
        )
        if not isinstance(result, Mapping):
            return False
        return (
            result.get("committed") is True
            or result.get("status") == "committed"
            or result.get("status") == "queued"
        )

    async def _capture_transcript(
        self, payload: Mapping[str, Any], context: ScopeContext
    ) -> tuple[int, set[str], str | None]:
        checkpoint = self._read_checkpoint(context.session_id)
        transcript = payload.get("transcript_path")
        if not isinstance(transcript, str) or not transcript.strip():
            return 0, set(), TRANSCRIPT_UNAVAILABLE
        target = Path(transcript).expanduser()
        try:
            current_identity = _current_transcript_identity(target)
        except OSError:
            return 0, set(), TRANSCRIPT_UNAVAILABLE
        checkpoint_path = str(target.resolve())
        raw_offset = checkpoint.get("offset", 0)
        if type(raw_offset) is not int or raw_offset < 0:
            raise ValueError("Claude checkpoint is corrupt")
        offset = raw_offset
        if (
            checkpoint.get("path") != checkpoint_path
            or checkpoint.get("identity") != current_identity
        ):
            offset = 0
        try:
            deltas, next_offset, identity = read_transcript_deltas(
                transcript,
                offset=offset,
            )
        except OSError:
            return 0, set(), TRANSCRIPT_UNAVAILABLE
        captured = 0
        contents: set[str] = set()
        pending_records = self._pending_records(checkpoint)
        for delta in deltas:
            fingerprint = self._content_fingerprint(delta.role, delta.content)
            pending_index = next(
                (
                    index
                    for index, item in enumerate(pending_records)
                    if item["fingerprint"] == fingerprint
                ),
                None,
            )
            if pending_index is not None:
                pending_records.pop(pending_index)
                contents.add(delta.content)
                continue
            if not await self._capture(
                context,
                role=delta.role,
                content=delta.content,
                source_ref=delta.source_ref,
                occurred_at=delta.occurred_at,
            ):
                return captured, contents, CAPTURE_NOT_ACKNOWLEDGED
            captured += 1
            contents.add(delta.content)
        if deltas or next_offset != offset or checkpoint.get("identity") != identity:
            self._write_checkpoint(
                context.session_id,
                {
                    "path": checkpoint_path,
                    "identity": identity,
                    "offset": next_offset,
                    "pending_dedup": pending_records[-16:],
                },
            )
        return captured, contents, None

    @staticmethod
    def _query(payload: Mapping[str, Any], event: str) -> str:
        prompt = payload.get("prompt")
        if event == "UserPromptSubmit" and isinstance(prompt, str):
            return prompt.strip()
        return "project context recent work"

    async def retrieve_context(self, event: str, payload: Mapping[str, Any]) -> HookResult:
        context = self._context(payload)
        if os.environ.get("MNEMOS_AUTO_RETRIEVE", "true").lower() in {"0", "false", "no", "off"}:
            return HookResult(status="skipped", committed=False)
        query = self._query(payload, event)
        if len(query) < 3:
            return HookResult(status="skipped", committed=False)
        result = await self.client.retrieve(
            {
                "query": query,
                "context": context.model_dump(mode="json"),
                "allowed_scopes": ["project", "workspace", "global"],
                "top_k": self.top_k,
                "max_context_tokens": 1200,
                "profile": "baseline",
            }
        )
        facts = result.get("facts", []) if isinstance(result, Mapping) else []
        lines: list[str] = []
        for index, fact in enumerate(facts, 1):
            if not isinstance(fact, Mapping) or not isinstance(fact.get("content"), str):
                continue
            content = " ".join(fact["content"].split())
            if content:
                lines.append(f"{index}. {content[:800]}")
        context_text = "\n".join(lines)[:MAX_CONTEXT_CHARS]
        return HookResult(status="ok", committed=False, context=context_text)

    async def handle(
        self, event: str, payload: Mapping[str, Any], *, include_retrieval: bool = True
    ) -> HookResult:
        if event in {"SessionStart", "UserPromptSubmit"}:
            try:
                context = self._context(payload)
            except Exception as exc:
                reason = reason_for_context_error(exc)
                record_unresolved_routing_failure(
                    host="claude-code",
                    event=event,
                    reason=reason,
                    session_id=(
                        payload.get("session_id")
                        if isinstance(payload.get("session_id"), str)
                        else None
                    ),
                    project_hint=(
                        payload.get("cwd") if isinstance(payload.get("cwd"), str) else None
                    ),
                    configured_profile=self.user_profile_id,
                )
                return HookResult(status="degraded", committed=False, error=reason)

            async def recall() -> HookResult:
                if not include_retrieval:
                    return HookResult(status="skipped", committed=False)
                try:
                    return await self.retrieve_context(event, payload)
                except Exception as exc:
                    return HookResult(status="degraded", committed=False, error=type(exc).__name__)

            if event == "SessionStart":
                return await recall()
            prompt = payload.get("prompt")
            captured = 0
            capture_error: str | None = None
            if isinstance(prompt, str) and prompt.strip():
                source_ref = str(
                    payload.get("prompt_id")
                    or f"prompt:{hashlib.sha256(prompt.encode()).hexdigest()}"
                )
                if await self._capture(
                    context,
                    role="user",
                    content=prompt.strip(),
                    source_ref=source_ref,
                    occurred_at=datetime.now(timezone.utc),
                ):
                    captured += 1
                    self._remember_pending_duplicate(
                        context.session_id,
                        "user",
                        prompt.strip(),
                        source_ref=source_ref,
                    )
                else:
                    capture_error = CAPTURE_NOT_ACKNOWLEDGED
            retrieved = await recall()
            return HookResult(
                status=(
                    "degraded"
                    if capture_error is not None or retrieved.status == "degraded"
                    else "ok"
                ),
                committed=captured > 0,
                captured=captured,
                context=retrieved.context,
                error=capture_error or retrieved.error,
            )
        if event in {"Stop", "PreCompact", "PostCompact"}:
            try:
                context = self._context(payload)
            except Exception as exc:
                reason = reason_for_context_error(exc)
                record_unresolved_routing_failure(
                    host="claude-code",
                    event=event,
                    reason=reason,
                    session_id=(
                        payload.get("session_id")
                        if isinstance(payload.get("session_id"), str)
                        else None
                    ),
                    project_hint=(
                        payload.get("cwd") if isinstance(payload.get("cwd"), str) else None
                    ),
                    configured_profile=self.user_profile_id,
                )
                return HookResult(status="degraded", committed=False, error=reason)
            captured, transcript_contents, transcript_error = await self._capture_transcript(
                payload, context
            )
            if event == "Stop":
                assistant = payload.get("last_assistant_message")
                assistant_content = assistant.strip() if isinstance(assistant, str) else ""
                checkpoint = self._read_checkpoint(context.session_id)
                pending_records = self._pending_records(checkpoint)
                source_ref = str(payload.get("prompt_id") or "last_assistant_message")
                if (
                    assistant_content
                    and assistant_content not in transcript_contents
                    and not any(
                        item["fingerprint"]
                        == self._content_fingerprint("assistant", assistant_content)
                        and item["source_ref"] == source_ref
                        for item in pending_records
                    )
                ):
                    if await self._capture(
                        context,
                        role="assistant",
                        content=assistant_content,
                        source_ref=source_ref,
                        occurred_at=datetime.now(timezone.utc),
                    ):
                        captured += 1
                        self._remember_pending_duplicate(
                            context.session_id,
                            "assistant",
                            assistant_content,
                            source_ref=source_ref,
                        )
                    else:
                        transcript_error = CAPTURE_NOT_ACKNOWLEDGED
            return HookResult(
                status="degraded" if transcript_error else "ok",
                committed=captured > 0,
                captured=captured,
                error=transcript_error,
            )
        return HookResult(status="skipped", committed=False, error=f"unsupported event: {event}")


def _native_output(event: str, result: HookResult) -> dict[str, Any]:
    if not result.context:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": result.context,
        }
    }


async def run_hook(event: str, payload: Mapping[str, Any]) -> HookResult:
    adapter = ClaudeCodeAdapter()
    return await adapter.handle(event, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", nargs="?", default="")
    args = parser.parse_args(argv)
    event = args.event
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return 0
        event = str(payload.get("hook_event_name") or args.event)
        if not event:
            return 0
        result = asyncio.run(run_hook(event, payload))
        if result.status == "degraded":
            emit_hook_diagnostic(host="claude-code", event=event, status=result.status)
        if event in {"SessionStart", "UserPromptSubmit"}:
            output = _native_output(event, result)
            if output:
                print(json.dumps(output, separators=(",", ":")))
        return 0
    except Exception as exc:
        # Hook capture is best effort and must never block Claude or expose
        # payloads and credentials through stderr.
        emit_hook_diagnostic(host="claude-code", event=event, status="error", exception=exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
