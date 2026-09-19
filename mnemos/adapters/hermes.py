"""Optional Hermes Agent memory provider for the local Mnemos service.

Hermes calls providers synchronously.  Mnemos writes therefore go through a
bounded background enqueue, while prefetch and required compression
checkpoints use the injected service bridge directly.  The adapter remains
importable when Hermes is not installed so setup and capability reporting can
fail honestly.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

from ..bridge import BridgeClient
from ..client import MnemosClient
from ..scopes import ScopeContext, ScopeRegistry, default_registry_path as shared_registry_path
from .context import format_retrieved_context
from .hermes_occurrences import occurrence_time
from .hermes_tools import dispatch_tool, get_tool_schemas

try:  # Hermes is an optional host dependency and is not installed in Mnemos' venv.
    from agent.memory_provider import MemoryProvider as _HermesMemoryProvider
    from agent.memory_provider import spawn_context_thread as _spawn_context_thread

    HERMES_API_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the normal Windows install.
    HERMES_API_AVAILABLE = False

    class _HermesMemoryProvider:  # type: ignore[no-redef]
        """Import-only fallback; never reports itself available."""

        pre_compress_checkpoint_api_version = 1

    _spawn_context_thread = None


MAX_CONTEXT_CHARS = 8_000
MAX_MESSAGE_CHARS = 100_000


@dataclass(frozen=True)
class HermesCapabilities:
    api_available: bool
    provider_name: str
    fallback: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_available": self.api_available,
            "provider_name": self.provider_name,
            "fallback": self.fallback,
        }


def default_registry_path() -> Path:
    return shared_registry_path()


def capabilities() -> HermesCapabilities:
    return HermesCapabilities(
        api_available=HERMES_API_AVAILABLE,
        provider_name="mnemos",
        fallback="manual_mcp",
    )


def load_registry(path: str | Path | None = None) -> ScopeRegistry:
    target = Path(path).expanduser() if path is not None else default_registry_path()
    if not target.exists():
        raise ValueError("Mnemos scope registry is not configured")
    return ScopeRegistry.load(target)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()[:MAX_MESSAGE_CHARS]
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, Mapping) and item.get("type") in {"text", "input_text"}:
                if isinstance(item.get("text"), str):
                    pieces.append(item["text"])
        return "".join(pieces).strip()[:MAX_MESSAGE_CHARS]
    return ""


def _run_sync(awaitable: Any) -> Any:
    """Run an async service call from Hermes' synchronous provider contract."""
    if not inspect.isawaitable(awaitable):
        return awaitable

    async def invoke() -> Any:
        return await awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(invoke())
    result: list[Any] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(invoke()))
        except BaseException as exc:  # re-raised in the caller thread.
            failure.append(exc)

    thread = threading.Thread(target=run, name="mnemos-hermes-call", daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    if thread.is_alive():
        raise TimeoutError("Mnemos service call exceeded the Hermes adapter timeout")
    if failure:
        raise failure[0]
    return result[0] if result else None


class MnemosHermesProvider(_HermesMemoryProvider):  # type: ignore[misc]  # Optional Hermes imports are Any when absent.
    """Hermes MemoryProvider backed by the configured local Mnemos service."""

    pre_compress_checkpoint_api_version = 2

    def __init__(
        self,
        *,
        client: Any | None = None,
        registry: ScopeRegistry | None = None,
        registry_path: str | Path | None = None,
        user_profile_id: str | None = None,
        descriptor_path: str | Path | None = None,
        max_context_tokens: int = 1200,
    ) -> None:
        if client is not None:
            self.client = client
            self.transport = "injected"
        elif os.environ.get("MNEMOS_BRIDGE_COMMAND"):
            self.client = BridgeClient.from_env()
            self.transport = "explicit_bridge"
        else:
            self.client = MnemosClient(descriptor_path=descriptor_path)
            self.transport = "local_client"
        self.registry = registry
        configured_registry_path = str(registry_path).strip() if registry_path is not None else ""
        self.registry_path = (
            Path(configured_registry_path).expanduser()
            if configured_registry_path
            else default_registry_path()
        )
        self.user_profile_id = user_profile_id or os.environ.get("MNEMOS_USER_PROFILE_ID")
        self.max_context_tokens = max(0, min(int(max_context_tokens), 32_000))
        self.session_id = ""
        self.hermes_home = ""
        self.cwd = ""
        self._context: ScopeContext | None = None
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.RLock()
        self._mirror_ids: set[str] = set()
        self._occurrence_times: dict[str, str] = {}
        self._lifecycle_diagnostics: list[dict[str, str]] = []
        self._failed_capture_ids: set[str] = set()
        self._failed_checkpoint = False

    @property
    def name(self) -> str:
        return "mnemos"

    def unavailable_reason(self) -> str:
        if not HERMES_API_AVAILABLE:
            return "Hermes Agent memory_provider API is not installed"
        if self.registry is None:
            if not self.registry_path or not self.registry_path.is_file():
                return "Mnemos scope registry is not configured"
            try:
                ScopeRegistry.load(self.registry_path)
            except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
                return "Mnemos scope registry is invalid"
        if not self.user_profile_id:
            return "MNEMOS_USER_PROFILE_ID or Hermes agent_identity is required"
        return ""

    def is_available(self) -> bool:
        return not self.unavailable_reason()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._context = None
        next_home = str(kwargs.get("hermes_home") or "").strip()
        if next_home != self.hermes_home:
            with self._threads_lock:
                self._occurrence_times.clear()
        self.session_id = str(session_id or "").strip()
        self.hermes_home = next_home
        self.cwd = str(kwargs.get("cwd") or "").strip()
        self.user_profile_id = (
            self.user_profile_id
            or str(kwargs.get("agent_identity") or "").strip()
            or os.environ.get("MNEMOS_USER_PROFILE_ID")
        )
        if self.registry is None:
            self.registry = load_registry(self.registry_path)
        if not self.session_id:
            raise ValueError("Hermes session_id is required")
        if not self.cwd:
            raise ValueError("Hermes session cwd is required")
        if not self.user_profile_id:
            raise ValueError("Hermes agent_identity or MNEMOS_USER_PROFILE_ID is required")
        self._context = self._resolve_context(self.session_id)

    def _resolve_context(self, session_id: str) -> ScopeContext:
        from ..routing_review import reason_for_context_error, record_unresolved_routing_failure

        try:
            if self.registry is None:
                raise ValueError("Mnemos scope registry is not configured")
            if not self.user_profile_id:
                raise ValueError("Hermes profile identity is not configured")
            if not self.cwd:
                raise ValueError("Hermes session cwd is required")
            return self.registry.resolve(
                self.cwd,
                user_profile_id=self.user_profile_id,
                session_id=session_id,
                host="hermes",
            )
        except Exception as exc:
            record_unresolved_routing_failure(
                host="hermes",
                event="ContextResolve",
                reason=reason_for_context_error(exc),
                session_id=session_id,
                project_hint=self.cwd,
                configured_profile=self.user_profile_id,
            )
            raise

    def _current_context(self, session_id: str = "") -> ScopeContext:
        chosen = str(session_id or self.session_id).strip()
        if not chosen:
            raise ValueError("Hermes session_id is required")
        if self._context is None or self._context.session_id != chosen:
            self._context = self._resolve_context(chosen)
        return self._context

    @staticmethod
    def _event_id(context: ScopeContext, *, role: str, content: str, source: str) -> str:
        material = json.dumps(
            [
                context.user_profile_id,
                context.project_id,
                context.workspace_id,
                context.session_id,
                role,
                source,
                content,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "hermes:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _capture_payload(
        self, context: ScopeContext, *, role: str, content: str, source: str
    ) -> dict[str, Any]:
        event_id = self._event_id(context, role=role, content=content, source=source)
        with self._threads_lock:
            occurred_at = self._occurrence_times.get(event_id)
            if occurred_at is None:
                occurred_at = datetime.now(timezone.utc).isoformat()
                if self.hermes_home:
                    occurred_at = occurrence_time(self.hermes_home, event_id, occurred_at)
                else:
                    self._record_lifecycle_diagnostic(
                        action="capture_identity", target="", status="nonpersistent_timestamp"
                    )
                self._occurrence_times[event_id] = occurred_at
        return {
            "event_id": event_id,
            "context": context.model_dump(mode="json"),
            "occurred_at": occurred_at,
            "role": role,
            "content": content,
            "source_kind": "hermes_memory_provider",
            "source_ref": source,
        }

    def _capture_now(self, payload: Mapping[str, Any]) -> bool:
        try:
            result = self.client.capture(dict(payload))
            result = _run_sync(result)
        except Exception:
            return False
        if not isinstance(result, Mapping):
            return False
        return bool(
            result.get("committed") is True or result.get("status") in {"committed", "queued"}
        )

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        if _spawn_context_thread is not None:
            thread = cast(threading.Thread, _spawn_context_thread(target, name=name))
        else:
            thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        with self._threads_lock:
            self._threads.append(thread)
        return thread

    def _enqueue_messages(
        self,
        context: ScopeContext,
        messages: list[tuple[str, str]],
        source: str,
        *,
        on_failure: Callable[[dict[str, Any]], None] | None = None,
        diagnostic_action: str = "capture",
        diagnostic_target: str = "",
    ) -> None:
        payloads = [
            self._capture_payload(context, role=role, content=content, source=source)
            for role, content in messages
            if role in {"user", "assistant"} and content
        ]
        self._enqueue_payloads(
            payloads,
            on_failure=on_failure,
            diagnostic_action=diagnostic_action,
            diagnostic_target=diagnostic_target or source,
        )

    def _transcript_payloads(
        self,
        context: ScopeContext,
        messages: list[tuple[str, str]],
        *,
        provisional: bool = False,
    ) -> list[dict[str, Any]]:
        # Hermes supplies the cumulative transcript. Bind each occurrence to
        # its own prefix so appending a later turn cannot rename earlier ones.
        # Position also distinguishes repeated identical text in one session.
        prefix = hashlib.sha256()
        payloads = []
        namespace = "provisional" if provisional else "transcript"
        for index, (role, content) in enumerate(messages):
            prefix.update(
                json.dumps([role, content], ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\0"
            )
            payloads.append(
                self._capture_payload(
                    context,
                    role=role,
                    content=content,
                    source=f"{namespace}:v1:{index}:{prefix.hexdigest()}",
                )
            )
            if provisional:
                payloads[-1]["identity_status"] = "provisional_unassociated"
        return payloads

    def _enqueue_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        on_failure: Callable[[dict[str, Any]], None] | None = None,
        diagnostic_action: str = "capture",
        diagnostic_target: str = "transcript",
    ) -> None:

        def write() -> None:
            for payload in payloads:
                if self._capture_now(payload):
                    with self._threads_lock:
                        self._failed_capture_ids.discard(payload["event_id"])
                else:
                    with self._threads_lock:
                        self._failed_capture_ids.add(payload["event_id"])
                    self._record_lifecycle_diagnostic(
                        action=diagnostic_action,
                        target=diagnostic_target,
                        status="capture_failed",
                    )
                    if on_failure is not None:
                        on_failure(payload)

        if payloads:
            self._spawn(write, "mnemos-hermes-sync")

    def _flush(self, timeout: float = 3.0) -> bool:
        deadline = datetime.now(timezone.utc).timestamp() + max(0.0, timeout)
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
            remaining = max(0.0, deadline - datetime.now(timezone.utc).timestamp())
            thread.join(remaining)
        with self._threads_lock:
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            return not self._threads and not self._failed_capture_ids

    def _record_lifecycle_diagnostic(self, *, action: str, target: str, status: str) -> None:
        diagnostic = {
            "action": action,
            "target_digest": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "status": status,
        }
        with self._threads_lock:
            self._lifecycle_diagnostics.append(diagnostic)
            del self._lifecycle_diagnostics[:-32]

    def get_lifecycle_diagnostics(self) -> list[dict[str, str]]:
        """Return bounded, content-free Hermes lifecycle diagnostics."""
        with self._threads_lock:
            return [dict(item) for item in self._lifecycle_diagnostics]

    def _format_context(self, result: Any) -> str:
        return format_retrieved_context(result, max_chars=MAX_CONTEXT_CHARS)

    def system_prompt_block(self) -> str:
        return (
            "Mnemos provides shared local memory through the mnemos_* tools. "
            "Retrieve or inspect a fact before correcting or forgetting its exact ID and version. "
            "Remember and correct return a durable job; check mnemos_job_status for completion. "
            "Built-in Hermes memory removals affect its own store; use mnemos_forget to remove "
            "a shared Mnemos fact."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        query = _text(query)
        if len(query) < 3:
            return ""
        try:
            context = self._current_context(session_id)
            result = self.client.retrieve(
                {
                    "query": query,
                    "context": context.model_dump(mode="json"),
                    "allowed_scopes": ["project", "workspace", "global"],
                    "top_k": 5,
                    "max_context_tokens": self.max_context_tokens,
                    "profile": "baseline",
                }
            )
            rendered = self._format_context(_run_sync(result))
            return rendered
        except Exception:
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Hermes calls this after a completed turn. Keep this best-effort warmup
        # asynchronous; ordinary prefetch always reads the current service.
        if not _text(query):
            return

        def warm() -> None:
            self.prefetch(query, session_id=session_id)

        self._spawn(warm, "mnemos-hermes-prefetch")

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        turn_author: Optional[Dict[str, Any]] = None,
    ) -> None:
        del turn_author
        try:
            context = self._current_context(session_id)
        except Exception:
            return
        selected: list[tuple[str, str]] = []
        if messages:
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                role = str(message.get("role") or "").strip().lower()
                content = _text(message.get("content"))
                if role in {"user", "assistant"} and content:
                    selected.append((role, content))
        provisional = not selected
        if provisional:
            selected = [("user", _text(user_content)), ("assistant", _text(assistant_content))]
        selected = [(role, content) for role, content in selected if content]
        if provisional and selected:
            self._record_lifecycle_diagnostic(
                action="capture_identity", target="", status="transcript_identity_unavailable"
            )
        self._enqueue_payloads(
            self._transcript_payloads(context, selected, provisional=provisional)
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self.sync_turn(
            "",
            "",
            session_id=self.session_id,
            messages=messages,
        )
        self._flush(timeout=3.0)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        del parent_session_id, rewound, kwargs
        chosen = str(new_session_id or "").strip()
        if not chosen:
            return
        self._flush(timeout=3.0)
        self.session_id = chosen
        self._context = self._resolve_context(chosen)
        if reset:
            self._mirror_ids.clear()

    def on_pre_compress(
        self,
        messages: List[Dict[str, Any]],
        *,
        require_checkpoint: bool = False,
    ) -> str:
        try:
            context = self._current_context()
        except Exception:
            if require_checkpoint:
                raise RuntimeError("Mnemos Hermes checkpoint has no validated scope")
            return ""
        evidence: list[tuple[str, str]] = []
        for message in messages or []:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "").strip().lower()
            content = _text(message.get("content"))
            if role in {"user", "assistant"} and content:
                evidence.append((role, content))
        accepted = True
        for payload in self._transcript_payloads(context, evidence):
            accepted = self._capture_now(payload) and accepted
        self._failed_checkpoint = not accepted
        if require_checkpoint and not accepted:
            raise RuntimeError("Mnemos Hermes checkpoint was not durably accepted")
        return (
            f"Mnemos checkpoint captured {len(evidence)} user/assistant messages."
            if accepted and evidence
            else ""
        )

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action not in {"add", "replace", "remove"}:
            return
        if action != "add":
            # Hermes target categories are not Mnemos fact IDs. Without a verified
            # mapping and expected version, mirroring would re-ingest removed
            # content or leave replaced assertions stale.
            self._record_lifecycle_diagnostic(action=action, target=target, status="unresolved")
            return
        content = _text(content)
        if not content:
            return
        metadata = metadata or {}
        try:
            session_id = str(metadata.get("session_id") or self.session_id)
            context = self._current_context(session_id)
        except Exception:
            return
        source = f"memory-write:{action}:{target}:{hashlib.sha256(content.encode()).hexdigest()}"
        event_id = self._event_id(context, role="user", content=content, source=source)
        with self._threads_lock:
            if event_id in self._mirror_ids:
                return
            self._mirror_ids.add(event_id)

        def failed(_: dict[str, Any]) -> None:
            with self._threads_lock:
                self._mirror_ids.discard(event_id)

        self._enqueue_messages(
            context,
            [("user", content)],
            source,
            on_failure=failed,
            diagnostic_action="add",
            diagnostic_target=target,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return get_tool_schemas()

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        try:
            context = self._current_context()
        except Exception:
            return json.dumps(
                {"status": "rejected", "error": "Mnemos provider has no validated session scope"}
            )
        return cast(str, _run_sync(dispatch_tool(self.client, context, tool_name, args)))

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        target_dir = Path(hermes_home).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "mnemos.json"
        payload = {
            key: values[key]
            for key in (
                "descriptor_path",
                "registry_path",
                "user_profile_id",
                "max_context_tokens",
                "bridge_command",
            )
            if key in values and values[key] not in (None, "")
        }
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "user_profile_id",
                "description": "Registered Mnemos user profile identity",
                "required": True,
                "env_var": "MNEMOS_USER_PROFILE_ID",
            },
            {
                "key": "registry_path",
                "description": "Path to the Mnemos scope registry",
                "type": "text",
            },
            {
                "key": "descriptor_path",
                "description": "Explicit Windows runtime descriptor path for WSL bridges",
                "type": "text",
            },
            {
                "key": "bridge_command",
                "description": "Explicit JSON argv for the Windows Mnemos bridge",
                "type": "text",
                "env_var": "MNEMOS_BRIDGE_COMMAND",
            },
        ]

    def shutdown(self) -> None:
        self._flush(timeout=3.0)


# The shorter name is convenient for direct integration tests and callers.
HermesMemoryProvider = MnemosHermesProvider


def register(ctx: Any) -> bool:
    """Hermes entry-point callback; do not register a fake provider."""
    provider = MnemosHermesProvider()
    if not provider.is_available():
        return False
    register_provider = getattr(ctx, "register_memory_provider", None)
    if not callable(register_provider):
        return False
    register_provider(provider)
    return True
