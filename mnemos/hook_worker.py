"""Experimental resident adapter using the existing authenticated service host.

Raw hook input crosses only this local authenticated boundary in memory. The
ordinary client still sanitizes and fsyncs its outbox before memory-service IO.
No host configuration is changed by starting this explicitly selected worker.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .adapters.claude_code import ClaudeCodeAdapter, _native_output
from .adapters.codex import CodexAdapter, hook_output
from .client import MnemosClient
from .outbox_replay import OutboxReplayPump
from .hook_client import HOOK_PROTOCOL, environment_fingerprint
from .scopes import ScopeRegistry, default_registry_path
from .service import MnemosService, _write_private_json, default_descriptor_path
from .settings import load_settings


def _settings_fingerprint(cwd: str | Path) -> str:
    resolved = load_settings(cwd=cwd)
    value = {
        "settings": resolved.settings.model_dump(mode="json"),
        "llm_provider_explicit": resolved.llm_provider_explicit,
        "warnings": resolved.warnings,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class HookDispatcher:
    """Serialize host events while refreshing settings and membership each time."""

    def __init__(
        self,
        host: str,
        *,
        cwd: str | Path,
        client: MnemosClient | None = None,
        queue_captures: bool = False,
    ) -> None:
        if host not in {"codex", "claude-code"}:
            raise ValueError("Unsupported adapter host")
        if queue_captures and host != "codex":
            raise ValueError("Queued capture is currently supported only for Codex")
        self.cwd = Path(cwd).resolve(strict=True)
        if self.cwd != Path.cwd().resolve():
            raise ValueError("Start the hook worker in its bound working directory")
        self.host = host
        self.environment_sha256 = environment_fingerprint()
        self.settings_sha256 = _settings_fingerprint(cwd)
        self.registry_path = default_registry_path()
        self.profile = os.environ.get("MNEMOS_USER_PROFILE_ID")
        if not self.profile:
            raise ValueError("MNEMOS_USER_PROFILE_ID is required")
        self.client = client if client is not None else MnemosClient()
        self._lock = asyncio.Lock()
        self.queue_captures = queue_captures
        self.replay = None
        if queue_captures:
            # Separate transport/startup state, same durable queue and firewall.
            replay_client = MnemosClient(
                descriptor_path=self.client.descriptor_path,
                outbox=self.client.outbox,
                auto_start=self.client.auto_start,
                startup_timeout=self.client.startup_timeout,
                request_timeout=self.client.request_timeout,
                python_executable=self.client.python_executable,
            )
            self.replay = OutboxReplayPump(replay_client)

    def start(self) -> None:
        if self.replay is not None:
            self.replay.start()

    async def dispatch(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        response = {
            "hook_protocol": HOOK_PROTOCOL,
            "request_id": request.get("request_id"),
            "host": self.host,
            "environment_sha256": self.environment_sha256,
            "status": "declined",
            "dispatched": False,
        }
        if (
            operation != "hook"
            or type(request.get("hook_protocol")) is not int
            or request["hook_protocol"] != HOOK_PROTOCOL
            or not isinstance(request.get("request_id"), str)
            or not request["request_id"]
            or request.get("host") != self.host
            or request.get("environment_sha256") != self.environment_sha256
            or not isinstance(request.get("event"), str)
            or not isinstance(request.get("payload"), dict)
            or not isinstance(request.get("cwd"), str)
            or not request["cwd"]
        ):
            return response
        async with self._lock:
            # Never reuse a profile or settings snapshot after its input changes.
            if environment_fingerprint() != self.environment_sha256:
                return response
            try:
                if (
                    Path(request["cwd"]).resolve(strict=True) != self.cwd
                    or Path.cwd().resolve() != self.cwd
                ):
                    return response
                if _settings_fingerprint(request["cwd"]) != self.settings_sha256:
                    return response
                registry = ScopeRegistry.load(self.registry_path)
            except (OSError, ValueError):
                return response
            self.client.begin_host_event()
            self.start()
            event, payload = request["event"], request["payload"]
            response.update(status="handled", dispatched=True)
            if self.host == "codex":
                codex = CodexAdapter(
                    client=self.client,
                    registry=registry,
                    user_profile_id=self.profile,
                    queue_capture=self.queue_captures,
                )
                result = await codex.handle(event, payload)
                response.update(
                    adapter_status=result.status,
                    degraded=result.degraded or result.status == "degraded",
                    captured=result.captured,
                    output=hook_output(event, result),
                )
                if self.queue_captures:
                    response.update(
                        capture_state="queued" if result.queued else "none",
                        queued=result.queued,
                        event_ids=list(result.event_ids),
                    )
                    if result.queued and self.replay is not None:
                        self.replay.notify()
            else:
                claude = ClaudeCodeAdapter(
                    client=self.client, registry=registry, user_profile_id=self.profile
                )
                claude_result = await claude.handle(event, payload)
                response.update(
                    adapter_status=claude_result.status,
                    degraded=claude_result.status == "degraded",
                    captured=claude_result.captured,
                    reason=claude_result.error,
                    output=(
                        _native_output(event, claude_result)
                        if event in {"SessionStart", "UserPromptSubmit"}
                        else {}
                    ),
                )
            return response

    async def close(self) -> None:
        if self.replay is not None:
            await self.replay.close()
        await self.client.close()


class HookWorker(MnemosService):
    """Separate runtime descriptor and lock for one host's resident adapter."""

    def __init__(
        self,
        host: str,
        *,
        descriptor_path: str | Path,
        cwd: str | Path,
        queue_captures: bool = False,
    ) -> None:
        if queue_captures and host != "codex":
            raise ValueError("Queued capture is currently supported only for Codex")
        target = Path(descriptor_path).expanduser().resolve()
        memory_descriptor = default_descriptor_path().expanduser().resolve()
        if target == memory_descriptor or target.with_suffix(
            ".lock"
        ) == memory_descriptor.with_suffix(".lock"):
            raise ValueError("Hook worker must not use the memory service descriptor")
        self.adapter_host = host
        self.capture_mode = "queued" if queue_captures else "synchronous"
        self.environment_sha256 = environment_fingerprint()

        def create_dispatcher() -> HookDispatcher:
            dispatcher = HookDispatcher(host, cwd=cwd, queue_captures=queue_captures)
            # Factory runs on the service owner loop before run_forever. Drain
            # restart leftovers even if no new hook event arrives.
            asyncio.get_event_loop().call_soon(dispatcher.start)
            return dispatcher

        super().__init__(
            descriptor_path=target,
            dispatcher_factory=create_dispatcher,
        )

    def start(self) -> dict[str, Any]:
        descriptor = super().start()
        expected = {
            "hook_protocol": HOOK_PROTOCOL,
            "adapter_host": self.adapter_host,
            "environment_sha256": self.environment_sha256,
            "capture_mode": self.capture_mode,
        }
        if self._external:
            if any(descriptor.get(key) != value for key, value in expected.items()):
                self.stop()
                raise RuntimeError("Existing runtime is not this hook worker")
            return descriptor
        try:
            descriptor.update(expected)
            _write_private_json(self.descriptor_path, descriptor)
            self._descriptor = descriptor
            return dict(descriptor)
        except BaseException:
            self.stop()
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an opt-in resident hook adapter")
    parser.add_argument("host", choices=("codex", "claude-code"))
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--queue-captures", action="store_true")
    args = parser.parse_args(argv)
    worker = HookWorker(
        args.host, descriptor_path=args.descriptor, cwd=args.cwd, queue_captures=args.queue_captures
    )
    try:
        worker.start()
        if not worker._external:
            while not worker.shutdown_requested.wait(0.25):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
