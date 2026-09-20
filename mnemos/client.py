"""Client and durable local outbox for the Mnemos loopback service."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Iterator, cast
from uuid import uuid4

import httpx

from .memory_safety import MemoryWriteFirewall
from .errors import (
    RELATED_FORGET_PREVIEW_STALE,
    RELATED_FORGET_PREVIEW_STALE_MESSAGE,
    RELATED_FORGET_SELECTION_INVALID,
    RELATED_FORGET_SELECTION_INVALID_MESSAGE,
    RELATED_FORGET_SOURCE_SELECTION_MESSAGE,
    RELATED_FORGET_SOURCE_SELECTION_REQUIRED,
)
from .service import (
    SCHEMA_VERSION,
    _protect_private_path,
    _ProcessLock,
    default_descriptor_path,
    read_runtime_descriptor,
)


class Outbox:
    """Small atomic JSON outbox for sanitized host capture events."""

    _path_locks: dict[str, Any] = {}
    _path_locks_guard = threading.Lock()

    def __init__(
        self,
        path: str | Path,
        *,
        firewall: MemoryWriteFirewall | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.firewall = firewall or MemoryWriteFirewall()
        with self._path_locks_guard:
            self._lock = self._path_locks.setdefault(str(self.path.resolve()), threading.RLock())

    @staticmethod
    def _copy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(dict(payload), ensure_ascii=False)))

    def _sanitize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = self._copy_payload(payload)

        def visit(value: Any, key: str | None = None) -> Any:
            if isinstance(value, dict):
                return {str(k): visit(v, str(k)) for k, v in value.items()}
            if isinstance(value, list):
                return [visit(item, key) for item in value]
            if isinstance(value, str):
                decision = self.firewall.apply(value)
                if not decision.allowed:
                    raise ValueError(decision.reason)
                if decision.content != value and key not in {"content", "source_ref"}:
                    raise ValueError("Source metadata violates write policy")
                return decision.content
            return value

        sanitized = visit(result)
        if not isinstance(sanitized, dict):
            raise TypeError("outbox payload must be an object")
        return sanitized

    def sanitize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a safety-filtered copy suitable for transport or disk."""
        return self._sanitize(payload)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock:
            lock = _ProcessLock(self.path.with_suffix(self.path.suffix + ".lock"))
            deadline = time.monotonic() + 0.5
            while not lock.acquire():
                if time.monotonic() >= deadline:
                    raise ValueError("Outbox is busy; capture was not acknowledged")
                time.sleep(0.005)
            try:
                yield
            finally:
                lock.release()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            # A pre-existing outbox can retain its inherited Windows ACL after
            # its parent directory is made private.  Repair the file ACL before
            # reading it so restart and upgrade paths remain usable.
            _protect_private_path(self.path)
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid outbox file: {self.path}") from exc
        if not isinstance(raw, list):
            raise ValueError(f"invalid outbox file: {self.path}")
        return [dict(item) for item in raw if isinstance(item, dict) and item.get("event_id")]

    def _write(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _protect_private_path(self.path.parent)
        fd, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            os.close(fd)
            _protect_private_path(Path(name))
            with Path(name).open("w", encoding="utf-8") as handle:
                json.dump(entries, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            _protect_private_path(self.path)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def append(
        self,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
        operation: str = "capture",
    ) -> str:
        sanitized = self._sanitize(payload)
        stable_id = str(event_id or sanitized.get("event_id") or uuid4())
        sanitized["event_id"] = stable_id
        context = sanitized.get("context")
        if isinstance(context, dict):
            stable_id = hashlib.sha256(
                json.dumps(
                    [
                        context.get("user_profile_id"),
                        context.get("host"),
                        context.get("session_id"),
                        stable_id,
                    ]
                ).encode("utf-8")
            ).hexdigest()
        entry = {"event_id": stable_id, "operation": operation, "payload": sanitized}
        with self._exclusive():
            entries = self._read()
            existing = next((item for item in entries if item.get("event_id") == stable_id), None)
            if existing is not None and existing != entry:
                raise ValueError("Outbox event identity already has a different payload")
            if existing is None:
                entries.append(entry)
                self._write(entries)
        return stable_id

    def pending(self) -> list[dict[str, Any]]:
        with self._exclusive():
            return [dict(item) for item in self._read()]

    def acknowledge(self, event_id: str) -> bool:
        with self._exclusive():
            entries = self._read()
            remaining = [item for item in entries if item.get("event_id") != event_id]
            if len(remaining) == len(entries):
                return False
            if remaining:
                self._write(remaining)
            else:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            return True

    def __len__(self) -> int:
        return len(self.pending())


class ServiceClientError(RuntimeError):
    """An authenticated service request failed at the transport boundary."""

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.unavailable = unavailable


class MnemosClient:
    """HTTP client with one bounded service autostart and honest degraded writes."""

    def __init__(
        self,
        *,
        descriptor_path: str | Path | None = None,
        outbox: Outbox | None = None,
        outbox_path: str | Path | None = None,
        auto_start: bool = True,
        startup_timeout: float = 5.0,
        request_timeout: float = 10.0,
        python_executable: str | None = None,
    ) -> None:
        self.descriptor_path = (
            Path(descriptor_path).expanduser() if descriptor_path else default_descriptor_path()
        )
        # An empty Outbox is false-valued because it defines __len__. Its
        # selected path and firewall must not depend on pending event count.
        self.outbox = (
            outbox
            if outbox is not None
            else (
                Outbox(outbox_path)
                if outbox_path is not None
                else Outbox(self.descriptor_path.with_name("outbox.json"))
            )
        )
        if startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("client timeouts must be positive")
        self.auto_start = auto_start
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.python_executable = python_executable or sys.executable
        self._autostart_attempted = False
        self._outbox_drained = False
        self._child: subprocess.Popen[bytes] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None

    def _descriptor(self) -> dict[str, Any] | None:
        return read_runtime_descriptor(self.descriptor_path)

    def _spawn_once(self) -> None:
        if self._autostart_attempted or not self.auto_start:
            return
        self._autostart_attempted = True
        command = [
            self.python_executable,
            "-m",
            "mnemos.service",
            "--descriptor",
            str(self.descriptor_path),
        ]
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        try:
            self._child = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            raise ServiceClientError("Mnemos service could not be started") from exc

    @staticmethod
    def _same_owner(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return all(
            left.get(key) == right.get(key)
            for key in ("schema_version", "host", "port", "pid", "token")
        )

    @staticmethod
    def _loopback_http_client(timeout: float) -> httpx.AsyncClient:
        # These routes use plaintext HTTP on loopback. Loading a CA bundle
        # cannot protect them, but can exhaust a short readiness deadline.
        # Keep TLS verification enabled with no trust roots so accidental
        # future HTTPS use fails closed rather than accepting certificates.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return httpx.AsyncClient(
            timeout=timeout, verify=context, trust_env=False, follow_redirects=False
        )

    async def _owner_reachable(
        self, descriptor: dict[str, Any], *, timeout: float | None = None
    ) -> bool:
        """Probe the authenticated transport without dispatching or replaying writes."""
        host = str(descriptor["host"])
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ServiceClientError("runtime descriptor is not loopback")
        authority = f"[{host}]" if host == "::1" else host
        url = f"http://{authority}:{int(descriptor['port'])}/v1/health"
        budget = min(0.25, self.startup_timeout, self.request_timeout)
        if timeout is not None:
            budget = min(budget, timeout)
        if budget <= 0:
            return False

        async def probe() -> httpx.Response:
            async with self._loopback_http_client(budget) as client:
                return await client.get(
                    url, headers={"Authorization": f"Bearer {descriptor['token']}"}
                )

        try:
            response = await asyncio.wait_for(probe(), timeout=budget)
        except (httpx.TransportError, OSError, asyncio.TimeoutError):
            return False
        if response.status_code in {401, 403}:
            raise ServiceClientError(
                f"Mnemos service rejected the readiness check ({response.status_code})"
            )
        if response.status_code != 200:
            raise ServiceClientError("Mnemos service returned an invalid readiness response")
        try:
            value = response.json()
        except ValueError as exc:
            raise ServiceClientError("Mnemos service returned invalid readiness JSON") from exc
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION
            or value.get("status") != "ok"
        ):
            raise ServiceClientError("Mnemos service returned an invalid readiness response")
        return True

    async def _wait_for_descriptor(
        self, *, failed_descriptor: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        while True:
            descriptor = self._descriptor()
            if descriptor is not None and (
                failed_descriptor is None or not self._same_owner(descriptor, failed_descriptor)
            ):
                return descriptor
            exit_code = self._child.poll() if self._child is not None else None
            if exit_code is not None:
                # A competing owner can publish its descriptor while this
                # launcher exits. Prefer that owner before reporting failure.
                descriptor = self._descriptor()
                if descriptor is not None:
                    if failed_descriptor is None or not self._same_owner(
                        descriptor, failed_descriptor
                    ):
                        return descriptor
                    # A live owner may recover after the first failed probe
                    # while the competing launcher exits. Do not mistake its
                    # unchanged descriptor for proof of readiness or failure.
                    if await self._owner_reachable(descriptor, timeout=deadline - loop.time()):
                        return descriptor
                if descriptor is None or loop.time() >= deadline:
                    raise ServiceClientError(
                        f"Mnemos service exited before startup (exit code {exit_code})",
                        unavailable=True,
                    )
                # An unchanged owner can still recover. Allow another probe
                # within the original startup budget without spawning again.
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ServiceClientError("Mnemos service startup timed out", unavailable=True)
            await asyncio.sleep(min(0.05, remaining))

    async def _dispatch_http(
        self, operation: str, payload: dict[str, Any], descriptor: dict[str, Any]
    ) -> dict[str, Any]:
        host = str(descriptor["host"])
        port = int(descriptor["port"])
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ServiceClientError("runtime descriptor is not loopback")
        url = f"http://{host}:{port}/v1/dispatch"
        headers = {
            "Authorization": f"Bearer {descriptor['token']}",
            "Content-Type": "application/json",
        }
        body = {"schema_version": SCHEMA_VERSION, "operation": operation, "payload": payload}
        try:
            loop = asyncio.get_running_loop()
            if (
                self._http_client is None
                or self._http_client.is_closed
                or self._http_loop is not loop
            ):
                # Keep one connection pool per owning event loop.  Host
                # adapters may invoke one client from a short-lived loop, so
                # never reuse an AsyncClient across loop boundaries.
                if self._http_client is not None and not self._http_client.is_closed:
                    await self._http_client.aclose()
                self._http_client = self._loopback_http_client(self.request_timeout)
                self._http_loop = loop
            response = await self._http_client.post(url, headers=headers, json=body)
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            OSError,
        ) as exc:
            # A committed capture can lose its response. Keep its stable event
            # identity on the existing retry/outbox path until acknowledged.
            raise ServiceClientError("Mnemos service is unavailable", unavailable=True) from exc
        if response.status_code in {401, 403}:
            raise ServiceClientError(
                f"Mnemos service rejected the request ({response.status_code})"
            )
        if response.status_code == 409 and operation.strip() == "forget_related":
            try:
                result = response.json()
            except ValueError as exc:
                raise ServiceClientError("Mnemos service returned HTTP 409") from exc
            expected_keys = {"schema_version", "status", "committed", "error", "message"}
            if (
                isinstance(result, dict)
                and set(result) == expected_keys
                and type(result.get("schema_version")) is int
                and result.get("schema_version") == SCHEMA_VERSION
                and result.get("status") == "rejected"
                and result.get("committed") is False
                and result.get("error") == RELATED_FORGET_SOURCE_SELECTION_REQUIRED
                and result.get("message") == RELATED_FORGET_SOURCE_SELECTION_MESSAGE
            ):
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "rejected",
                    "committed": False,
                    "error": RELATED_FORGET_SOURCE_SELECTION_REQUIRED,
                    "message": RELATED_FORGET_SOURCE_SELECTION_MESSAGE,
                }
            if (
                isinstance(result, dict)
                and set(result) == expected_keys
                and type(result.get("schema_version")) is int
                and result.get("schema_version") == SCHEMA_VERSION
                and result.get("status") == "rejected"
                and result.get("committed") is False
                and result.get("error") == RELATED_FORGET_PREVIEW_STALE
                and result.get("message") == RELATED_FORGET_PREVIEW_STALE_MESSAGE
            ):
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "rejected",
                    "committed": False,
                    "error": RELATED_FORGET_PREVIEW_STALE,
                    "message": RELATED_FORGET_PREVIEW_STALE_MESSAGE,
                }
            if (
                isinstance(result, dict)
                and set(result) == expected_keys
                and type(result.get("schema_version")) is int
                and result.get("schema_version") == SCHEMA_VERSION
                and result.get("status") == "rejected"
                and result.get("committed") is False
                and result.get("error") == RELATED_FORGET_SELECTION_INVALID
                and result.get("message") == RELATED_FORGET_SELECTION_INVALID_MESSAGE
            ):
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "rejected",
                    "committed": False,
                    "error": RELATED_FORGET_SELECTION_INVALID,
                    "message": RELATED_FORGET_SELECTION_INVALID_MESSAGE,
                }
            raise ServiceClientError("Mnemos service returned HTTP 409")
        if response.status_code >= 400:
            raise ServiceClientError(f"Mnemos service returned HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            raise ServiceClientError("Mnemos service returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ServiceClientError("Mnemos service returned a non-object response")
        return dict(result)

    @staticmethod
    def _committed(result: Mapping[str, Any]) -> bool:
        return result.get("committed") is True or result.get("status") == "committed"

    async def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        queue_on_failure: bool | None = None,
        _draining: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        data = dict(payload or {})
        is_capture = operation.strip() in {"capture", "mnemos_capture"}
        if queue_on_failure is None:
            queue_on_failure = is_capture
        pending_event_id: str | None = None
        if is_capture:
            try:
                data = self.outbox.sanitize(data)
                if queue_on_failure:
                    # A host can terminate this process during startup or HTTP
                    # dispatch. Persist the sanitized event before either wait,
                    # keeping its identity unchanged when replayed after restart.
                    data["event_id"] = data.get("event_id") or str(uuid4())
                    pending_event_id = self.outbox.append(data, operation=operation.strip())
                    self._outbox_drained = False
            except (TypeError, ValueError) as safety_error:
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "committed": False,
                    "error": str(safety_error),
                }
        try:
            descriptor = self._descriptor()
            if descriptor is None:
                self._spawn_once()
                descriptor = (
                    await self._wait_for_descriptor() if self._autostart_attempted else None
                )
            if descriptor is None:
                raise ServiceClientError("Mnemos service descriptor is unavailable")
            try:
                result = await self._dispatch_http(operation.strip(), data, descriptor)
            except ServiceClientError as exc:
                if not exc.unavailable or self._autostart_attempted or not self.auto_start:
                    raise
                retry_descriptor: dict[str, Any] | None
                if await self._owner_reachable(descriptor):
                    retry_descriptor = descriptor
                else:
                    self._spawn_once()
                    retry_descriptor = await self._wait_for_descriptor(failed_descriptor=descriptor)
                if retry_descriptor is None:
                    raise
                result = await self._dispatch_http(operation.strip(), data, retry_descriptor)
        except ServiceClientError as exc:
            if queue_on_failure and is_capture:
                try:
                    event_id = pending_event_id or self.outbox.append(
                        data, operation=operation.strip()
                    )
                    self._outbox_drained = False
                except (TypeError, ValueError) as safety_error:
                    return {
                        "schema_version": SCHEMA_VERSION,
                        "status": "blocked",
                        "committed": False,
                        "error": str(safety_error),
                    }
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "queued",
                    "committed": False,
                    "degraded": True,
                    "event_id": event_id,
                    "error": str(exc),
                }
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "degraded",
                "committed": False,
                "degraded": True,
                "error": str(exc),
            }
        if (
            operation.strip() == "forget_related"
            and result.get("status") == "rejected"
            and result.get("error")
            in {RELATED_FORGET_SOURCE_SELECTION_REQUIRED, RELATED_FORGET_PREVIEW_STALE}
        ):
            return result
        if pending_event_id is not None:
            if self._committed(result) or result.get("status") in {"rejected", "blocked"}:
                self.outbox.acknowledge(pending_event_id)
            else:
                # An ambiguous reply is not an acknowledgement. Retain the
                # durable event without immediately resending the same request.
                return result
        if operation.strip() != "shutdown" and not _draining and not self._outbox_drained:
            replay = await self.replay_outbox()
            self._outbox_drained = replay["pending"] == 0
        return result

    async def replay_outbox(self) -> dict[str, Any]:
        acknowledged = 0
        failed = 0
        for entry in self.outbox.pending():
            payload = entry.get("payload")
            operation = entry.get("operation", "capture")
            event_id = str(entry.get("event_id"))
            if not isinstance(payload, dict) or not isinstance(operation, str):
                failed += 1
                continue
            try:
                result = await self.request(
                    operation, payload, queue_on_failure=False, _draining=True
                )
            except Exception:
                failed += 1
                continue
            if self._committed(result):
                if self.outbox.acknowledge(event_id):
                    acknowledged += 1
            else:
                failed += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "acknowledged": acknowledged,
            "failed": failed,
            "pending": len(self.outbox),
        }

    async def capture(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.request("capture", payload)

    async def retrieve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.request("retrieve", payload, queue_on_failure=False)

    async def close(self) -> None:
        # A host session may have started the shared runtime, but it does not
        # own its lifetime. Other hosts and background maintenance still use it.
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = None
        self._http_loop = None
        self._child = None


Client = MnemosClient
LocalOutbox = Outbox
