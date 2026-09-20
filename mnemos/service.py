"""Authenticated loopback runtime for the local Mnemos memory service."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import json
import logging
import os
import secrets
import signal
import socket
import sqlite3
import sys
import threading
import time
from ctypes import (
    POINTER,
    byref,
    c_int,
    c_char,
    c_long,
    c_ulong,
    c_ushort,
    c_void_p,
    c_wchar_p,
    cast,
    create_string_buffer,
    sizeof,
    string_at,
)
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .jobs import FactNotFoundError, JobNotFoundError
from .errors import (
    RelatedForgetPreviewStale,
    RelatedForgetSourceSelectionRequired,
)
from .observability import (
    ServiceLogBinding,
    install_service_file_logging,
    log_event,
    remove_service_file_logging,
)

SCHEMA_VERSION = 1
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_STARTUP_TIMEOUT = 5.0
DEFAULT_DISPATCH_TIMEOUT = 30.0


def default_runtime_directory() -> Path:
    configured = os.environ.get("MNEMOS_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return Path(base or Path.home() / "AppData" / "Local") / "Mnemos"
    return Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".local" / "run") / "mnemos"


def default_descriptor_path() -> Path:
    configured = os.environ.get("MNEMOS_RUNTIME_DESCRIPTOR")
    return (
        Path(configured).expanduser()
        if configured
        else default_runtime_directory() / "runtime.json"
    )


def _windows_dacl_matches(path: Path, desired: Any, advapi32: Any, kernel32: Any) -> bool:
    """Inspect the current ACL each time; never cache a private-path decision."""
    descriptor = c_void_p()
    actual = c_void_p()
    get_security = advapi32.GetNamedSecurityInfoW
    get_security.argtypes = [
        c_wchar_p,
        c_long,
        c_ulong,
        c_void_p,
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_void_p),
    ]
    get_security.restype = c_ulong
    try:
        result = get_security(str(path), 1, 4, None, None, byref(actual), None, byref(descriptor))
        if result != 0 or not actual or not descriptor:
            return False
        control = c_ushort()
        revision = c_ulong()
        get_control = advapi32.GetSecurityDescriptorControl
        get_control.argtypes = [c_void_p, POINTER(c_ushort), POINTER(c_ulong)]
        get_control.restype = c_int
        if not get_control(descriptor, byref(control), byref(revision)):
            return False
        # Both SE_DACL_PRESENT and SE_DACL_PROTECTED are required. An identical
        # but inheritable DACL is not private against later parent ACL changes.
        if control.value & 0x1004 != 0x1004:
            return False
        get_acl_info = advapi32.GetAclInformation
        get_acl_info.argtypes = [c_void_p, c_void_p, c_ulong, c_int]
        get_acl_info.restype = c_int
        snapshots = []
        for acl in (actual, desired):
            # ACL_SIZE_INFORMATION: AceCount, AclBytesInUse, AclBytesFree.
            info = (c_ulong * 3)()
            if not get_acl_info(acl, info, sizeof(info), 2):
                return False
            if not 8 <= info[1] <= 65535:
                return False
            snapshots.append(string_at(acl, info[1]))
        # Exact bytes include ACE types, SID, permissions, and inheritance.
        # Any representation difference conservatively takes the repair path.
        return snapshots[0] == snapshots[1]
    finally:
        if descriptor:
            kernel32.LocalFree(descriptor)


def _protect_private_path(path: Path) -> None:
    """Apply owner-only ACLs where POSIX mode bits are not sufficient."""
    if os.name != "nt":
        try:
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        except OSError as exc:
            raise PermissionError(f"could not protect private path: {path}") from exc
        return
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import ctypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        sid_size = c_ulong(0)
        token = c_void_p()
        kernel32.GetCurrentProcess.restype = c_void_p
        kernel32.CloseHandle.argtypes = [c_void_p]
        kernel32.CloseHandle.restype = c_int
        kernel32.LocalFree.argtypes = [c_void_p]
        kernel32.LocalFree.restype = c_void_p
        open_token = advapi32.OpenProcessToken
        open_token.argtypes = [c_void_p, c_ulong, POINTER(c_void_p)]
        open_token.restype = c_int
        if not open_token(kernel32.GetCurrentProcess(), 0x0008, byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            get_token_info = advapi32.GetTokenInformation
            get_token_info.argtypes = [c_void_p, c_long, c_void_p, c_ulong, POINTER(c_ulong)]
            get_token_info.restype = c_int
            get_token_info(token, 1, None, 0, byref(sid_size))
            token_user = create_string_buffer(sid_size.value)
            if not get_token_info(token, 1, token_user, sid_size, byref(sid_size)):
                raise ctypes.WinError(ctypes.get_last_error())
            sid_ptr = cast(token_user, POINTER(c_void_p))[0]
            sid_text = c_wchar_p()
            convert_sid = advapi32.ConvertSidToStringSidW
            convert_sid.argtypes = [c_void_p, POINTER(c_wchar_p)]
            convert_sid.restype = c_int
            if not convert_sid(sid_ptr, byref(sid_text)):
                raise ctypes.WinError(ctypes.get_last_error())
            sid_value = sid_text.value
            kernel32.LocalFree(cast(sid_text, c_void_p))
            if not sid_value:
                raise PermissionError("current Windows user SID is unavailable")
        finally:
            kernel32.CloseHandle(token)
        security_descriptor = c_void_p()
        convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [c_wchar_p, c_long, POINTER(c_void_p), POINTER(c_long)]
        convert.restype = c_int
        inheritance = "OICI" if path.is_dir() else ""
        if not convert(
            f"D:P(A;{inheritance};FA;;;{sid_value})", 1, byref(security_descriptor), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            dacl = c_void_p()
            present = c_int()
            defaulted = c_int()
            get_dacl = advapi32.GetSecurityDescriptorDacl
            get_dacl.argtypes = [c_void_p, POINTER(c_int), POINTER(c_void_p), POINTER(c_int)]
            get_dacl.restype = c_int
            if not get_dacl(security_descriptor, byref(present), byref(dacl), byref(defaulted)):
                raise ctypes.WinError(ctypes.get_last_error())
            if _windows_dacl_matches(path, dacl, advapi32, kernel32):
                return
            set_security = advapi32.SetNamedSecurityInfoW
            set_security.argtypes = [
                c_wchar_p,
                c_long,
                c_ulong,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
            ]
            set_security.restype = c_long
            # SE_FILE_OBJECT=1, DACL_SECURITY_INFORMATION=4,
            # PROTECTED_DACL_SECURITY_INFORMATION=0x80000000.
            result = set_security(str(path), 1, 0x80000004, None, None, dacl, None)
            if result != 0:
                raise ctypes.WinError(result)
        finally:
            kernel32.LocalFree.argtypes = [c_void_p]
            kernel32.LocalFree(security_descriptor)
    except (OSError, ImportError, AttributeError) as exc:
        raise PermissionError(f"could not protect private path: {path}") from exc


class _ProcessLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Protect both paths before opening the lock. Existing files can retain
        # explicit legacy grants, so protecting only the parent cannot establish
        # the exact owner-only file policy.
        try:
            _protect_private_path(self.path.parent)
            if not self.path.exists():
                try:
                    fd = os.open(
                        self.path,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    os.close(fd)
                except FileExistsError:
                    pass
            _protect_private_path(self.path)
            handle = self.path.open("a+b")
        except (OSError, PermissionError):
            return False
        try:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _write_private_json(
    path: Path, value: Mapping[str, Any], *, replace_timeout: float = 0.25
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _protect_private_path(path.parent)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor_fd)
        _protect_private_path(temporary)
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        deadline = time.monotonic() + min(0.25, max(0.0, replace_timeout))
        while True:
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                # Windows readers can briefly deny replacement even when the
                # destination ACL is correct. Retry only the atomic operation;
                # never remove the old descriptor or relax its permissions.
                remaining = deadline - time.monotonic()
                if (
                    sys.platform != "win32"
                    or getattr(exc, "winerror", None) not in {5, 32, 33}
                    or remaining <= 0
                ):
                    raise
                time.sleep(min(0.01, remaining))
        _protect_private_path(path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def read_runtime_descriptor(path: str | Path | None = None) -> dict[str, Any] | None:
    target = Path(path).expanduser() if path is not None else default_descriptor_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        port = int(raw["port"])
        pid = int(raw["pid"])
        token = str(raw["token"])
        host = str(raw["host"])
        schema = int(raw["schema_version"])
    except (KeyError, TypeError, ValueError):
        return None
    if schema != SCHEMA_VERSION or not (1 <= port <= 65535) or pid <= 0 or not token:
        return None
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None
    return dict(raw)


class _LoopbackHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: "MnemosService") -> None:
        self.service = service
        super().__init__(address, _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    server: _LoopbackHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Request logs must never accidentally include bearer tokens or payloads.
        return

    def _write_json(self, status: int, body: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(body), sort_keys=True).encode("utf-8")
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        # An unread request body can make Windows reset the connection and
        # discard this response. Finish sending, then drain a bounded amount.
        if self.command == "POST" and not getattr(self, "_body_read", False):
            try:
                remaining = min(
                    max(int(self.headers.get("Content-Length", "0")), 0), DEFAULT_MAX_BODY_BYTES
                )
                self.connection.shutdown(socket.SHUT_WR)
                deadline = time.monotonic() + 0.1
                while remaining > 0 and time.monotonic() < deadline:
                    self.connection.settimeout(max(0.001, deadline - time.monotonic()))
                    data = self.rfile.read1(min(remaining, 65536))
                    if not data:
                        break
                    remaining -= len(data)
            except (ValueError, OSError):
                pass

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        expected_hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc in expected_hosts
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.service.token}"
        return hmac.compare_digest(header.encode("utf-8"), expected.encode("utf-8"))

    def _guard(self) -> bool:
        if not self._origin_allowed():
            self._write_json(403, {"schema_version": SCHEMA_VERSION, "error": "origin_not_allowed"})
            return False
        if not self._authorized():
            self._write_json(401, {"schema_version": SCHEMA_VERSION, "error": "unauthorized"})
            return False
        return True

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._write_json(403, {"schema_version": SCHEMA_VERSION, "error": "origin_not_allowed"})
            return
        self.close_connection = True
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/health", "/v1/health"} or not self._guard():
            if self.path not in {"/health", "/v1/health"}:
                self._write_json(404, {"schema_version": SCHEMA_VERSION, "error": "not_found"})
            return
        self._write_json(200, self.server.service.health())

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/dispatch", "/v1/dispatch"}:
            self._write_json(404, {"schema_version": SCHEMA_VERSION, "error": "not_found"})
            return
        if not self._guard():
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._write_json(
                411, {"schema_version": SCHEMA_VERSION, "error": "content_length_required"}
            )
            return
        if length > self.server.service.max_body_bytes:
            self._write_json(413, {"schema_version": SCHEMA_VERSION, "error": "body_too_large"})
            return
        try:
            self.connection.settimeout(5.0)
            raw_body = self.rfile.read(length)
            self._body_read = True
            body = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            self._write_json(400, {"schema_version": SCHEMA_VERSION, "error": "invalid_json"})
            return
        if not isinstance(body, dict):
            self._write_json(
                400, {"schema_version": SCHEMA_VERSION, "error": "request_object_required"}
            )
            return
        schema_version = body.get("schema_version", SCHEMA_VERSION)
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SCHEMA_VERSION
        ):
            self._write_json(400, {"schema_version": SCHEMA_VERSION, "error": "unsupported_schema"})
            return
        operation = body.get("operation", body.get("op"))
        payload = body.get("payload", {})
        if not isinstance(operation, str) or not operation.strip() or not isinstance(payload, dict):
            self._write_json(400, {"schema_version": SCHEMA_VERSION, "error": "invalid_dispatch"})
            return
        try:
            result = self.server.service.submit(operation.strip(), payload)
        except TimeoutError:
            self._write_json(504, {"schema_version": SCHEMA_VERSION, "error": "dispatch_timeout"})
            return
        except JobNotFoundError:
            self._write_json(404, {"schema_version": SCHEMA_VERSION, "error": "job_not_found"})
            return
        except FactNotFoundError:
            self._write_json(404, {"schema_version": SCHEMA_VERSION, "error": "fact_not_found"})
            return
        except RelatedForgetSourceSelectionRequired:
            self._write_json(
                409,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "rejected",
                    "committed": False,
                    "error": RelatedForgetSourceSelectionRequired.code,
                    "message": RelatedForgetSourceSelectionRequired.safe_message,
                },
            )
            return
        except RelatedForgetPreviewStale:
            self._write_json(
                409,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "rejected",
                    "committed": False,
                    "error": RelatedForgetPreviewStale.code,
                    "message": RelatedForgetPreviewStale.safe_message,
                },
            )
            return
        except Exception:
            self._write_json(500, {"schema_version": SCHEMA_VERSION, "error": "dispatch_failed"})
            return
        self._write_json(200, result)


class MnemosService:
    """Own one event loop and expose an authenticated loopback dispatcher."""

    def __init__(
        self,
        dispatcher: Any | None = None,
        *,
        dispatcher_factory: Callable[[], Any] | None = None,
        descriptor_path: str | Path | None = None,
        lock_path: str | Path | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        dispatch_timeout: float = DEFAULT_DISPATCH_TIMEOUT,
        shutdown_timeout: float = 30.0,
        service_log_path: str | Path | None = None,
    ) -> None:
        if dispatcher is not None and dispatcher_factory is not None:
            raise ValueError("pass dispatcher or dispatcher_factory, not both")
        if (
            max_body_bytes <= 0
            or startup_timeout <= 0
            or dispatch_timeout <= 0
            or shutdown_timeout <= 0
        ):
            raise ValueError("service limits must be positive")
        self.descriptor_path = (
            Path(descriptor_path).expanduser() if descriptor_path else default_descriptor_path()
        )
        self.lock_path = (
            Path(lock_path).expanduser() if lock_path else self.descriptor_path.with_suffix(".lock")
        )
        self.max_body_bytes = max_body_bytes
        self.startup_timeout = startup_timeout
        self.dispatch_timeout = dispatch_timeout
        self.shutdown_timeout = shutdown_timeout
        self.service_log_path = (
            Path(service_log_path).expanduser()
            if service_log_path is not None
            else self.descriptor_path.with_name("service.log")
        )
        self._service_logging_enabled = service_log_path is not None or (
            dispatcher is None and dispatcher_factory is None
        )
        self._service_log_binding: ServiceLogBinding | None = None
        self._maintenance_target: Any | None = None
        self._previous_maintenance_observer: Any | None = None
        self._dispatcher = dispatcher
        self._dispatcher_factory = dispatcher_factory
        self._owner_lock = _ProcessLock(self.lock_path)
        self._owns_process = False
        self._external = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owner_thread: threading.Thread | None = None
        self._owner_stopping = False
        self._owner_ready = threading.Event()
        self.shutdown_requested = threading.Event()
        self._owner_error: BaseException | None = None
        self._maintenance_task: asyncio.Task[Any] | None = None
        self._httpd: _LoopbackHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._descriptor: dict[str, Any] | None = None
        self.token = ""

    def _enable_service_logging(self) -> None:
        if self._service_logging_enabled and self._service_log_binding is None:
            self._service_log_binding = install_service_file_logging(
                self.service_log_path,
                protect_path=_protect_private_path,
            )

    def _disable_service_logging(self) -> None:
        binding = self._service_log_binding
        self._service_log_binding = None
        remove_service_file_logging(binding)

    def _log_sqlite_failure(self, operation: str, error: BaseException) -> None:
        if self._service_log_binding is None:
            return
        log_event(
            "mnemos.provider_failure",
            level=logging.ERROR,
            provider="sqlite" if isinstance(error, sqlite3.Error) else "runtime",
            operation=operation,
            error_type=type(error).__name__,
        )

    def _log_maintenance_failure(self, error: BaseException) -> None:
        try:
            self._log_sqlite_failure("maintenance", error)
        finally:
            if callable(self._previous_maintenance_observer):
                self._previous_maintenance_observer(error)

    def _set_maintenance_observer(self, dispatcher: Any) -> None:
        maintenance = getattr(dispatcher, "maintenance", None)
        if maintenance is not None and self._service_log_binding is not None:
            self._maintenance_target = maintenance
            self._previous_maintenance_observer = getattr(maintenance, "on_error", None)
            setattr(maintenance, "on_error", self._log_maintenance_failure)

    def _restore_maintenance_observer(self) -> None:
        maintenance = self._maintenance_target
        if maintenance is not None:
            setattr(maintenance, "on_error", self._previous_maintenance_observer)
        self._maintenance_target = None
        self._previous_maintenance_observer = None

    @staticmethod
    def _component_label(value: Any, fallback: str) -> str:
        label = type(value).__name__ if value is not None else ""
        return label if label.isidentifier() else fallback

    def _log_started(self) -> None:
        if self._service_log_binding is None:
            return
        dispatcher = self._dispatcher
        engine = getattr(dispatcher, "engine", None)
        store = getattr(engine, "store", None)
        provider = getattr(engine, "cognition_provider", None)
        if provider is None:
            provider = getattr(engine, "llm", None)
        log_event(
            "mnemos.startup",
            status="running",
            backend=self._component_label(store, "unknown"),
            provider=self._component_label(provider, "unavailable"),
        )

    @property
    def owner_thread_id(self) -> int | None:
        return self._owner_thread.ident if self._owner_thread else None

    @property
    def descriptor(self) -> dict[str, Any] | None:
        return dict(self._descriptor) if self._descriptor else None

    def _make_default_dispatcher(self) -> Any:
        from .memory_service import LocalMemoryService
        from .runtime import build_configured_engine
        from .scopes import ScopeRegistry, default_registry_path

        engine = build_configured_engine(
            default_store_type="sqlite", cwd=default_runtime_directory()
        )
        registry_path = default_registry_path()
        registry = (
            ScopeRegistry.load(registry_path)
            if registry_path and Path(registry_path).exists()
            else ScopeRegistry()
        )
        return LocalMemoryService(engine, registry)

    def _owner_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            created_default = self._dispatcher is None and self._dispatcher_factory is None
            if self._dispatcher is None:
                factory = self._dispatcher_factory or self._make_default_dispatcher
                self._dispatcher = factory()
            self._set_maintenance_observer(self._dispatcher)
            if created_default:
                maintenance = getattr(self._dispatcher, "maintenance", None)
                run_maintenance = getattr(maintenance, "run", None)
                if callable(run_maintenance):
                    self._maintenance_task = loop.create_task(run_maintenance())
            self._owner_ready.set()
            loop.run_forever()
        except BaseException as exc:  # surface initialization failures to start()
            self._log_sqlite_failure("startup", exc)
            self._owner_error = exc
            self._owner_ready.set()
        finally:
            maintenance = getattr(self._dispatcher, "maintenance", None)
            stop_maintenance = getattr(maintenance, "stop", None)
            if callable(stop_maintenance):
                stop_maintenance()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            dispatcher = self._dispatcher
            closer = getattr(dispatcher, "close", None)
            try:
                if callable(closer):
                    result = closer()
                    if inspect.isawaitable(result):
                        loop.run_until_complete(result)
                else:
                    engine = getattr(dispatcher, "engine", None)
                    store = getattr(engine, "store", None)
                    store_close = getattr(store, "close", None)
                    if callable(store_close):
                        store_close()
            except Exception:
                pass
            loop.close()
            self._loop = None

    @staticmethod
    def _existing_owner_ready(descriptor: dict[str, Any], timeout: float) -> bool:
        """Verify the lock owner's transport without submitting memory work."""
        connection = HTTPConnection(
            str(descriptor["host"]), int(descriptor["port"]), timeout=timeout
        )
        try:
            connection.request(
                "GET", "/v1/health", headers={"Authorization": f"Bearer {descriptor['token']}"}
            )
            response = connection.getresponse()
            if response.status in {401, 403}:
                raise RuntimeError("existing Mnemos service rejected the readiness check")
            if response.status != 200:
                raise RuntimeError("existing Mnemos service returned invalid readiness")
            # Read only a small health object, never an unbounded response body.
            body = response.read(4097)
            if len(body) > 4096:
                raise RuntimeError("existing Mnemos service returned oversized readiness")
            try:
                value = json.loads(body)
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError("existing Mnemos service returned invalid readiness") from exc
            if (
                not isinstance(value, dict)
                or type(value.get("schema_version")) is not int
                or value["schema_version"] != SCHEMA_VERSION
                or value.get("status") != "ok"
            ):
                raise RuntimeError("existing Mnemos service returned invalid readiness")
            return True
        except (OSError, HTTPException):
            return False
        finally:
            connection.close()

    def start(self) -> dict[str, Any]:
        if self._descriptor is not None:
            return dict(self._descriptor)
        if not self._owner_lock.acquire():
            deadline = time.monotonic() + self.startup_timeout
            while True:
                existing = read_runtime_descriptor(self.descriptor_path)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "another Mnemos service owns the lock but is not ready before startup timeout"
                    )
                if existing is not None and self._existing_owner_ready(
                    existing, min(0.25, remaining)
                ):
                    break
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            self._descriptor = existing
            self.token = str(existing["token"])
            self._external = True
            return dict(existing)
        self._owns_process = True
        startup_deadline = time.monotonic() + self.startup_timeout
        self.token = secrets.token_urlsafe(32)
        self._owner_error = None
        self._owner_ready.clear()
        self._owner_thread = threading.Thread(
            target=self._owner_main, name="mnemos-owner", daemon=True
        )
        try:
            self._enable_service_logging()
            self._owner_thread.start()
            if not self._owner_ready.wait(self.startup_timeout):
                raise TimeoutError("Mnemos service dispatcher did not initialize")
            if self._owner_error is not None:
                raise RuntimeError(
                    "Mnemos service dispatcher failed to initialize"
                ) from self._owner_error
            self._httpd = _LoopbackHTTPServer(("127.0.0.1", 0), self)
            self._http_thread = threading.Thread(
                target=self._httpd.serve_forever,
                name="mnemos-http",
                daemon=True,
            )
            self._http_thread.start()
            now = datetime.now(timezone.utc).isoformat()
            self._descriptor = {
                "schema_version": SCHEMA_VERSION,
                "host": "127.0.0.1",
                "port": self._httpd.server_port,
                "pid": os.getpid(),
                "token": self.token,
                "started_at": now,
            }
            _write_private_json(
                self.descriptor_path,
                self._descriptor,
                replace_timeout=max(0.0, startup_deadline - time.monotonic()),
            )
            self._log_started()
            return dict(self._descriptor)
        except BaseException as exc:
            if self._owner_error is None:
                self._log_sqlite_failure("startup", exc)
            self.stop()
            raise

    async def _call_dispatcher(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation == "shutdown":
            self.shutdown_requested.set()
            return {"schema_version": SCHEMA_VERSION, "status": "stopping", "committed": False}
        dispatcher = self._dispatcher
        if dispatcher is None:
            raise RuntimeError("dispatcher is not initialized")
        method = getattr(dispatcher, "dispatch", None)
        result = method(operation, payload) if callable(method) else dispatcher(operation, payload)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError("dispatcher must return a JSON object")
        response = dict(result)
        response.setdefault("schema_version", SCHEMA_VERSION)
        return response

    def submit(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        loop = self._loop
        if loop is None or self._external:
            raise RuntimeError("service is not the owning runtime")
        future = asyncio.run_coroutine_threadsafe(self._call_dispatcher(operation, payload), loop)
        try:
            return future.result(self.dispatch_timeout)
        except BaseException as exc:
            self._log_sqlite_failure("dispatch", exc)
            raise

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "committed": False,
            "owner_thread_id": self.owner_thread_id,
        }
        return result

    def stop(self) -> None:
        if self._external:
            self._descriptor = None
            self._external = False
            return
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=self.startup_timeout)
            self._http_thread = None
        loop = self._loop
        if loop is not None and not loop.is_closed() and not self._owner_stopping:
            self._owner_stopping = True
            loop.call_soon_threadsafe(loop.stop)
        if self._owner_thread is not None:
            if self._owner_thread.is_alive():
                self._owner_thread.join(timeout=self.shutdown_timeout)
            if self._owner_thread.is_alive():
                # Closing SQLite can outlast startup, particularly while a
                # durable write is finishing. Keep ownership and the thread
                # handle so a later stop can finish cleanup safely.
                raise TimeoutError("Mnemos owner has not finished shutting down")
            self._owner_thread = None
        if self._owns_process:
            try:
                current = read_runtime_descriptor(self.descriptor_path)
                if (
                    current is not None
                    and current.get("pid") == os.getpid()
                    and current.get("token") == self.token
                ):
                    self.descriptor_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._restore_maintenance_observer()
        self._disable_service_logging()
        self._owner_lock.release()
        self._owns_process = False
        self._owner_stopping = False
        self._descriptor = None
        self.token = ""

    close = stop


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Mnemos loopback service")
    parser.add_argument("--descriptor", type=Path, default=None)
    args = parser.parse_args()
    service = MnemosService(descriptor_path=args.descriptor)
    service.start()
    if service._external:
        # A simultaneous client already started the owner. This launcher has
        # no work and must not linger as another background process.
        return
    stopped = service.shutdown_requested

    def request_stop(*_: Any) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopped.wait(0.5):
            pass
    finally:
        service.stop()


if __name__ == "__main__":
    _main()


MemoryService = MnemosService
LocalMemoryServer = MnemosService
