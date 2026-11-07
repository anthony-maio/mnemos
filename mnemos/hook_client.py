"""Opt-in command client for a resident hook adapter, with no runtime imports.

Only a refusal before adapter dispatch permits the ordinary one-shot fallback.
After sending a request, a lost or malformed reply is an unknown outcome.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from uuid import uuid4

HOOK_PROTOCOL = 1
MAX_FRAME_BYTES = 1_048_576
WORKER_DESCRIPTOR_ENV = "MNEMOS_HOOK_WORKER_DESCRIPTOR"


class HookOutcomeUnknown(RuntimeError):
    """A worker may have captured the event; do not blindly dispatch it again."""


def _post_http(port: int, token: str, wire: bytes, timeout: float) -> bytes | None:
    from http.client import HTTPConnection

    connection = HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        try:
            connection.connect()
        except OSError:
            return None
        connection.request(
            "POST",
            "/v1/dispatch",
            body=wire,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read(MAX_FRAME_BYTES + 1)
        if response.status != 200 or len(raw) > MAX_FRAME_BYTES:
            raise ValueError("Invalid worker response")
        return raw
    finally:
        connection.close()


def _post(port: int, token: str, wire: bytes, timeout: float) -> bytes | None:
    if sys.platform == "win32":
        from ._winhttp_hook import WinHTTPUnavailable, post

        try:
            return post(port, token, wire, timeout=timeout, limit=MAX_FRAME_BYTES)
        except WinHTTPUnavailable:
            pass  # Only preparation failure permits another transport.
    return _post_http(port, token, wire, timeout)


def environment_fingerprint(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    selected = {key: value for key, value in source.items() if key.upper() != WORKER_DESCRIPTOR_ENV}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode("utf-8")).hexdigest()


def request_hook(
    host: str,
    event: str,
    payload: dict[str, Any],
    *,
    cwd: str,
    descriptor_path: str | Path,
    timeout: float = 30,
) -> dict[str, Any] | None:
    """Return a bound worker reply, or None only when fallback is safe."""
    if host not in {"codex", "claude-code"} or timeout <= 0:
        raise ValueError("Invalid hook client settings")
    try:
        descriptor = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(descriptor, dict)
        or type(descriptor.get("hook_protocol")) is not int
        or descriptor["hook_protocol"] != HOOK_PROTOCOL
        or descriptor.get("adapter_host") != host
        or descriptor.get("host") != "127.0.0.1"
        or type(descriptor.get("port")) is not int
        or not 1 <= descriptor["port"] <= 65535
        or not isinstance(descriptor.get("token"), str)
        or not descriptor["token"]
        or descriptor.get("environment_sha256") != environment_fingerprint()
    ):
        return None
    identifier = str(uuid4())
    envelope = {
        "hook_protocol": HOOK_PROTOCOL,
        "request_id": identifier,
        "host": host,
        "event": event,
        "payload": payload,
        "cwd": cwd,
        "environment_sha256": environment_fingerprint(),
    }
    wire = json.dumps({"schema_version": 1, "operation": "hook", "payload": envelope}).encode()
    if len(wire) > MAX_FRAME_BYTES:
        return None
    try:
        raw = _post(descriptor["port"], descriptor["token"], wire, timeout)
        if raw is None:
            return None  # No request bytes were sent.
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or type(value.get("hook_protocol")) is not int
            or value["hook_protocol"] != HOOK_PROTOCOL
            or value.get("request_id") != identifier
            or value.get("host") != host
            or value.get("environment_sha256") != envelope["environment_sha256"]
        ):
            raise ValueError("Unbound worker response")
        if value.get("status") == "declined" and value.get("dispatched") is False:
            return None
        if (
            value.get("status") != "handled"
            or value.get("dispatched") is not True
            or not isinstance(value.get("output"), dict)
            or not isinstance(value.get("adapter_status"), str)
        ):
            raise ValueError("Invalid worker acknowledgement")
        return value
    except Exception as exc:
        raise HookOutcomeUnknown("Hook worker outcome is unknown") from exc


def _fallback(host: str, raw: str, event: str) -> int:
    from importlib import import_module

    module = import_module(
        "mnemos.adapters.codex" if host == "codex" else "mnemos.adapters.claude_code"
    )
    previous = sys.stdin
    try:
        sys.stdin = io.StringIO(raw)
        return int(module.main([event]))
    finally:
        sys.stdin = previous


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] not in {"codex", "claude-code"}:
        raise ValueError("Expected codex or claude-code")
    host = arguments[0]
    event = arguments[1] if len(arguments) > 1 else ""
    from .adapters.diagnostics import emit_hook_diagnostic
    from .adapters.hook_input import read_hook_input

    try:
        raw = read_hook_input(sys.stdin)
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return 0
        supplied_event = payload.get("hook_event_name")
        if isinstance(supplied_event, str) and supplied_event.strip():
            event = supplied_event.strip()
        if not event:
            return 0
        path = os.environ.get(WORKER_DESCRIPTOR_ENV)
        result = (
            request_hook(host, event, payload, cwd=str(Path.cwd()), descriptor_path=path)
            if path
            else None
        )
        if result is None:
            return _fallback(host, raw, event)
        if result.get("degraded") is True:
            emit_hook_diagnostic(
                host=host, event=event, status=result["adapter_status"], reason=result.get("reason")
            )
        if result["output"]:
            print(json.dumps(result["output"], separators=(",", ":")))
    except Exception as exc:
        emit_hook_diagnostic(host=host, event=event, status="error", exception=exc)
        if host == "codex" and event == "Stop":
            print('{"continue":true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
