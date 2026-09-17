"""Explicit cross-environment JSON bridge for a local Mnemos service.

The bridge is intentionally a one-request process. A WSL or other client must
provide an explicit JSON argv; this module never guesses a host address, opens
a listener, or invokes a shell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import MnemosClient

DEFAULT_MAX_REQUEST_BYTES = 1_000_000
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_DEADLINE_SECONDS = 15.0


def parse_bridge_command(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse the explicit JSON argv used by a non-Windows bridge client."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("MNEMOS_BRIDGE_COMMAND must be a JSON argv array") from exc
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("MNEMOS_BRIDGE_COMMAND must be a non-empty JSON argv array")
    if any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise ValueError("MNEMOS_BRIDGE_COMMAND argv entries must be non-empty strings")
    return tuple(parsed)


def _json_result(status: str, *, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": 1, "status": status, "committed": False}
    if error:
        result["error"] = error
    return result


class BridgeClient:
    """Client for one explicitly configured bridge executable."""

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        deadline: float = DEFAULT_DEADLINE_SECONDS,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.command = parse_bridge_command(command)
        if deadline <= 0 or max_request_bytes <= 0 or max_response_bytes <= 0:
            raise ValueError("bridge limits must be positive")
        self.deadline = float(deadline)
        self.max_request_bytes = int(max_request_bytes)
        self.max_response_bytes = int(max_response_bytes)

    @classmethod
    def from_env(cls) -> "BridgeClient":
        command = os.environ.get("MNEMOS_BRIDGE_COMMAND")
        if not command:
            raise ValueError("MNEMOS_BRIDGE_COMMAND is required for bridge transport")
        return cls(
            command,
            deadline=float(os.environ.get("MNEMOS_BRIDGE_DEADLINE", DEFAULT_DEADLINE_SECONDS)),
            max_request_bytes=int(
                os.environ.get("MNEMOS_BRIDGE_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES)
            ),
            max_response_bytes=int(
                os.environ.get("MNEMOS_BRIDGE_MAX_RESPONSE_BYTES", DEFAULT_MAX_RESPONSE_BYTES)
            ),
        )

    async def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        queue_on_failure: bool | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(
            {"schema_version": 1, "operation": operation, "payload": dict(payload or {})},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > self.max_request_bytes:
            return _json_result("degraded", error="bridge request exceeds size limit")
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(body)
            await process.stdin.drain()
            process.stdin.close()
            output = await asyncio.wait_for(
                process.stdout.read(self.max_response_bytes + 1), self.deadline
            )
            if len(output) > self.max_response_bytes:
                process.terminate()
                await process.wait()
                return _json_result("degraded", error="bridge response exceeds size limit")
            await asyncio.wait_for(process.wait(), self.deadline)
            if process.returncode != 0:
                return _json_result("degraded", error="bridge process failed")
            result = json.loads(output.decode("utf-8"))
            if not isinstance(result, dict):
                return _json_result("degraded", error="bridge returned a non-object response")
            return dict(result)
        except asyncio.TimeoutError:
            if process is not None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 1.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    process.kill()
            return _json_result("degraded", error="bridge request timed out")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return _json_result("degraded", error=f"bridge request failed: {exc}")

    async def capture(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.request("capture", payload)

    async def retrieve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.request("retrieve", payload, queue_on_failure=False)


async def _serve_once(*, descriptor: Path, max_request_bytes: int, deadline: float) -> int:
    raw = sys.stdin.buffer.read(max_request_bytes + 1)
    if len(raw) > max_request_bytes:
        print(json.dumps(_json_result("degraded", error="bridge request exceeds size limit")))
        return 2
    try:
        request = json.loads(raw.decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("bridge request must be an object")
        operation = request.get("operation")
        payload = request.get("payload", {})
        if not isinstance(operation, str) or not operation.strip() or not isinstance(payload, dict):
            raise ValueError("bridge request requires operation and object payload")
        client = MnemosClient(
            descriptor_path=descriptor,
            auto_start=True,
            startup_timeout=deadline,
            request_timeout=deadline,
        )
        normalized_operation = operation.strip()
        queue_on_failure = None if normalized_operation in {"capture", "mnemos_capture"} else False
        result = await asyncio.wait_for(
            client.request(normalized_operation, payload, queue_on_failure=queue_on_failure),
            deadline,
        )
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        await client.close()
        return 0
    except Exception as exc:
        print(json.dumps(_json_result("degraded", error=str(exc)), separators=(",", ":")))
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-request Mnemos bridge")
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args(argv)
    if args.max_request_bytes <= 0 or args.deadline <= 0:
        return 2
    return asyncio.run(
        _serve_once(
            descriptor=args.descriptor.expanduser(),
            max_request_bytes=args.max_request_bytes,
            deadline=args.deadline,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
