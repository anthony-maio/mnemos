"""One-shot OpenAI-compatible Ollama readiness check with effective context proof."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable
from urllib import request

ENDPOINT = "http://127.0.0.1:11434"
_transport: Callable[[str, Any, float], tuple[int, bytes]]


def _urllib_transport(route: str, payload: Any, timeout: float) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    url = ENDPOINT + ("/v1/chat/completions" if route == "chat" else "/api/" + route)
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return int(response.status), response.read()


_transport = _urllib_transport


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _request_once(
    output: Path, route: str, payload: Any, timeout: float, attempts: list[dict[str, Any]]
) -> bytes:
    started = time.monotonic()
    ordinal = len(attempts) + 1
    request_bytes = b"" if payload is None else json.dumps(payload).encode("utf-8")
    url = ENDPOINT + ("/v1/chat/completions" if route == "chat" else "/api/" + route)
    entry: dict[str, Any] = {
        "route": route,
        "request": payload,
        "url": url,
        "method": "POST" if payload is not None else "GET",
        "request_bytes_file": f"http-{ordinal:03d}.request",
        "started_at": _utc(),
        "timeout_seconds": timeout,
    }
    (output / entry["request_bytes_file"]).write_bytes(request_bytes)
    try:
        status, body = _transport(route, payload, timeout)
    except BaseException as exc:
        raw_status = getattr(exc, "code", None)
        status_code = (
            raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else None
        )
        raw_body = b""
        reader = getattr(exc, "read", None)
        if callable(reader):
            try:
                candidate = reader()
                if isinstance(candidate, bytes):
                    raw_body = candidate
            except BaseException:
                raw_body = b""
        entry.update(
            status_code=status_code,
            finished_at=_utc(),
            duration_seconds=time.monotonic() - started,
            error_type=type(exc).__name__,
            error=str(exc),
            body_file=f"http-{ordinal:03d}.body",
        )
        (output / entry["body_file"]).write_bytes(raw_body)
        attempts.append(entry)
        _save_json(output / f"http-{ordinal:03d}.json", entry)
        raise
    if not isinstance(status, int) or isinstance(status, bool):
        raise TypeError("transport returned an invalid HTTP status")
    if not isinstance(body, bytes):
        raise TypeError("transport returned a non-byte response body")
    entry.update(
        status_code=status, finished_at=_utc(), duration_seconds=time.monotonic() - started
    )
    entry["body_file"] = f"http-{ordinal:03d}.body"
    (output / entry["body_file"]).write_bytes(body)
    if status != 200:
        entry.update(error_type="RuntimeError", error=f"Ollama {route} returned HTTP {status}")
    attempts.append(entry)
    _save_json(output / f"http-{ordinal:03d}.json", entry)
    if status != 200:
        raise RuntimeError(f"Ollama {route} returned HTTP {status}")
    return body


def _json(body: bytes, route: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Ollama {route} response was not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Ollama {route} response was not an object")
    return value


def _model(rows: dict[str, Any], model: str) -> dict[str, Any] | None:
    models = rows.get("models")
    if not isinstance(models, list):
        return None
    matches = [row for row in models if isinstance(row, dict) and row.get("name") == model]
    return matches[0] if len(matches) == 1 else None


def _show_parameters(show: dict[str, Any]) -> dict[str, str]:
    parameters = show.get("parameters")
    if isinstance(parameters, dict):
        return {str(key): str(value) for key, value in parameters.items()}
    if not isinstance(parameters, str):
        raise ValueError("model alias did not report parameters")
    values: dict[str, str] = {}
    for line in parameters.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            values[fields[0]] = fields[1]
    return values


def _require_parameters(show: dict[str, Any], context_length: int) -> None:
    values = _show_parameters(show)
    expected = {
        "num_ctx": str(context_length),
        "temperature": "0",
        "seed": "42",
        "presence_penalty": "0",
        "top_k": "20",
        "top_p": "0.95",
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("model alias parameters do not match readiness profile")


def require_loaded_profile(
    ps: dict[str, Any], *, model: str, expected_digest: str, context_length: int
) -> dict[str, Any]:
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        raise ValueError("context_length must be a positive integer")
    rows = ps.get("models")
    if not isinstance(rows, list):
        raise ValueError("loaded model response did not contain a models list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and (row.get("name") == model or row.get("model") == model)
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one loaded model row")
    row = matches[0]
    if any(key in row and row[key] != model for key in ("name", "model")):
        raise ValueError("loaded model alias fields are inconsistent")
    if row.get("digest") != expected_digest:
        raise ValueError("loaded model digest does not match readiness")
    context = row.get("context_length")
    vram = row.get("size_vram")
    if (
        isinstance(context, bool)
        or not isinstance(context, int)
        or context < 1
        or context != context_length
        or isinstance(vram, bool)
        or not isinstance(vram, (int, float))
        or not math.isfinite(float(vram))
        or vram <= 0
    ):
        raise ValueError("loaded model context or GPU residency does not match readiness")
    return row


def _run_readiness(
    output: Path,
    *,
    model: str,
    expected_digest: str,
    context_length: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "status": "running",
        "model": model,
        "expected_digest": expected_digest,
        "context_length": context_length,
        "endpoint": ENDPOINT + "/v1/chat/completions",
        "started_at": _utc(),
        "http_attempts": attempts,
    }
    _save_json(output / "readiness.json", record)
    try:
        version = _json(_request_once(output, "version", None, 10, attempts), "version")
        tags_before = _json(_request_once(output, "tags", None, 10, attempts), "tags")
        show = _json(_request_once(output, "show", {"model": model}, 10, attempts), "show")
        ps_before = _json(_request_once(output, "ps", None, 10, attempts), "ps")
        before = _model(tags_before, model)
        if before is None or before.get("digest") != expected_digest:
            raise ValueError("model digest mismatch before readiness request")
        _require_parameters(show, context_length)
        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with READY."}],
            "stream": False,
            "think": False,
            "max_tokens": 16,
            "reasoning_effort": "none",
        }
        chat: dict[str, Any] | None = None
        chat_error: BaseException | None = None
        try:
            chat = _json(_request_once(output, "chat", request_body, 180, attempts), "chat")
        except BaseException as exc:
            chat_error = exc
        ps: dict[str, Any] | None = None
        ps_error: BaseException | None = None
        try:
            ps = _json(_request_once(output, "ps", None, 10, attempts), "ps")
        except BaseException as exc:
            ps_error = exc
        tags_after: dict[str, Any] | None = None
        tags_after_error: BaseException | None = None
        try:
            tags_after = _json(_request_once(output, "tags", None, 10, attempts), "tags")
        except BaseException as exc:
            tags_after_error = exc
        if chat_error is not None:
            raise chat_error
        if ps_error is not None:
            raise ps_error
        if tags_after_error is not None:
            raise tags_after_error
        assert chat is not None and ps is not None and tags_after is not None
        after = _model(tags_after, model)
        if after is None or after.get("digest") != expected_digest:
            raise ValueError("model digest mismatch after readiness request")
        choices = chat.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if (
            chat.get("error")
            or not isinstance(choice, dict)
            or not isinstance(choice.get("finish_reason"), str)
            or choice["finish_reason"] not in {"stop", "length"}
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            or message.get("tool_calls")
        ):
            raise ValueError("readiness response was not a successful assistant response")
        loaded = require_loaded_profile(
            ps, model=model, expected_digest=expected_digest, context_length=context_length
        )
        record.update(
            status="ready",
            version=version,
            tags_before=tags_before,
            show=show,
            ps_before=ps_before,
            chat=chat,
            ps_after=ps,
            tags_after=tags_after,
            gpu_residency={"digest": expected_digest, "size_vram": loaded["size_vram"]},
        )
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["finished_at"] = _utc()
        record["http_attempts"] = attempts
        _save_json(output / "readiness.json", record)
    return record


def run_readiness(
    output: Path, *, model: str, expected_digest: str, context_length: int
) -> dict[str, Any]:
    """Run exactly one bounded warmup in a fresh output directory."""
    output = Path(output)
    if output.exists():
        raise ValueError("readiness output directory already exists")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        raise ValueError("context_length must be a positive integer")
    output.mkdir(parents=True)
    return _run_readiness(
        output, model=model, expected_digest=expected_digest, context_length=context_length
    )


if __name__ == "__main__":
    raise SystemExit("Use run_readiness from a pinned workflow")
