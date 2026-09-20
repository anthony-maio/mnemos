from __future__ import annotations

import json
import io
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from benchmarks.native_hosts import ollama_readiness_v2 as readiness

MODEL = "fixture-model"
DIGEST = "sha256:" + "a" * 64


def _responses(
    *,
    content: str = "READY",
    done: bool = True,
    error: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    digest: str = DIGEST,
) -> dict[str, dict[str, Any]]:
    return {
        "version": {"version": "fixture"},
        "tags": {"models": [{"name": MODEL, "digest": digest, "size": 123}]},
        "show": {
            "parameters": "num_ctx 32768\ntemperature 0\nseed 42\npresence_penalty 0\ntop_k 20\ntop_p 0.95"
        },
        "chat": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": tool_calls} if tool_calls is not None else {}),
                    },
                }
            ],
            **({"error": error} if error is not None else {}),
        },
        "ps": {
            "models": [
                {"model": MODEL, "digest": digest, "size_vram": 100, "context_length": 32768}
            ]
        },
    }


class FakeTransport:
    def __init__(
        self, responses: dict[str, dict[str, Any]], *, fail_route: str | None = None
    ) -> None:
        self.responses = responses
        self.fail_route = fail_route
        self.calls: list[tuple[str, Any, float]] = []

    def __call__(self, route: str, payload: Any, timeout: float) -> tuple[int, bytes]:
        self.calls.append((route, payload, timeout))
        if route == self.fail_route:
            raise TimeoutError("fixture timeout")
        if route == "tags" and sum(item[0] == "tags" for item in self.calls) == 2:
            return 200, json.dumps(self.responses.get("tags_after", self.responses[route])).encode(
                "utf-8"
            )
        return 200, json.dumps(self.responses[route]).encode("utf-8")


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, dict[str, Any]],
    **kwargs: Any,
) -> tuple[FakeTransport, dict[str, Any]]:
    transport = FakeTransport(responses, **kwargs)
    monkeypatch.setattr(readiness, "_transport", transport)
    return transport, readiness.run_readiness(
        tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
    )


def test_ready_response_is_accepted_without_exact_punctuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, result = run(tmp_path, monkeypatch, _responses(content="READY."))
    assert result["status"] == "ready"
    assert [route for route, _, _ in transport.calls] == [
        "version",
        "tags",
        "show",
        "ps",
        "chat",
        "ps",
        "tags",
    ]
    assert result["gpu_residency"]["size_vram"] > 0
    assert result["endpoint"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert (tmp_path / "readiness" / "http-005.body").read_bytes()
    assert json.loads((tmp_path / "readiness" / "readiness.json").read_text())["status"] == "ready"
    chat_entry = json.loads((tmp_path / "readiness" / "http-005.json").read_text())
    assert chat_entry["url"] == result["endpoint"]
    assert chat_entry["request"]["max_tokens"] == 16
    assert chat_entry["request"]["reasoning_effort"] == "none"
    assert chat_entry["request"]["think"] is False
    assert "options" not in chat_entry["request"]
    assert (tmp_path / "readiness" / chat_entry["request_bytes_file"]).read_bytes() == json.dumps(
        chat_entry["request"]
    ).encode("utf-8")


@pytest.mark.parametrize("context", [16384, True, 0])
def test_loaded_context_must_match_strict_positive_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: Any
) -> None:
    responses = _responses()
    responses["ps"] = {
        "models": [{"model": MODEL, "digest": DIGEST, "size_vram": 100, "context_length": context}]
    }
    with pytest.raises(ValueError, match="context"):
        run(tmp_path, monkeypatch, responses)


def test_duplicate_loaded_rows_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    responses["ps"] = {
        "models": [
            {"model": MODEL, "digest": DIGEST, "size_vram": 100, "context_length": 32768},
            {
                "name": MODEL,
                "digest": "sha256:" + "b" * 64,
                "size_vram": 100,
                "context_length": 32768,
            },
        ]
    }
    with pytest.raises(ValueError, match="exactly one"):
        run(tmp_path, monkeypatch, responses)


def test_alias_parameters_are_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = _responses()
    responses["show"][
        "parameters"
    ] = "num_ctx 16384\ntemperature 0\nseed 42\npresence_penalty 0\ntop_k 20\ntop_p 0.95"
    with pytest.raises(ValueError, match="parameters"):
        run(tmp_path, monkeypatch, responses)


@pytest.mark.parametrize("include_valid_row", [False, True])
def test_inconsistent_alias_cannot_be_ignored(include_valid_row: bool) -> None:
    row = _responses()["ps"]["models"][0]
    rows = [dict(row, name="another-model")]
    if include_valid_row:
        rows.append(row)
    with pytest.raises(ValueError):
        readiness.require_loaded_profile(
            {"models": rows}, model=MODEL, expected_digest=DIGEST, context_length=32768
        )


@pytest.mark.parametrize("invalid_context", [True, 0, 32768.0])
def test_requested_context_is_strict_positive_integer(tmp_path: Path, invalid_context: Any) -> None:
    with pytest.raises(ValueError, match="context_length"):
        readiness.run_readiness(
            tmp_path / "readiness",
            model=MODEL,
            expected_digest=DIGEST,
            context_length=invalid_context,
        )


def test_digest_mismatch_before_chat_aborts_without_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _responses(digest="sha256:" + "b" * 64)
    transport = FakeTransport(responses)
    monkeypatch.setattr(readiness, "_transport", transport)
    with pytest.raises(ValueError, match="digest"):
        readiness.run_readiness(
            tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
        )
    assert [route for route, _, _ in transport.calls] == ["version", "tags", "show", "ps"]


def test_digest_change_after_chat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _responses()
    responses["tags_after"] = {"models": [{"name": MODEL, "digest": "sha256:" + "b" * 64}]}
    transport = FakeTransport(responses)
    monkeypatch.setattr(readiness, "_transport", transport)
    with pytest.raises(ValueError, match="after"):
        readiness.run_readiness(
            tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
        )
    assert [route for route, _, _ in transport.calls] == [
        "version",
        "tags",
        "show",
        "ps",
        "chat",
        "ps",
        "tags",
    ]


@pytest.mark.parametrize(
    "chat",
    [
        {"choices": []},
        {
            "error": "failed",
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": "READY"}}
            ],
        },
        {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"role": "assistant", "content": "READY", "tool_calls": [{}]},
                }
            ]
        },
    ],
)
def test_malformed_warmup_response_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chat: dict[str, Any],
) -> None:
    responses = _responses()
    responses["chat"] = chat
    with pytest.raises(ValueError, match="readiness"):
        run(tmp_path, monkeypatch, responses)


def test_timeout_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = _responses()
    transport = FakeTransport(responses, fail_route="chat")
    monkeypatch.setattr(readiness, "_transport", transport)
    with pytest.raises(TimeoutError):
        readiness.run_readiness(
            tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
        )
    assert [route for route, _, _ in transport.calls] == [
        "version",
        "tags",
        "show",
        "ps",
        "chat",
        "ps",
        "tags",
    ]
    report = json.loads((tmp_path / "readiness" / "readiness.json").read_text())
    assert report["status"] == "failed"
    assert sum(item["route"] == "chat" for item in report["http_attempts"]) == 1


def test_malformed_json_preserves_raw_response_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MalformedTransport(FakeTransport):
        def __call__(self, route: str, payload: Any, timeout: float) -> tuple[int, bytes]:
            if route == "chat":
                self.calls.append((route, payload, timeout))
                return 200, b"not-json"
            return super().__call__(route, payload, timeout)

    transport = MalformedTransport(_responses())
    monkeypatch.setattr(readiness, "_transport", transport)
    with pytest.raises(ValueError, match="not JSON"):
        readiness.run_readiness(
            tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
        )
    assert (tmp_path / "readiness" / "http-005.body").read_bytes() == b"not-json"


def test_missing_gpu_residency_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = _responses()
    responses["ps"] = {"models": [{"model": MODEL, "digest": DIGEST, "size_vram": 0}]}
    with pytest.raises(ValueError, match="GPU"):
        run(tmp_path, monkeypatch, responses)


def test_http_error_keeps_status_and_raw_body_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HttpErrorTransport(FakeTransport):
        def __call__(self, route: str, payload: Any, timeout: float) -> tuple[int, bytes]:
            self.calls.append((route, payload, timeout))
            if route == "tags" and len(self.calls) == 2:
                raise HTTPError(
                    "http://127.0.0.1:11434/api/tags",
                    503,
                    "busy",
                    Message(),
                    io.BytesIO(b'{"error":"busy"}'),
                )
            return 200, json.dumps(self.responses[route]).encode("utf-8")

    transport = HttpErrorTransport(_responses())
    monkeypatch.setattr(readiness, "_transport", transport)
    with pytest.raises(HTTPError):
        readiness.run_readiness(
            tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
        )
    attempts = json.loads((tmp_path / "readiness" / "readiness.json").read_text())["http_attempts"]
    assert len(attempts) == 2
    assert attempts[-1]["status_code"] == 503
    assert len(list((tmp_path / "readiness").glob("http-*.json"))) == 2
    assert (tmp_path / "readiness" / "http-002.body").read_bytes() == b'{"error":"busy"}'


def test_non_200_response_is_rejected_and_request_bytes_are_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CreatedTransport(FakeTransport):
        def __call__(self, route: str, payload: Any, timeout: float) -> tuple[int, bytes]:
            self.calls.append((route, payload, timeout))
            if route == "tags":
                return 201, json.dumps(self.responses[route]).encode("utf-8")
            return super().__call__(route, payload, timeout)

    transport = CreatedTransport(_responses())
    monkeypatch.setattr(readiness, "_transport", transport)
    with pytest.raises(RuntimeError, match="HTTP 201"):
        readiness.run_readiness(
            tmp_path / "readiness", model=MODEL, expected_digest=DIGEST, context_length=32768
        )
    report = json.loads((tmp_path / "readiness" / "readiness.json").read_text())
    assert len(report["http_attempts"]) == 2
    assert report["http_attempts"][-1]["status_code"] == 201
    assert (tmp_path / "readiness" / "http-001.request").read_bytes() == b""


@pytest.mark.parametrize("size_vram", [True, float("nan"), "100"])
def test_invalid_gpu_residency_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size_vram: Any
) -> None:
    responses = _responses()
    responses["ps"] = {"models": [{"model": MODEL, "digest": DIGEST, "size_vram": size_vram}]}
    with pytest.raises(ValueError, match="GPU"):
        run(tmp_path, monkeypatch, responses)


def test_existing_output_directory_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "readiness"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        readiness.run_readiness(output, model=MODEL, expected_digest=DIGEST, context_length=32768)
    assert marker.read_text() == "keep"
