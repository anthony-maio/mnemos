"""Offline evidence checks for a native Hermes three-host phase."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hermes_trace_v2 import correlate_native_tools
from .ollama_readiness_v2 import require_loaded_profile


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool arguments are not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be objects")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _reject_ambiguous_pending(trace: list[dict[str, Any]]) -> None:
    pending: dict[tuple[str, str], int] = {}
    for row in trace:
        operation = row.get("operation")
        if operation == "model_response":
            for call in row.get("tool_calls", []):
                key = (call.get("name"), _canonical(call.get("arguments")))
                pending[key] = pending.get(key, 0) + 1
                if pending[key] > 1:
                    raise ValueError("ambiguous simultaneous native tool calls")
        elif operation == "tool_result":
            key = (row.get("name"), _canonical(row.get("args")))
            if key not in pending:
                raise ValueError("tool result has no pending native call")
            pending[key] -= 1
            if pending[key] == 0:
                del pending[key]


def _request_artifacts(phase_root: Path) -> list[tuple[dict[str, Any], bytes]]:
    request_root = phase_root / "provider-requests"
    metadata_paths = sorted(request_root.glob("request-*.json"))
    if not metadata_paths:
        raise ValueError("no archived provider request metadata")
    result: list[tuple[dict[str, Any], bytes]] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("provider request metadata must be an object")
        body_name = metadata.get("body_file")
        if not isinstance(body_name, str) or Path(body_name).name != body_name:
            raise ValueError("provider request body path is invalid")
        body_path = request_root / body_name
        body = body_path.read_bytes()
        if metadata.get("method") != "POST" or metadata.get("url") not in {
            "http://127.0.0.1:11434/v1/chat/completions",
            "http://localhost:11434/v1/chat/completions",
        }:
            raise ValueError("provider request endpoint is not the pinned local endpoint")
        if (
            metadata.get("byte_count") != len(body)
            or metadata.get("sha256") != hashlib.sha256(body).hexdigest()
        ):
            raise ValueError("provider request bytes do not match metadata")
        result.append((metadata, body))
    return result


def _wire_calls(requests: list[tuple[dict[str, Any], bytes]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for _metadata, body in requests:
        payload = json.loads(body.decode("utf-8"))
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list):
            raise ValueError("provider request has no messages")
        assistants: dict[str, dict[str, Any]] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
                for call in message["tool_calls"]:
                    function = call.get("function") if isinstance(call, dict) else None
                    call_id = call.get("id") if isinstance(call, dict) else None
                    if isinstance(function, dict) and isinstance(call_id, str):
                        assistants[call_id] = {
                            "provider_call_id": call_id,
                            "name": function.get("name"),
                            "args": _canonical(function.get("arguments")),
                        }
            elif message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if isinstance(call_id, str) and call_id in assistants:
                    item = dict(assistants[call_id])
                    item["result"] = _canonical(message.get("content"))
                    found.append(item)
    return found


def _check_requests(
    trace: list[dict[str, Any]], *, model: str, context_length: int
) -> list[dict[str, Any]]:
    requests = [row for row in trace if row.get("operation") == "model_request"]
    if not requests:
        raise ValueError("trace has no model requests")
    for row in requests:
        if row.get("model") != model or row.get("reasoning_effort") != "none":
            raise ValueError("model request is not pinned")
        if row.get("contains_foreign_label") is not False:
            raise ValueError("foreign label reached provider input")
        options = row.get("options") or {}
        if options.get("num_ctx") != context_length:
            raise ValueError("model request context length is not pinned")
        for key, expected in {"temperature": 0, "presence_penalty": 0, "seed": 42}.items():
            if row.get(key) not in (None, expected) or options.get(key, expected) != expected:
                raise ValueError("model request contains a conflicting sampling override")
    return requests


def hermes_evidence(
    trace: list[dict[str, Any]],
    phase_root: Path,
    *,
    model: str,
    digest: str,
    context_length: int = 65536,
) -> dict[str, Any]:
    """Validate native correlation, provider wire evidence, and effective context."""
    _reject_ambiguous_pending(trace)
    calls = correlate_native_tools(trace)
    requests = _check_requests(trace, model=model, context_length=context_length)
    if any(row.get("operation") == "provider_error" for row in trace):
        raise ValueError("provider error present in native trace")
    responses = [row for row in trace if row.get("operation") == "model_response"]
    if not responses:
        raise ValueError("trace has no model responses")
    loaded = [row for row in trace if row.get("operation") == "loaded_models_after_response"]
    verified = [row for row in trace if row.get("operation") == "loaded_profile_verified"]
    expected_numbers = list(range(1, len(responses) + 1))
    if [row.get("response_number") for row in loaded] != expected_numbers:
        raise ValueError("loaded model checks do not cover every response in order")
    if [row.get("response_number") for row in verified] != expected_numbers:
        raise ValueError("loaded profile checks do not cover every response in order")
    for row in loaded:
        loaded_result = row.get("result")
        if not isinstance(loaded_result, dict):
            raise ValueError("loaded model check has no structured result")
        require_loaded_profile(
            loaded_result, model=model, expected_digest=digest, context_length=context_length
        )
    request_artifacts = _request_artifacts(Path(phase_root))
    archived = _wire_calls(request_artifacts)
    provider_ids: dict[str, str] = {}
    for response_number, row in enumerate(responses, 1):
        for call in row.get("tool_calls", []):
            call_id = call.get("id")
            provider_id = call.get("provider_call_id")
            if not isinstance(call_id, str) or not isinstance(provider_id, str) or not provider_id:
                raise ValueError("model response lacks provider call identity")
            if call_id != f"response-{response_number}:{provider_id}":
                raise ValueError("model response call identity is not provider-correlated")
            provider_ids[call_id] = provider_id
    allowed_wire_results = {
        (
            provider_ids[call["call_id"]],
            call["name"],
            _canonical(call["args"]),
            _canonical(call["result"]),
        )
        for call in calls
    }
    if any(
        (item["provider_call_id"], item["name"], item["args"], item["result"])
        not in allowed_wire_results
        for item in archived
    ):
        raise ValueError("provider wire contains an altered or unobserved native result")
    for call in calls:
        provider_id = provider_ids.get(call["call_id"])
        matches = [
            item
            for item in archived
            if item["provider_call_id"] == provider_id
            and item["name"] == call["name"]
            and item["args"] == _canonical(call["args"])
        ]
        expected_result = _canonical(call["result"])
        if not any(item["result"] == expected_result for item in matches):
            raise ValueError("native tool call is missing or altered on provider wire")
        call["model_call_correlated"] = True
        call["correlation_level"] = "unambiguous_native_arguments_with_wire_call_id_and_result"
    final_response = responses[-1]
    if final_response.get("tool_calls") or not isinstance(final_response.get("content"), str):
        raise ValueError("last native response is not a final answer")
    final = final_response["content"]
    if not final.strip():
        raise ValueError("native trace has no nonempty final answer")
    return {
        "calls": calls,
        "final_answer": final,
        "model_requests": requests,
        "recorded_context": json.dumps(trace, ensure_ascii=False, sort_keys=True),
        "context_coverage": "native trace correlation plus archived provider request bytes",
        "provenance": {
            "phase_root": str(Path(phase_root).resolve()),
            "request_sha256": [metadata["sha256"] for metadata, _body in request_artifacts],
        },
    }


def require_scope_exclusion(
    calls: list[dict[str, Any]], *, forbidden_ids: set[str], forbidden_scope_ids: set[str]
) -> None:
    """Reject forbidden fact and scope identifiers anywhere in structured results."""

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"id", "fact_id"} and isinstance(child, str) and child in forbidden_ids:
                    raise ValueError("forbidden fact identity appeared in native result")
                if (
                    key in {"scope_id", "project_id"}
                    and isinstance(child, str)
                    and child in forbidden_scope_ids
                ):
                    raise ValueError("forbidden scope identity appeared in native result")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for call in calls:
        walk(call.get("result"))
