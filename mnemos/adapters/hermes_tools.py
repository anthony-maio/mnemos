"""Scoped lifecycle tools exposed by the Hermes memory provider."""

from __future__ import annotations

import json
import hashlib
from typing import Any, cast

from ..scopes import ScopeContext

_SCHEMAS = (
    {
        "name": "mnemos_retrieve",
        "description": "Retrieve current Mnemos facts in the provider-owned project scope.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 10000},
                "top_k": {"type": "integer", "minimum": 0, "maximum": 100},
                "max_context_tokens": {"type": "integer", "minimum": 0, "maximum": 32000},
            },
            "required": ["query"],
        },
    },
    {
        "name": "mnemos_inspect",
        "description": "Inspect one exact Mnemos fact in the provider-owned scope.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact_id": {"type": "string", "minLength": 1, "maxLength": 512}},
            "required": ["fact_id"],
        },
    },
    {
        "name": "mnemos_remember",
        "description": "Remember one explicitly supplied user fact in the provider-owned scope.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {"type": "string", "minLength": 1, "maxLength": 10000},
                "source_ref": {"type": "string", "minLength": 1, "maxLength": 4096},
                "request_id": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["content"],
        },
    },
    {
        "name": "mnemos_correct",
        "description": "Correct one exact fact using its current optimistic-concurrency version.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fact_id": {"type": "string", "minLength": 1, "maxLength": 512},
                "content": {"type": "string", "minLength": 1, "maxLength": 10000},
                "expected_version": {"type": "integer", "minimum": 1},
                "source_ref": {"type": "string", "minLength": 1, "maxLength": 4096},
                "request_id": {"type": "string", "minLength": 1, "maxLength": 256},
            },
            "required": ["fact_id", "content", "expected_version"],
        },
    },
    {
        "name": "mnemos_forget",
        "description": "Forget one exact fact using its current optimistic-concurrency version.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fact_id": {"type": "string", "minLength": 1, "maxLength": 512},
                "expected_version": {"type": "integer", "minimum": 1},
            },
            "required": ["fact_id", "expected_version"],
        },
    },
    {
        "name": "mnemos_job_status",
        "description": "Read durable status for one Mnemos job.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"job_id": {"type": "string", "minLength": 1, "maxLength": 512}},
            "required": ["job_id"],
        },
    },
)

_TO_OPERATION = {
    "mnemos_retrieve": "retrieve",
    "mnemos_inspect": "inspect",
    "mnemos_remember": "remember",
    "mnemos_correct": "correct",
    "mnemos_forget": "forget",
    "mnemos_job_status": "job",
}
_ALLOWED = {
    "mnemos_retrieve": {"query", "top_k", "max_context_tokens"},
    "mnemos_inspect": {"fact_id"},
    "mnemos_remember": {"content", "source_ref", "request_id"},
    "mnemos_correct": {"fact_id", "content", "expected_version", "source_ref", "request_id"},
    "mnemos_forget": {"fact_id", "expected_version"},
    "mnemos_job_status": {"job_id"},
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return detached Hermes tool schemas."""
    return cast(list[dict[str, Any]], json.loads(json.dumps(_SCHEMAS)))


def _rejected(message: str) -> str:
    return json.dumps({"schema_version": 1, "status": "rejected", "error": message})


def _validate(name: str, args: Any) -> str | None:
    if name not in _TO_OPERATION:
        return f"Unknown Mnemos provider tool: {name}"
    if not isinstance(args, dict):
        return "Tool arguments must be an object"
    extra = set(args) - _ALLOWED[name]
    if extra:
        return "Scope and unknown argument overrides are not accepted"
    required = {
        "mnemos_retrieve": {"query"},
        "mnemos_inspect": {"fact_id"},
        "mnemos_remember": {"content", "request_id"},
        "mnemos_correct": {"fact_id", "content", "expected_version", "request_id"},
        "mnemos_forget": {"fact_id", "expected_version"},
        "mnemos_job_status": {"job_id"},
    }[name]
    if required - args.keys():
        return "Required argument is missing"
    string_limits = {
        "query": 10000,
        "content": 10000,
        "fact_id": 512,
        "job_id": 512,
        "source_ref": 4096,
        "request_id": 256,
    }
    for key, value in args.items():
        if key in string_limits and (not isinstance(value, str) or not value.strip()):
            return f"{key} must be a non-empty string"
        if key in string_limits and len(value) > string_limits[key]:
            return f"{key} is too long"
        if key in {"top_k", "max_context_tokens", "expected_version"} and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return f"{key} must be an integer"
        if key == "expected_version" and value < 1:
            return "expected_version must be positive"
        if key == "top_k" and not 0 <= value <= 100:
            return "top_k is out of bounds"
        if key == "max_context_tokens" and not 0 <= value <= 32000:
            return "max_context_tokens is out of bounds"
    return None


async def dispatch_tool(
    client: Any, context: ScopeContext, tool_name: str, args: dict[str, Any]
) -> str:
    """Dispatch one provider tool with scope injected by the initialized provider."""
    error = _validate(tool_name, args)
    if error:
        return _rejected(error)
    payload: dict[str, Any] = {"context": context.model_dump(mode="json")}
    if tool_name == "mnemos_retrieve":
        payload.update(
            {
                "query": args["query"],
                "allowed_scopes": ["project"],
                "top_k": args.get("top_k", 5),
                "max_context_tokens": args.get("max_context_tokens", 1200),
                "profile": "baseline",
            }
        )
    elif tool_name == "mnemos_remember":
        payload.update(
            {
                "content": args["content"],
                "role": "user",
                "scope": "project",
                "event_id": _stable_event_id(context, tool_name, args["request_id"]),
            }
        )
        if "source_ref" in args:
            payload["source_ref"] = args["source_ref"]
    else:
        payload.update(args)
        if tool_name == "mnemos_correct":
            payload["event_id"] = _stable_event_id(context, tool_name, args["request_id"])
            payload.pop("request_id", None)
    try:
        result = await client.request(_TO_OPERATION[tool_name], payload, queue_on_failure=False)
    except Exception as exc:
        return json.dumps(
            {
                "schema_version": 1,
                "status": "degraded",
                "committed": False,
                "error": f"Mnemos service request failed: {exc}",
            }
        )
    if not isinstance(result, dict):
        return json.dumps(
            {
                "schema_version": 1,
                "status": "degraded",
                "committed": False,
                "error": "Mnemos service returned a non-object result",
            }
        )
    return json.dumps(result, ensure_ascii=False)


def _stable_event_id(context: ScopeContext, tool_name: str, request_id: str) -> str:
    identity = {
        "tool": tool_name,
        "request_id": request_id,
        "context": context.model_dump(mode="json"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "hermes-tool:" + hashlib.sha256(encoded).hexdigest()
