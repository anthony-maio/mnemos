"""Strict normalization for selected native Codex and Claude tool traces.

This module accepts only the two saved provider formats used by the native
archives.  It does not infer a tool call from assistant prose or from a
database row.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast

_RAW_TOOLS = {
    "mnemos_store",
    "mnemos_consolidate",
    "mnemos_remember",
    "mnemos_retrieve",
    "mnemos_inspect",
    "mnemos_correct",
    "mnemos_forget",
    "mnemos_forget_preview",
    "mnemos_forget_related",
    "mnemos_job_status",
}
_CLAUDE_PREFIX = "mcp__plugin_mnemos-memory_mnemos__"
_BAD_STATUS = {"rejected", "degraded", "error", "failed", "failure"}


@dataclass(frozen=True)
class TraceProvenance:
    """Controller-supplied origin for one native session trace."""

    host: str
    process_id: int | str
    session_id: str
    source_file: str
    source_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("trace host is required")
        if isinstance(self.process_id, bool) or (
            not isinstance(self.process_id, (int, str))
            or (isinstance(self.process_id, int) and self.process_id <= 0)
            or (isinstance(self.process_id, str) and not self.process_id.strip())
        ):
            raise ValueError("trace process_id is required")
        for name, value in (("session_id", self.session_id), ("source_file", self.source_file)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"trace {name} is required")
        if not _sha256(self.source_file_sha256):
            raise ValueError("source_file_sha256 must be a SHA-256 digest")


@dataclass(frozen=True)
class NativeToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]
    provenance: TraceProvenance
    line_number: int
    line_sha256: str
    provider_call_id: str | None = None
    model_call_correlated: bool = False


def require_tool_sequence(
    calls: Sequence[NativeToolCall],
    expected_names: Sequence[str],
    *,
    require_model_correlation: bool = True,
) -> tuple[NativeToolCall, ...]:
    """Require every expected native operation, independently and in order."""

    if len(calls) != len(expected_names):
        raise ValueError("native tool count differs from the expected phase")
    for call, expected in zip(calls, expected_names):
        if call.name != expected:
            raise ValueError("native tool sequence differs from the expected phase")
        if require_model_correlation and not call.model_call_correlated:
            raise ValueError("native tool lacks direct model-call correlation")
    return tuple(calls)


def load_source_records(path: Path, *, provenance: TraceProvenance) -> tuple[dict[str, Any], ...]:
    """Bind complete native JSONL bytes and source lines before normalization.

    Session/process provenance is supplied by the controller. This loader does
    not infer process ownership from a filename or authenticate a native host.
    """
    if path.resolve() != Path(provenance.source_file).resolve():
        raise ValueError("Trace path differs from its provenance")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance.source_file_sha256:
        raise ValueError("Trace source bytes changed")
    records = []
    for number, line in enumerate(raw.splitlines(), start=1):
        record = _object(json.loads(line.decode("utf-8")), "native JSONL record")
        if "line" in record or "line_sha256" in record:
            raise ValueError("Native record contains reserved locator fields")
        records.append({**record, "line": number, "line_sha256": hashlib.sha256(line).hexdigest()})
    return tuple(records)


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical(value: Any, label: str) -> str:
    try:
        if isinstance(value, Mapping):
            value = dict(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
    return _object(value, "tool arguments")


def _result(value: Any) -> dict[str, Any]:
    value = _object(value, "tool result")
    if value.get("isError") is True or value.get("is_error") is True:
        raise ValueError("tool result was unsuccessful")
    if "status" in value and (
        not isinstance(value["status"], str) or value["status"].lower() in _BAD_STATUS
    ):
        raise ValueError("tool result was unsuccessful")
    if value.get("error"):
        raise ValueError("tool result contains an error")
    return cast(dict[str, Any], value)


def _frozen_object(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(copy.deepcopy(value))


def _locator(entry: Mapping[str, Any], provenance: TraceProvenance) -> tuple[int, str]:
    line = entry.get("line")
    digest = entry.get("line_sha256")
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise ValueError("trace line number is required")
    if not _sha256(digest):
        raise ValueError("trace line_sha256 must be a SHA-256 digest")
    return line, cast(str, digest)


def _tool_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("native tool name is required")
    if value.startswith(_CLAUDE_PREFIX):
        value = value[len(_CLAUDE_PREFIX) :]
    if value not in _RAW_TOOLS:
        raise ValueError(f"unsupported native tool: {value}")
    return cast(str, value)


def _codex_invocation(payload: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    invocation = _object(payload.get("invocation"), "Codex invocation")
    if invocation.get("server") != "mnemos":
        raise ValueError("Codex trace contains a non-Mnemos invocation")
    name = _tool_name(invocation.get("tool"))
    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("Codex invocation requires a call ID")
    return call_id, name, _arguments(invocation.get("arguments"))


def _codex_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _object(payload.get("result"), "Codex result")
    if set(result) != {"Ok"}:
        raise ValueError("Codex result contains Ok and/or Err alternatives")
    ok = result.get("Ok")
    if not isinstance(ok, dict) or ok.get("isError") is not False:
        raise ValueError("Codex result was unsuccessful")
    structured = _object(ok.get("structuredContent"), "Codex structured result")
    content = ok.get("content")
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise ValueError("Codex result content is malformed")
    structured_value = structured.get("result")
    content_value = content[0].get("text")
    if not isinstance(structured_value, str) or not isinstance(content_value, str):
        raise ValueError("Codex result is not JSON text")
    try:
        structured_value = json.loads(structured_value)
        content_value = json.loads(content_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Codex result is not JSON") from exc
    if structured_value != content_value:
        raise ValueError("Codex structured/content results differ")
    return _result(structured_value)


def parse_codex_mcp_records(
    records: Sequence[Mapping[str, Any]], *, provenance: TraceProvenance
) -> tuple[NativeToolCall, ...]:
    """Correlate saved Codex MCP begin/end records in chronological order."""

    pending: dict[str, tuple[str, str, dict[str, Any]]] = {}
    seen: set[str] = set()
    matched: list[NativeToolCall] = []
    response_calls: dict[str, tuple[str, str, int]] = {}
    saw_native = False
    previous_line = 0
    for entry in records:
        if not isinstance(entry, Mapping):
            raise ValueError("Codex trace entry must be an object")
        line, line_hash = _locator(entry, provenance)
        if line <= previous_line:
            raise ValueError("Codex source lines must be strictly increasing")
        previous_line = line
        payload = _object(entry.get("payload"), "Codex payload")
        kind = payload.get("type")
        if entry.get("type") == "response_item":
            # Current Codex can route MCP through model-authored exec code. Its
            # exec IDs differ from native MCP IDs, and sessions omit MCP begin.
            # Preserve completion evidence below, but never assert direct model
            # correlation by inspecting JavaScript text or timing adjacency.
            if kind == "custom_tool_call":
                if payload.get("name") != "exec":
                    raise ValueError("Unsupported Codex custom model tool")
                continue
            if kind != "function_call":
                continue
            item = payload
            if item.get("name") == "wait":
                continue
            call_id = item.get("call_id") or item.get("id")
            raw_name = item.get("name")
            if isinstance(raw_name, str) and raw_name.startswith("mcp__mnemos__"):
                raw_name = raw_name[len("mcp__mnemos__") :]
            name = _tool_name(raw_name)
            if not isinstance(call_id, str) or not call_id.strip() or call_id in response_calls:
                raise ValueError("duplicate Codex model call ID")
            arguments = _arguments(item.get("arguments"))
            response_calls[call_id] = (name, _canonical(arguments, "arguments"), line)
            continue
        if kind not in {"mcp_tool_call_begin", "mcp_tool_call_end"}:
            continue
        saw_native = True
        call_id, name, arguments = _codex_invocation(payload)
        if call_id in seen and kind == "mcp_tool_call_begin":
            raise ValueError("duplicate Codex call ID")
        if kind == "mcp_tool_call_begin":
            pending[call_id] = (name, _canonical(arguments, "arguments"), arguments)
            seen.add(call_id)
            continue
        if call_id not in pending:
            if call_id in seen:
                raise ValueError("duplicate Codex call ID")
            # Native sessions currently retain completion events without begin.
            seen.add(call_id)
            expected_arguments = arguments
        else:
            expected_name, expected_args, expected_arguments = pending.pop(call_id)
            if expected_name != name or expected_args != _canonical(arguments, "arguments"):
                raise ValueError("Codex begin/end arguments differ")
        matched.append(
            NativeToolCall(
                call_id=call_id,
                name=name,
                arguments=_frozen_object(expected_arguments),
                result=_frozen_object(_codex_result(payload)),
                provenance=provenance,
                line_number=line,
                line_sha256=line_hash,
            )
        )
    if not saw_native:
        raise ValueError("Codex trace has no native MCP call")
    if pending:
        raise ValueError("Codex calls remain undispatched")
    if response_calls:
        if set(response_calls) != {call.call_id for call in matched}:
            raise ValueError("Codex model and MCP call sets differ")
        for call_id, (name, expected_key, model_line) in response_calls.items():
            completed = next((item for item in matched if item.call_id == call_id), None)
            if completed is None:
                raise ValueError("Codex model call has no matching MCP result")
            if model_line >= completed.line_number:
                raise ValueError("Codex model call must precede its MCP result")
            if (
                completed.name != name
                or _canonical(completed.arguments, "arguments") != expected_key
            ):
                raise ValueError("Codex model call and MCP invocation differ")
    if not matched:
        raise ValueError("Codex trace has no completed native MCP call")
    if response_calls:
        matched = [
            NativeToolCall(
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
                result=item.result,
                provenance=item.provenance,
                line_number=item.line_number,
                line_sha256=item.line_sha256,
                provider_call_id=item.provider_call_id,
                model_call_correlated=True,
            )
            for item in matched
        ]
    return tuple(matched)


def _claude_result(node: Mapping[str, Any]) -> dict[str, Any]:
    if node.get("is_error") is True:
        raise ValueError("Claude tool result was unsuccessful")
    content = node.get("content")
    text: Any
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    else:
        text = content
    if not isinstance(text, str):
        raise ValueError("Claude tool result content is malformed")
    try:
        value: Any = json.loads(text)
        if isinstance(value, dict) and "result" in value:
            value = json.loads(value["result"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Claude tool result is not JSON") from exc
    return _result(value)


def parse_claude_tool_records(
    records: Sequence[Mapping[str, Any]], *, provenance: TraceProvenance
) -> tuple[NativeToolCall, ...]:
    """Correlate Claude assistant tool_use/tool_result JSONL records."""

    pending: dict[str, tuple[str, str, dict[str, Any]]] = {}
    issued: set[tuple[str, str]] = set()
    matched: list[NativeToolCall] = []
    saw_response = False
    previous_line = 0
    for entry in records:
        if not isinstance(entry, Mapping):
            raise ValueError("Claude trace entry must be an object")
        line, line_hash = _locator(entry, provenance)
        if line <= previous_line:
            raise ValueError("Claude source lines must be strictly increasing")
        previous_line = line
        if entry.get("type") not in {"assistant", "user"}:
            if "message" not in entry:
                continue
            raise ValueError("Claude message has an unexpected record type")
        message = _object(entry.get("message"), "Claude message")
        nodes = message.get("content")
        if not isinstance(nodes, list):
            raise ValueError("Claude message content must be a list")
        uses = [node for node in nodes if isinstance(node, dict) and node.get("type") == "tool_use"]
        results = [
            node for node in nodes if isinstance(node, dict) and node.get("type") == "tool_result"
        ]
        if uses:
            saw_response = True
            if entry.get("type") != "assistant" or message.get("role") != "assistant":
                raise ValueError("Claude tool_use must be in an assistant message")
            response_id = message.get("id")
            if not isinstance(response_id, str) or not response_id.strip():
                raise ValueError("Claude assistant response ID is required")
            response_ids: set[str] = set()
            for node in uses:
                provider_id = node.get("id")
                if (
                    not isinstance(provider_id, str)
                    or not provider_id.strip()
                    or provider_id in response_ids
                ):
                    raise ValueError("duplicate Claude provider call ID")
                response_ids.add(provider_id)
                raw_name = node.get("name")
                if not isinstance(raw_name, str) or not raw_name.startswith(_CLAUDE_PREFIX):
                    raise ValueError("Claude tool name is not the verified Mnemos prefix")
                name = _tool_name(raw_name)
                arguments = _arguments(node.get("input"))
                if (response_id, provider_id) in issued or provider_id in pending:
                    raise ValueError("duplicate or ambiguous Claude provider call ID")
                issued.add((response_id, provider_id))
                call_id = f"{response_id}:{provider_id}"
                pending[provider_id] = (call_id, name, arguments)
        for node in results:
            if entry.get("type") != "user" or message.get("role") != "user":
                raise ValueError("Claude tool_result must be in a user message")
            provider_id = node.get("tool_use_id")
            if not isinstance(provider_id, str) or not provider_id.strip():
                raise ValueError("Claude tool_result requires tool_use_id")
            if provider_id not in pending:
                raise ValueError("Claude tool_result has no preceding unmatched tool_use")
            call_id, name, arguments = pending.pop(provider_id)
            result = _claude_result(node)
            matched.append(
                NativeToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=_frozen_object(arguments),
                    result=_frozen_object(result),
                    provenance=provenance,
                    line_number=line,
                    line_sha256=line_hash,
                    provider_call_id=provider_id,
                    model_call_correlated=True,
                )
            )
    if not saw_response:
        raise ValueError("Claude trace has no tool_use response")
    if pending:
        raise ValueError("Claude calls remain undispatched")
    if not matched:
        raise ValueError("Claude trace has no correlated tool_result")
    return tuple(matched)
