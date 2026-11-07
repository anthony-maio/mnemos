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


def parse_codex_code_mode_trace(
    directory: Path, *, provenance: TraceProvenance, model: str
) -> tuple[tuple[NativeToolCall, ...], dict[str, str]]:
    """Bind model -> cell -> MCP through native IDs in a schema-1 rollout.

    The controller owns the host process and private trace directory. Hashes
    bind the evidence read here; they do not authenticate an arbitrary file.
    JavaScript equality is an integrity check, never a source of parent IDs.
    Streaming may start a cell before its inference-completed record arrives.
    """
    if provenance.host != "codex" or directory.is_symlink():
        raise ValueError("Expected an owned Codex trace directory")
    root = directory.resolve(strict=True)
    hashes: dict[str, str] = {}
    raw_files: dict[str, bytes] = {}

    def read(relative: str) -> bytes:
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Trace payload escapes its directory")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if relative in hashes and hashes[relative] != digest:
            raise ValueError("Trace payload bytes changed")
        hashes[relative], raw_files[relative] = digest, raw
        return raw

    manifest = _object(json.loads(read("manifest.json")), "rollout manifest")
    session_id = provenance.session_id
    if (
        manifest.get("schema_version") != 1
        or manifest.get("rollout_id") != session_id
        or manifest.get("root_thread_id") != session_id
        or manifest.get("raw_event_log") != "trace.jsonl"
        or manifest.get("payloads_dir") != "payloads"
    ):
        raise ValueError("Rollout manifest differs from native session")
    log_raw = read("trace.jsonl")
    log_provenance = TraceProvenance(
        "codex", provenance.process_id, session_id, str(root / "trace.jsonl"), hashes["trace.jsonl"]
    )
    records = load_source_records(root / "trace.jsonl", provenance=log_provenance)
    session = load_source_records(Path(provenance.source_file), provenance=provenance)
    if [r["payload"].get("id") for r in session if r.get("type") == "session_meta"] != [session_id]:
        raise ValueError("Native session identity differs")
    model_calls: dict[str, str] = {}

    def model_call(item: dict[str, Any], target: dict[str, str]) -> None:
        if item.get("type") == "function_call":
            raise ValueError("Direct calls are outside this Code Mode protocol")
        if item.get("type") != "custom_tool_call":
            return
        identity = item.get("call_id")
        code = item.get("input")
        if (
            item.get("name") != "exec"
            or not isinstance(identity, str)
            or not identity
            or identity in target
            or not isinstance(code, str)
            or not code
        ):
            raise ValueError("Missing, duplicate or unsupported model call")
        target[identity] = code

    for entry in session:
        if entry.get("type") == "response_item":
            model_call(_object(entry.get("payload"), "model item"), model_calls)
    turns = [r["payload"] for r in records if r["payload"].get("type") == "codex_turn_started"]
    if len(turns) != 1 or not isinstance(turns[0].get("codex_turn_id"), str):
        raise ValueError("Exactly one native turn is required")
    turn_id = turns[0]["codex_turn_id"]
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    references: dict[str, tuple[str, str]] = {}
    payload_cache: dict[str, dict[str, Any]] = {}
    ordinary = {
        "rollout_started",
        "rollout_ended",
        "thread_started",
        "thread_ended",
        "codex_turn_started",
        "codex_turn_ended",
        "protocol_event_observed",
    }
    keyed = {
        "inference_started": "inference_call_id",
        "inference_completed": "inference_call_id",
        "code_cell_started": "runtime_cell_id",
        "code_cell_initial_response": "runtime_cell_id",
        "code_cell_ended": "runtime_cell_id",
        "tool_call_started": "tool_call_id",
        "tool_call_runtime_started": "tool_call_id",
        "tool_call_runtime_ended": "tool_call_id",
        "tool_call_ended": "tool_call_id",
    }

    def payload(ref: Any, expected_kind: str) -> dict[str, Any]:
        value = _object(ref, "raw payload reference")
        identity, relative = value.get("raw_payload_id"), value.get("path")
        if not isinstance(identity, str) or not identity.startswith("raw_payload:"):
            raise ValueError("Invalid raw payload ID")
        suffix = identity[len("raw_payload:") :]
        if (
            not suffix.isascii()
            or not suffix.isdigit()
            or int(suffix) < 1
            or relative != f"payloads/{suffix}.json"
            or value.get("kind") != {"type": expected_kind}
        ):
            raise ValueError("Invalid raw payload path or kind")
        previous = references.setdefault(identity, (relative, expected_kind))
        if previous != (relative, expected_kind):
            raise ValueError("Raw payload reference was reused inconsistently")
        if identity not in payload_cache:
            payload_cache[identity] = _object(json.loads(read(relative)), "raw payload")
        return payload_cache[identity]

    reference_kinds = {
        "metadata_payload": "session_metadata",
        "event_payload": "protocol_event",
        "request_payload": "inference_request",
        "invocation_payload": "tool_invocation",
        "runtime_payload": "tool_runtime_event",
        "result_payload": "tool_result",
    }
    for number, row in enumerate(records, 1):
        if (
            row.get("schema_version") != 1
            or row.get("seq") != number
            or row.get("rollout_id") != session_id
            or row.get("thread_id") not in (None, session_id)
            or row.get("codex_turn_id") not in (None, turn_id)
        ):
            raise ValueError("Rollout identity or event sequence differs")
        event = _object(row.get("payload"), "rollout event")
        kind = event.get("type")
        if kind not in ordinary and kind not in keyed:
            raise ValueError("Unknown rollout event")
        if event.get("thread_id") not in (None, session_id) or event.get("codex_turn_id") not in (
            None,
            turn_id,
        ):
            raise ValueError("Event crosses native thread or turn")
        if kind in keyed:
            if row.get("thread_id") != session_id or row.get("codex_turn_id") != turn_id:
                raise ValueError("Operational event lacks native ownership")
            identity = event.get(keyed[kind])
            if not isinstance(identity, str) or not identity:
                raise ValueError("Operational event lacks an ID")
        else:
            identity = str(number) if kind == "protocol_event_observed" else "only"
        group = groups.setdefault(kind, {})
        if identity in group:
            raise ValueError("Duplicate native event ID")
        group[identity] = row
        for field, value in event.items():
            if field.endswith("_payload"):
                expected = reference_kinds.get(field)
                if field == "response_payload":
                    expected = (
                        "inference_response" if kind == "inference_completed" else "tool_result"
                    )
                if expected is None:
                    raise ValueError("Unknown payload reference")
                payload(value, expected)
    for kind in ordinary - {"protocol_event_observed"}:
        if len(groups.get(kind, {})) != 1:
            raise ValueError("Native rollout lifecycle is incomplete")
    for kind in ("rollout_ended", "thread_ended", "codex_turn_ended"):
        if groups[kind]["only"]["payload"].get("status") != "completed":
            raise ValueError("Native rollout did not complete")
    if (
        records[0]["payload"].get("type") != "rollout_started"
        or records[-1]["payload"].get("type") != "rollout_ended"
    ):
        raise ValueError("Native rollout endpoints differ")
    if groups["rollout_started"]["only"]["payload"].get("trace_id") != manifest.get("trace_id"):
        raise ValueError("Native trace ID differs")

    def pairs(start: str, end: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        left, right = groups.get(start, {}), groups.get(end, {})
        if set(left) != set(right):
            raise ValueError("Native event pair is incomplete")
        result = []
        for identity, first in left.items():
            last = right[identity]
            if first["seq"] >= last["seq"]:
                raise ValueError("Native event pair is out of order")
            result.append((identity, first, last))
        return result

    observed_calls: dict[str, str] = {}
    for _, start, end in pairs("inference_started", "inference_completed"):
        if start["payload"].get("model") != model:
            raise ValueError("Inference used a different model")
        response = payload(end["payload"].get("response_payload"), "inference_response")
        if response.get("response_id") != end["payload"].get("response_id"):
            raise ValueError("Inference response identity differs")
        items = response.get("output_items")
        if not isinstance(items, list):
            raise ValueError("Inference output is missing")
        for item in items:
            model_call(_object(item, "inference item"), observed_calls)
    if not model_calls or observed_calls != model_calls:
        raise ValueError("Rollout and session model calls differ")
    cells = {}
    for identity, start, end in pairs("code_cell_started", "code_cell_ended"):
        event = start["payload"]
        call_id = event.get("model_visible_call_id")
        if (
            call_id not in model_calls
            or event.get("source_js") != model_calls[call_id]
            or end["payload"].get("status") != "completed"
        ):
            raise ValueError("Code cell does not match its completed model call")
        cells[identity] = (call_id, start["seq"], end["seq"])
    if len(cells) != len(model_calls) or {v[0] for v in cells.values()} != set(model_calls):
        raise ValueError("Model calls and cells do not match one-to-one")
    for identity, initial in groups.get("code_cell_initial_response", {}).items():
        if identity not in cells or not cells[identity][1] < initial["seq"] < cells[identity][2]:
            raise ValueError("Code cell initial response has no active parent")
    pairs("tool_call_started", "tool_call_runtime_started")
    pairs("tool_call_runtime_started", "tool_call_runtime_ended")
    pairs("tool_call_runtime_ended", "tool_call_ended")
    calls = []
    local_ids: set[tuple[str, str]] = set()
    for identity, start, end in pairs("tool_call_started", "tool_call_ended"):
        event = start["payload"]
        requester = _object(event.get("requester"), "tool requester")
        cell_id = requester.get("runtime_cell_id")
        if requester.get("type") != "code_cell" or cell_id not in cells:
            raise ValueError("MCP call lacks its recorded parent cell")
        model_id, cell_start, cell_end = cells[cell_id]
        local_id = event.get("code_mode_runtime_tool_id")
        if not isinstance(local_id, str) or not local_id or (cell_id, local_id) in local_ids:
            raise ValueError("Missing or duplicate cell-local tool ID")
        local_ids.add((cell_id, local_id))
        if (
            event.get("model_visible_call_id") is not None
            or not cell_start < start["seq"] < end["seq"] < cell_end
        ):
            raise ValueError("MCP call does not lie within its parent cell")
        invocation = payload(event.get("invocation_payload"), "tool_invocation")
        function = _object(invocation.get("payload"), "function invocation")
        name = _tool_name(invocation.get("tool_name"))
        args = _arguments(function.get("arguments"))
        if invocation.get("tool_namespace") != "mcp__mnemos" or function.get("type") != "function":
            raise ValueError("Call is outside the Mnemos MCP namespace")
        runtime_end = None
        for kind in ("tool_call_runtime_started", "tool_call_runtime_ended"):
            record = groups[kind][identity]["payload"]
            runtime = payload(record.get("runtime_payload"), "tool_runtime_event")
            observed_id, observed_name, observed_args = _codex_invocation(runtime)
            if (observed_id, observed_name, observed_args, runtime.get("turn_id")) != (
                identity,
                name,
                args,
                turn_id,
            ):
                raise ValueError("Runtime invocation differs from recorded dispatch")
            if kind == "tool_call_runtime_ended":
                if record.get("status") != "completed":
                    raise ValueError("Native MCP runtime failed")
                runtime_end = runtime
        assert runtime_end is not None
        result = _codex_result(runtime_end)
        returned = payload(end["payload"].get("result_payload"), "tool_result")
        visible = _result(returned.get("value"))
        if (
            end["payload"].get("status") != "completed"
            or returned.get("type") != "code_mode_response"
            or visible.get("content") != runtime_end["result"]["Ok"]["content"]
        ):
            raise ValueError("Code Mode result differs from native MCP result")
        calls.append(
            NativeToolCall(
                identity,
                name,
                _frozen_object(args),
                _frozen_object(result),
                log_provenance,
                end["line"],
                end["line_sha256"],
                model_id,
                True,
            )
        )
    if not calls:
        raise ValueError("Rollout contains no completed native MCP calls")
    # Detect a concurrent edit while joining records. Keep every consumed raw
    # payload hash, including requests, for the private evidence receipt.
    if log_raw != raw_files["trace.jsonl"]:
        raise ValueError("Trace changed while loading")
    for relative, raw in raw_files.items():
        if (root / relative).read_bytes() != raw:
            raise ValueError("Trace changed while parsing")
    load_source_records(Path(provenance.source_file), provenance=provenance)
    return tuple(calls), hashes


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
