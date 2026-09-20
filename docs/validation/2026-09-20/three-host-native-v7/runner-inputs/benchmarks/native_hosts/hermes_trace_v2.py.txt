"""Synthetic, structured-trace validation for native Hermes lifecycle probes."""

from __future__ import annotations

import json
import re
from typing import Any, cast

_TOOLS = {
    "mnemos_remember",
    "mnemos_retrieve",
    "mnemos_inspect",
    "mnemos_correct",
    "mnemos_forget",
    "mnemos_forget_preview",
    "mnemos_forget_related",
    "mnemos_job_status",
}
_BAD_STATUSES = {"rejected", "degraded", "error", "failed", "failure"}
_VALID_STATUSES = {"committed", "ok", "pending", "consolidated", "suppressed", "preview"}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _args(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
    return _object(value, "tool arguments")


def _canonical(value: dict[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("tool arguments are not canonical JSON") from exc


def _validate_result(result: Any) -> dict[str, Any]:
    result = _object(result, "tool result")
    status = result.get("status")
    if "status" in result:
        if not isinstance(status, str) or status.lower() in _BAD_STATUSES:
            raise ValueError(f"tool result has unsuccessful status: {status}")
        if status.lower() not in _VALID_STATUSES:
            raise ValueError(f"tool result has malformed status: {status}")
    error = result.get("error")
    if error:
        raise ValueError("tool result contains an error")
    return cast(dict[str, Any], result)


def correlate_native_tools(
    records: list[dict[str, Any]], *, include_positions: bool = False
) -> list[dict[str, Any]]:
    """Correlate only native Hermes tool dispatches and their structured results."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    pending: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    matched: list[dict[str, Any]] = []
    saw_response = False
    for trace_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("trace record must be an object")
        event_type = record.get("operation")
        if event_type == "model_response":
            saw_response = True
            calls = record.get("tool_calls", [])
            if not isinstance(calls, list):
                raise ValueError("model_response tool_calls must be a list")
            for call in calls:
                call = _object(call, "tool call")
                call_id = call.get("id")
                name = call.get("name")
                if not isinstance(call_id, str) or not call_id.strip() or call_id in seen_ids:
                    raise ValueError("tool call IDs must be globally unique and nonempty")
                if name not in _TOOLS:
                    raise ValueError(f"unsupported native tool: {name}")
                arguments = _args(call.get("arguments"))
                seen_ids.add(call_id)
                pending.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": arguments,
                        "key": _canonical(arguments),
                        "call_trace_index": trace_index,
                    }
                )
        elif event_type == "tool_result":
            name = record.get("name")
            if name not in _TOOLS:
                raise ValueError("tool result names must identify allowed native tools")
            arguments = _args(record.get("args"))
            result = _validate_result(record.get("result"))
            key = _canonical(arguments)
            index = next(
                (
                    i
                    for i, call in enumerate(pending)
                    if call["name"] == name and call["key"] == key
                ),
                None,
            )
            if index is None:
                raise ValueError("tool result has no preceding unmatched matching call")
            call = pending.pop(index)
            matched.append(
                {"call_id": call["call_id"], "name": name, "args": arguments, "result": result}
            )
            if include_positions:
                matched[-1].update(
                    call_trace_index=call["call_trace_index"], result_trace_index=trace_index
                )
    if not saw_response:
        raise ValueError("trace has no model response")
    if not matched:
        raise ValueError("trace has no correlated native tool")
    if pending:
        raise ValueError("model tool calls remain undispatched")
    return matched


def _schema_result(result: dict[str, Any]) -> bool:
    return result.get("schema_version") == 1 and not isinstance(result.get("schema_version"), bool)


def _fact(result: dict[str, Any], target_id: str, version: int, content: str | None = None) -> bool:
    fact = result.get("fact")
    if not _schema_result(result) or not isinstance(fact, dict):
        return False
    if (
        not isinstance(fact.get("id"), str)
        or not fact["id"]
        or not isinstance(fact.get("version"), int)
        or isinstance(fact["version"], bool)
        or fact["version"] <= 0
        or not isinstance(fact.get("content"), str)
        or fact.get("id") != target_id
        or fact.get("version") != version
    ):
        return False
    return content is None or fact.get("content") == content


def _successful(result: dict[str, Any]) -> bool:
    return result.get("status") == "committed"


def _successful_retrieve(result: dict[str, Any]) -> bool:
    return ("status" not in result) or result.get("status") in {"ok", "committed"}


def _facts(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not _schema_result(result) or not isinstance(result.get("facts"), list):
        return None
    facts = result["facts"]
    if not all(
        isinstance(fact, dict)
        and isinstance(fact.get("id"), str)
        and bool(fact["id"])
        and isinstance(fact.get("content"), str)
        and bool(fact["content"])
        and isinstance(fact.get("version"), int)
        and not isinstance(fact["version"], bool)
        and fact["version"] > 0
        for fact in facts
    ):
        return None
    return cast(list[dict[str, Any]], facts)


def _ids(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return None
    if len(set(value)) != len(value):
        return None
    return cast(list[str], value)


def _inspected_before(
    rows: list[dict[str, Any]],
    call_row: dict[str, Any],
    fact_id: str,
    version: int,
    content: str | None = None,
) -> bool:
    return any(
        row["name"] == "mnemos_inspect"
        and row["result_trace_index"] < call_row["call_trace_index"]
        and row["args"].get("fact_id") == fact_id
        and _fact(row["result"], fact_id, version, content)
        for row in rows
    )


def _reviewed_apply(rows: list[dict[str, Any]], applied: dict[str, Any]) -> bool:
    args, result = applied["args"], applied["result"]
    target_id, version = args.get("fact_id"), args.get("expected_version")
    selected = _ids(args.get("selected_fact_ids", []))
    sources = _ids(args.get("selected_source_event_ids", []))
    if (
        not isinstance(target_id, str)
        or not target_id
        or type(version) is not int
        or version < 1
        or selected is None
        or sources is None
        or not _schema_result(result)
        or not _successful(result)
        or result.get("forgotten") is not True
        or result.get("fact_id") != target_id
        or _ids(result.get("selected_fact_ids")) is None
        or set(result["selected_fact_ids"]) != set(selected) | {target_id}
        or _ids(result.get("selected_source_event_ids")) is None
        or set(result["selected_source_event_ids"]) != set(sources)
        or not _inspected_before(rows, applied, target_id, version)
    ):
        return False
    for previous in rows:
        if (
            previous["name"] != "mnemos_forget_preview"
            or previous["result_trace_index"] >= applied["call_trace_index"]
        ):
            continue
        preview, preview_args = previous["result"], previous["args"]
        target = preview.get("target", {})
        digest = preview.get("preview_digest")
        limit = preview.get("limit")
        if (
            not _schema_result(preview)
            or preview.get("status") != "preview"
            or preview.get("complete") is not True
            or not isinstance(target, dict)
            or not isinstance(target.get("content"), str)
            or not target["content"]
            or target.get("fact_id") != target_id
            or target.get("version") != version
            or type(target.get("version")) is not int
            or preview_args.get("fact_id") != target_id
            or preview_args.get("expected_version") != version
            or type(preview_args.get("expected_version")) is not int
            or type(limit) is not int
            or not 1 <= limit <= 500
            or preview_args.get("limit", 100) != limit
            or type(preview_args.get("limit", 100)) is not int
            or args.get("preview_limit", 100) != limit
            or type(args.get("preview_limit", 100)) is not int
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
            or args.get("preview_digest") != digest
            or not _inspected_before(rows, previous, target_id, version, target.get("content"))
        ):
            continue
        candidates = preview.get("candidates")
        source_events = preview.get("source_events")
        if not isinstance(candidates, list) or not isinstance(source_events, list):
            continue
        if not all(isinstance(item, dict) for item in candidates + source_events):
            continue
        candidate_ids = _ids([item.get("fact_id") for item in candidates])
        source_ids = _ids([item.get("source_event_id") for item in source_events])
        if candidate_ids is None or source_ids is None:
            continue
        selected_extra = set(selected) - {target_id}
        if not selected_extra <= set(candidate_ids) or not set(sources) <= set(source_ids):
            continue
        candidate_map = {item["fact_id"]: item for item in candidates}
        source_map = {item["source_event_id"]: item for item in source_events}
        valid = True
        for fact_id in selected_extra:
            candidate = candidate_map[fact_id]
            required = _ids(candidate.get("required_source_event_ids"))
            if (
                type(candidate.get("version")) is not int
                or candidate["version"] < 1
                or not isinstance(candidate.get("content"), str)
                or not candidate["content"]
                or required is None
                or not set(required) <= set(sources)
                or candidate.get("blocked_source_event_ids") != []
                or not _inspected_before(
                    rows, applied, fact_id, candidate["version"], candidate["content"]
                )
            ):
                valid = False
        for source_id in sources:
            source = source_map[source_id]
            derived = _ids(source.get("derived_fact_ids"))
            if (
                source.get("selection_allowed") is not True
                or derived is None
                or not set(derived) <= selected_extra | {target_id}
            ):
                valid = False
        if valid:
            return True
    return False


def reviewed_deletions_persisted(
    records: list[dict[str, Any]], state: dict[str, list[dict[str, Any]]]
) -> bool:
    """Check every reviewed deletion against persisted facts, sources, and tombstones."""
    rows = correlate_native_tools(records, include_positions=True)
    applied = [row for row in rows if row["name"] == "mnemos_forget_related"]
    if not applied or not all(_reviewed_apply(rows, row) for row in applied):
        return False
    facts = {fact_id for row in applied for fact_id in row["result"]["selected_fact_ids"]}
    sources = {
        source_id for row in applied for source_id in row["result"]["selected_source_event_ids"]
    }
    if not all(
        name in state for name in ("facts", "events", "evidence", "revisions", "suppressions")
    ):
        return False
    return (
        all(row.get("id") not in facts for row in state["facts"])
        and all(row.get("id") not in sources for row in state["events"])
        and all(
            row.get("fact_id") not in facts and row.get("source_event_id") not in sources
            for row in state["evidence"]
        )
        and all(row.get("fact_id") not in facts for row in state["revisions"])
        and facts <= {row.get("fact_id") for row in state["suppressions"]}
        and sources <= {row.get("source_event_id") for row in state["suppressions"]}
    )


def no_retained_labels(
    state: dict[str, list[dict[str, Any]]],
    *,
    context: dict[str, Any],
    labels: list[str],
    known_fact_ids: set[str] | None = None,
) -> bool:
    """Check scoped persisted fixture records, not only bounded retrieval output."""
    if not context.get("project_id") or not context.get("user_profile_id") or not labels:
        return False
    if not all(name in state for name in ("facts", "events", "evidence", "revisions", "jobs")):
        return False

    def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
        value = row.get(key, {})
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError(f"Persisted {key} is not an object")
        return value

    def in_scope(metadata: dict[str, Any]) -> bool:
        return bool(
            metadata.get("user_profile_id") == context["user_profile_id"]
            and metadata.get("scope", "project") == "project"
            and metadata.get("scope_id", metadata.get("project_id")) == context["project_id"]
        )

    pattern = re.compile(
        r"(?<![a-z0-9_-])(?:"
        + "|".join(re.escape(label.casefold()) for label in labels)
        + r")(?![a-z0-9_-])"
    )

    def absent(row: dict[str, Any]) -> bool:
        serialized = json.dumps(row, ensure_ascii=False).casefold()
        return pattern.search(serialized) is None

    scoped_facts = [row for row in state["facts"] if in_scope(object_field(row, "metadata"))]
    fact_ids = {row["id"] for row in scoped_facts} | (known_fact_ids or set())
    checked = list(scoped_facts)
    checked.extend(row for row in state["events"] if in_scope(object_field(row, "metadata")))
    checked.extend(
        row
        for name in ("evidence", "revisions")
        for row in state[name]
        if row.get("fact_id") in fact_ids
    )
    for row in state["jobs"]:
        payload = object_field(row, "payload")
        metadata = object_field(payload, "context")
        if in_scope(metadata) or in_scope(object_field(payload, "metadata")):
            checked.append(row)
    return all(absent(row) for row in checked)


def evaluate_phase(
    records: list[dict[str, Any]],
    *,
    phase: str,
    target_id: str | None,
    old_content: str,
    new_content: str,
) -> dict[str, bool]:
    """Evaluate one lifecycle phase using only correlated structured results."""
    rows = correlate_native_tools(records, include_positions=True)
    outcome: dict[str, bool] = {"correlated": True}
    fact_target = target_id or ""
    allowed = {
        "teach": {"mnemos_remember", "mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
        "correct": {"mnemos_correct", "mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
        "forget": {
            "mnemos_forget",
            "mnemos_forget_preview",
            "mnemos_forget_related",
            "mnemos_retrieve",
            "mnemos_inspect",
            "mnemos_job_status",
        },
        "recall_v1": {"mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
        "recall_v2": {"mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
        "recall_forgotten": {"mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
    }
    outcome["phase_tools_allowed"] = phase in allowed and all(
        row["name"] in allowed[phase] for row in rows
    )
    outcome["mutations_inspected"] = all(
        any(
            previous["name"] == "mnemos_inspect"
            and previous["args"].get("fact_id") == row["args"].get("fact_id")
            and _fact(
                previous["result"],
                row["args"].get("fact_id", ""),
                row["args"].get("expected_version", 0),
            )
            for previous in rows[:index]
        )
        for index, row in enumerate(rows)
        if row["name"] in {"mnemos_correct", "mnemos_forget", "mnemos_forget_related"}
    )
    if phase == "teach":
        outcome["teach"] = any(
            row["name"] == "mnemos_remember"
            and row["args"].get("content") == old_content
            and _schema_result(row["result"])
            and _successful(row["result"])
            and isinstance(row["result"].get("event_id"), str)
            and bool(row["result"].get("event_id"))
            and isinstance(row["result"].get("job_id"), str)
            and bool(row["result"].get("job_id"))
            and any(
                status["name"] == "mnemos_job_status"
                and status["args"].get("job_id") == row["result"].get("job_id")
                and _schema_result(status["result"])
                and isinstance(status["result"].get("job"), dict)
                and status["result"]["job"].get("id") == row["result"].get("job_id")
                and status["result"]["job"].get("status") in {"pending", "leased", "completed"}
                for status in rows[index + 1 :]
            )
            for index, row in enumerate(rows)
        )
    elif phase in {"recall_v1", "recall_v2"}:
        expected_version = 1 if phase.endswith("v1") else 2
        outcome[phase] = any(
            row["name"] == "mnemos_retrieve"
            and _successful_retrieve(row["result"])
            and any(
                fact.get("id") == fact_target
                and fact.get("version") == expected_version
                and fact.get("content") == (old_content if expected_version == 1 else new_content)
                for fact in (_facts(row["result"]) or [])
            )
            for row in rows
        )
    elif phase == "correct":
        outcome["correct"] = any(
            row["name"] == "mnemos_correct"
            and row["args"].get("fact_id") == fact_target
            and row["args"].get("expected_version") == 1
            and row["args"].get("content") == new_content
            and _schema_result(row["result"])
            and _successful(row["result"])
            and bool(row["result"].get("event_id"))
            and bool(row["result"].get("job_id"))
            and any(
                _fact(previous["result"], fact_target, 1)
                for previous in rows[: rows.index(row)]
                if previous["name"] == "mnemos_inspect"
            )
            for row in rows
        )
    elif phase == "forget":
        applied = [row for row in rows if row["name"] == "mnemos_forget_related"]
        outcome["reviewed_forget_verified"] = bool(applied) and all(
            _reviewed_apply(rows, row) for row in applied
        )
        outcome["forget"] = any(
            row["args"].get("fact_id") == fact_target
            and row["args"].get("expected_version") == 2
            and _reviewed_apply(rows, row)
            for row in applied
        )
    elif phase == "recall_forgotten":
        outcome["recall_forgotten"] = any(
            row["name"] == "mnemos_retrieve"
            and _successful_retrieve(row["result"])
            and _facts(row["result"]) is not None
            and all(
                not (
                    fact.get("id") == fact_target
                    or fact.get("content") in {old_content, new_content}
                )
                for fact in (_facts(row["result"]) or [])
            )
            for row in rows
        )
    else:
        raise ValueError(f"unknown lifecycle phase: {phase}")
    return outcome
