"""Synthetic, structured-trace validation for native Hermes lifecycle probes."""

from __future__ import annotations

import json
from typing import Any, cast

_TOOLS = {
    "mnemos_remember",
    "mnemos_retrieve",
    "mnemos_inspect",
    "mnemos_correct",
    "mnemos_forget",
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


def correlate_native_tools(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correlate only native Hermes tool dispatches and their structured results."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    pending: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    matched: list[dict[str, Any]] = []
    saw_response = False
    for record in records:
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


def evaluate_phase(
    records: list[dict[str, Any]],
    *,
    phase: str,
    target_id: str | None,
    old_content: str,
    new_content: str,
) -> dict[str, bool]:
    """Evaluate one lifecycle phase using only correlated structured results."""
    rows = correlate_native_tools(records)
    outcome: dict[str, bool] = {"correlated": True}
    fact_target = target_id or ""
    allowed = {
        "teach": {"mnemos_remember", "mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
        "correct": {"mnemos_correct", "mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
        "forget": {"mnemos_forget", "mnemos_retrieve", "mnemos_inspect", "mnemos_job_status"},
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
        if row["name"] in {"mnemos_correct", "mnemos_forget"}
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
        outcome["forget"] = any(
            row["name"] == "mnemos_forget"
            and row["args"].get("fact_id") == fact_target
            and row["args"].get("expected_version") == 2
            and _schema_result(row["result"])
            and row["result"].get("forgotten") is True
            and any(
                _fact(previous["result"], fact_target, 2)
                for previous in rows[: rows.index(row)]
                if previous["name"] == "mnemos_inspect"
            )
            for row in rows
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
