"""Acceptance rules for the prospective three-host synthetic lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence, cast


@dataclass(frozen=True)
class Scenario:
    old: str = "The project release label is cedar-ridge-731."
    new: str = "The project release label is maple-ridge-842."
    retained: str = "The project review label is amber-grove-264."
    unrelated: str = "The build artifact checksum marker is slate-safe-553."
    foreign: str = "The project release label is forbidden-foreign-ridge-992."
    target_query: str = "current asserted project release label"
    retained_query: str = "current asserted project review label"

    @property
    def forgotten_labels(self) -> tuple[str, str]:
        return ("cedar-ridge-731", "maple-ridge-842")


def asserted_fact(state: Mapping[str, Any], content: str, scope_id: str) -> dict[str, Any]:
    matches = []
    for fact in state["facts"]:
        metadata = fact["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if (
            fact["content"] == content
            and metadata.get("assertion_status") == "asserted"
            and metadata.get("scope_id") == scope_id
        ):
            matches.append(fact)
    if len(matches) != 1:
        raise ValueError("Expected exactly one asserted fact in the target project")
    return dict(matches[0])


def require_native_calls(calls: Sequence[Mapping[str, Any]], allowed: set[str]) -> None:
    if not calls:
        raise ValueError("No model-correlated native tools were recorded")
    if any(call.get("model_call_correlated") is not True for call in calls):
        raise ValueError("Native call lacks model correlation")
    if any(call["name"] not in allowed for call in calls):
        raise ValueError("Native phase issued an unapproved tool")


def require_job_completion(calls: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]) -> None:
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("Mutation receipt has no durable job ID")
    if not any(
        call["name"] == "mnemos_job_status"
        and call["args"].get("job_id") == job_id
        and isinstance(call["result"].get("job"), dict)
        and call["result"]["job"].get("id") == job_id
        and call["result"]["job"].get("status") == "completed"
        for call in calls
    ):
        raise ValueError("Native status call did not establish completion of the actual job")


def require_teach(calls: Sequence[Mapping[str, Any]], content: str) -> Mapping[str, Any]:
    require_native_calls(calls, {"mnemos_store", "mnemos_consolidate", "mnemos_job_status"})
    writes = [(i, call) for i, call in enumerate(calls) if call["name"] == "mnemos_store"]
    if len(writes) != 1 or writes[0][1]["args"].get("content") != content:
        raise ValueError("Teaching must issue exactly one explicit write with the selected content")
    index, write = writes[0]
    receipt = write["result"]
    if receipt.get("status") != "committed" or not receipt.get("event_id"):
        raise ValueError("Teaching did not commit its original source")
    require_job_completion(calls[index + 1 :], receipt)
    return cast(Mapping[str, Any], receipt)


def require_recalled(
    calls: Sequence[Mapping[str, Any]],
    *,
    fact_id: str,
    content: str,
    version: int,
    scope_id: str,
    query: str,
) -> None:
    if not any(
        call["name"] == "mnemos_retrieve"
        and call["args"].get("query") == query
        and any(
            fact.get("id") == fact_id
            and fact.get("content") == content
            and fact.get("version") == version
            and fact.get("scope") == "project"
            and fact.get("scope_id") == scope_id
            and fact.get("assertion_status") == "asserted"
            for fact in call["result"].get("facts", [])
        )
        for call in calls
    ):
        raise ValueError("Native recall did not return the exact scoped asserted fact and version")


def require_correct(calls: Sequence[Mapping[str, Any]], fact_id: str, replacement: str) -> None:
    require_native_calls(
        calls, {"mnemos_retrieve", "mnemos_inspect", "mnemos_correct", "mnemos_job_status"}
    )
    corrections = [(i, call) for i, call in enumerate(calls) if call["name"] == "mnemos_correct"]
    if len(corrections) != 1:
        raise ValueError("Expected exactly one native correction")
    index, correction = corrections[0]
    args = correction["args"]
    if (
        args.get("fact_id") != fact_id
        or args.get("expected_version") != 1
        or args.get("content") != replacement
    ):
        raise ValueError("Correction did not select the original fact at version one")
    _require_inspection(calls[:index], fact_id, 1)
    if correction["result"].get("status") != "committed":
        raise ValueError("Native correction was not committed")
    require_job_completion(calls[index + 1 :], correction["result"])


def _require_inspection(calls: Sequence[Mapping[str, Any]], fact_id: str, version: int) -> None:
    if not any(
        call["name"] == "mnemos_inspect"
        and call["args"].get("fact_id", call["args"].get("chunk_id")) == fact_id
        and isinstance(call["result"].get("fact"), dict)
        and call["result"]["fact"].get("id") == fact_id
        and call["result"]["fact"].get("version") == version
        for call in calls
    ):
        raise ValueError("Mutation lacks prior inspection of the same fact and version")


def require_forget(calls: Sequence[Mapping[str, Any]], fact_id: str) -> None:
    require_native_calls(
        calls,
        {"mnemos_retrieve", "mnemos_inspect", "mnemos_forget_preview", "mnemos_forget_related"},
    )
    writes = [(i, call) for i, call in enumerate(calls) if call["name"] == "mnemos_forget_related"]
    if len(writes) != 1:
        raise ValueError("Expected one reviewed related-forget operation")
    index, write = writes[0]
    _require_inspection(calls[:index], fact_id, 2)
    args = write["args"]
    if args.get("fact_id") != fact_id or args.get("expected_version") != 2:
        raise ValueError("Forget did not select the corrected fact and version")
    previews = [
        call
        for call in calls[:index]
        if call["name"] == "mnemos_forget_preview"
        and call["args"].get("fact_id") == fact_id
        and call["args"].get("expected_version") == 2
        and call["result"].get("preview_digest") == args.get("preview_digest")
    ]
    if len(previews) != 1 or previews[0]["result"].get("complete") is not True:
        raise ValueError("Forget lacks a matching complete native preview")
    preview = previews[0]["result"]
    selected_facts = args.get("selected_fact_ids") or []
    selected_sources = args.get("selected_source_event_ids") or []
    if any(
        not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        or len(values) != len(set(values))
        for values in (selected_facts, selected_sources)
    ):
        raise ValueError("Forget selections are malformed")
    candidates = {row["fact_id"] for row in preview.get("candidates", [])} | {fact_id}
    sources = {row["source_event_id"] for row in preview.get("source_events", [])}
    if not set(selected_facts).issubset(candidates) or not set(selected_sources).issubset(sources):
        raise ValueError("Forget selected IDs that were not in the reviewed preview")
    if set(write["result"].get("selected_fact_ids", [])) != set(selected_facts) | {fact_id} or set(
        write["result"].get("selected_source_event_ids", [])
    ) != set(selected_sources):
        raise ValueError("Forget result differs from the reviewed selections")
    if (
        write["result"].get("forgotten") is not True
        or write["result"].get("fact_id") != fact_id
        or write["result"].get("status") != "committed"
    ):
        raise ValueError("Reviewed native forget did not complete")


def require_absent(state: Mapping[str, Any], fact_id: str, labels: Sequence[str]) -> None:
    if any(row["id"] == fact_id for row in state["facts"]):
        raise ValueError("Forgotten target still exists")
    if not any(row["fact_id"] == fact_id for row in state["suppressions"]):
        raise ValueError("Forgotten target has no suppression record")
    # Suppression keys deliberately remain; retained content must not.
    for table in ("facts", "events", "revisions", "evidence", "jobs"):
        if any(label in json.dumps(state[table], ensure_ascii=False) for label in labels):
            raise ValueError("Forgotten content remains in " + table)
    known_ids = {row["id"] for row in state["facts"]}
    if any(
        edge["source_id"] not in known_ids or edge["target_id"] not in known_ids
        for edge in state["edges"]
    ):
        raise ValueError("Derived graph state still refers to a removed fact")
    if any(job["status"] != "completed" for job in state["jobs"]):
        raise ValueError("Unfinished jobs remain after forgetting or replay")
