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


def _forget_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"Forget {label} are malformed")
    if len(value) != len(set(value)):
        raise ValueError(f"Forget {label} contain duplicates")
    return value


def _forget_result(result: Any, label: str = "preview") -> Mapping[str, Any]:
    if (
        not isinstance(result, Mapping)
        or not isinstance(result.get("schema_version"), int)
        or isinstance(result.get("schema_version"), bool)
        or result.get("schema_version") != 1
    ):
        raise ValueError(f"Forget {label} has an unknown result schema")
    return result


def _forget_fact(
    value: Any, fact_id: str, version: int, content: str | None = None
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("id") != fact_id:
        raise ValueError("Forget inspection returned the wrong fact")
    if (
        not isinstance(value.get("version"), int)
        or isinstance(value.get("version"), bool)
        or value.get("version") != version
    ):
        raise ValueError("Forget inspection returned the wrong fact version")
    if content is not None and value.get("content") != content:
        raise ValueError("Forget inspection content is stale")
    if not isinstance(value.get("content"), str):
        raise ValueError("Forget inspection has malformed content")
    return value


def _forget_inspected_before(
    calls: Sequence[Mapping[str, Any]],
    end: int,
    fact_id: str,
    version: int,
    content: str | None = None,
) -> None:
    for call in calls[:end]:
        if call.get("name") != "mnemos_inspect":
            continue
        args = call.get("args")
        result = call.get("result")
        if not isinstance(args, Mapping) or not isinstance(result, Mapping):
            continue
        inspected_id = args.get("fact_id", args.get("chunk_id"))
        if inspected_id != fact_id:
            continue
        try:
            schema = _forget_result(result)
            _forget_fact(schema.get("fact"), fact_id, version, content)
        except ValueError:
            continue
        return
    raise ValueError("Forget lacks prior inspection of the same fact and version")


def _forget_limit(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 500:
        raise ValueError(f"Forget {label} is malformed")
    return value


def _forget_version(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Forget {label} is malformed")
    return value


def _forget_digest(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("Forget preview digest is malformed")
    return value


def _reviewed_forget(
    calls: Sequence[Mapping[str, Any]],
    apply_index: int,
    write: Mapping[str, Any],
    fact_id: str,
) -> bool:
    args = write.get("args")
    result = write.get("result")
    if not isinstance(args, Mapping) or not isinstance(result, Mapping):
        raise ValueError("Forget mutation has malformed arguments or result")
    _forget_result(result, "apply")
    if (
        args.get("fact_id") != fact_id
        or _forget_version(args.get("expected_version"), "apply version") != 2
    ):
        raise ValueError("Forget did not select the corrected fact and version")
    _forget_inspected_before(calls, apply_index, fact_id, 2)

    selected_facts = _forget_ids(args.get("selected_fact_ids", []), "fact selections")
    selected_sources = _forget_ids(args.get("selected_source_event_ids", []), "source selections")
    requested_digest = _forget_digest(args.get("preview_digest"))
    requested_limit = _forget_limit(args.get("preview_limit", 100), "apply limit")

    matching: list[tuple[int, Mapping[str, Any]]] = []
    for preview_index, call in enumerate(calls[:apply_index]):
        if call.get("name") != "mnemos_forget_preview":
            continue
        preview_args = call.get("args")
        preview_value = call.get("result")
        if not isinstance(preview_args, Mapping) or not isinstance(preview_value, Mapping):
            continue
        if (
            preview_args.get("fact_id") == fact_id
            and preview_args.get("expected_version") == 2
            and preview_value.get("preview_digest") == requested_digest
        ):
            matching.append((preview_index, preview_value))
    if len(matching) != 1:
        raise ValueError("Forget lacks one matching complete native preview")
    preview_index, preview = matching[0]
    _forget_result(preview)
    if preview.get("status") != "preview" or preview.get("complete") is not True:
        raise ValueError("Forget lacks a matching complete native preview")

    preview_call = calls[preview_index]
    preview_args = preview_call.get("args")
    if not isinstance(preview_args, Mapping):
        raise ValueError("Forget preview arguments are malformed")
    if (
        preview_args.get("fact_id") != fact_id
        or _forget_version(preview_args.get("expected_version"), "preview version") != 2
    ):
        raise ValueError("Forget preview selected the wrong fact and version")
    preview_limit = _forget_limit(preview.get("limit"), "preview limit")
    if _forget_limit(preview_args.get("limit", 100), "preview argument limit") != preview_limit:
        raise ValueError("Forget preview limit does not match its request")
    if requested_limit != preview_limit:
        raise ValueError("Forget apply limit does not match the reviewed preview")
    if _forget_digest(preview.get("preview_digest")) != requested_digest:
        raise ValueError("Forget digest does not match the reviewed preview")

    target = preview.get("target")
    if not isinstance(target, Mapping) or target.get("fact_id") != fact_id:
        raise ValueError("Forget preview target is malformed")
    target_version = target.get("version")
    target_content = target.get("content")
    if (
        not isinstance(target_version, int)
        or isinstance(target_version, bool)
        or target_version != 2
        or not isinstance(target_content, str)
        or not target_content
    ):
        raise ValueError("Forget preview target is malformed")
    _forget_inspected_before(calls, preview_index, fact_id, 2, target_content)
    _forget_inspected_before(calls, apply_index, fact_id, 2, target_content)

    candidates_value = preview.get("candidates")
    sources_value = preview.get("source_events")
    if not isinstance(candidates_value, list) or not isinstance(sources_value, list):
        raise ValueError("Forget preview candidate closure is malformed")
    candidates: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates_value:
        if (
            not isinstance(candidate, Mapping)
            or not isinstance(candidate.get("fact_id"), str)
            or not candidate.get("fact_id", "").strip()
        ):
            raise ValueError("Forget preview candidate is malformed")
        candidate_id = candidate["fact_id"]
        if candidate_id in candidates:
            raise ValueError("Forget preview candidate IDs are malformed")
        if candidate_id == fact_id:
            if (
                candidate.get("version") != 2
                or isinstance(candidate.get("version"), bool)
                or not isinstance(candidate.get("version"), int)
                or candidate.get("content") != target_content
                or not candidate.get("content")
            ):
                raise ValueError("Forget preview target candidate is inconsistent")
            _forget_ids(candidate.get("required_source_event_ids"), "required source IDs")
            if _forget_ids(candidate.get("blocked_source_event_ids"), "blocked source IDs"):
                raise ValueError("Forget preview target candidate has blocked sources")
        candidates[candidate_id] = candidate
    sources: dict[str, Mapping[str, Any]] = {}
    for source in sources_value:
        if (
            not isinstance(source, Mapping)
            or not isinstance(source.get("source_event_id"), str)
            or not source.get("source_event_id", "").strip()
        ):
            raise ValueError("Forget preview source closure is malformed")
        source_id = source["source_event_id"]
        if source_id in sources:
            raise ValueError("Forget preview source IDs are malformed")
        sources[source_id] = source

    selected_extra_ids = set(selected_facts) - {fact_id}
    if not selected_extra_ids.issubset(candidates):
        raise ValueError("Forget selected IDs that were not in the reviewed preview")
    if not set(selected_sources).issubset(sources):
        raise ValueError("Forget selected IDs that were not in the reviewed preview")

    selected_fact_set = selected_extra_ids | {fact_id}
    for candidate_id in selected_extra_ids:
        candidate = candidates[candidate_id]
        candidate_version = candidate.get("version")
        candidate_content = candidate.get("content")
        if (
            isinstance(candidate_version, bool)
            or not isinstance(candidate_version, int)
            or candidate_version < 1
            or not isinstance(candidate_content, str)
            or not candidate_content
        ):
            raise ValueError("Forget selected candidate is malformed")
        required_sources = _forget_ids(
            candidate.get("required_source_event_ids"), "required source IDs"
        )
        blocked_sources = _forget_ids(
            candidate.get("blocked_source_event_ids"), "blocked source IDs"
        )
        if blocked_sources:
            raise ValueError("Forget selected candidate has blocked sources")
        if not set(required_sources).issubset(selected_sources):
            raise ValueError("Forget selected candidate is missing a required source")
        _forget_inspected_before(
            calls, apply_index, candidate_id, candidate_version, candidate_content
        )

    for source_id in selected_sources:
        source = sources[source_id]
        if source.get("selection_allowed") is not True:
            raise ValueError("Forget selected a blocked source")
        derived = _forget_ids(source.get("derived_fact_ids"), "derived fact IDs")
        if not set(derived).issubset(selected_fact_set):
            raise ValueError("Forget source closure includes an unselected fact")

    result_facts = _forget_ids(result.get("selected_fact_ids"), "result fact IDs")
    result_sources = _forget_ids(result.get("selected_source_event_ids"), "result source IDs")
    if set(result_facts) != selected_fact_set or set(result_sources) != set(selected_sources):
        raise ValueError("Forget result differs from the reviewed selections")
    if (
        result.get("forgotten") is not True
        or result.get("fact_id") != fact_id
        or result.get("status") != "committed"
    ):
        raise ValueError("Reviewed native forget did not complete")
    return True


def require_forget(calls: Sequence[Mapping[str, Any]], fact_id: str) -> None:
    require_native_calls(
        calls,
        {"mnemos_retrieve", "mnemos_inspect", "mnemos_forget_preview", "mnemos_forget_related"},
    )
    writes = [(i, call) for i, call in enumerate(calls) if call["name"] == "mnemos_forget_related"]
    if len(writes) != 1:
        raise ValueError("Expected one reviewed related-forget operation")
    index, write = writes[0]
    _reviewed_forget(calls, index, write, fact_id)


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
