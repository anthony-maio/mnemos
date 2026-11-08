"""Acceptance rules for the prospective three-host synthetic lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import re
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


def _review_integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Forget review has a malformed fragment position")
    return value


def _review_snapshot(fragments: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Independently reconstruct native evidence, rejecting gaps and overwrites."""
    root: dict[str, Any] = {}
    written: set[tuple[str | int, ...]] = set()
    ancestors: set[tuple[str | int, ...]] = set()
    partial: tuple[tuple[str | int, ...], int, str] | None = None

    def assign(path: tuple[str | int, ...], value: Any) -> None:
        nonlocal root
        if (
            path in written
            or path in ancestors
            or any(path[:i] in written for i in range(len(path)))
        ):
            raise ValueError("Forget review has overlapping fragment paths")
        written.add(path)
        ancestors.update(path[:i] for i in range(len(path)))
        if not path:
            if not isinstance(value, dict):
                raise ValueError("Forget review root is not an object")
            root = deepcopy(value)
            return
        node: Any = root
        for index, key in enumerate(path):
            final = index == len(path) - 1
            child: Any = deepcopy(value) if final else ([] if type(path[index + 1]) is int else {})
            if isinstance(node, dict) and isinstance(key, str):
                if final:
                    if key in node:
                        raise ValueError("Forget review overwrites a field")
                    node[key] = child
                else:
                    node.setdefault(key, child)
                node = node[key]
            elif isinstance(node, list) and type(key) is int and 0 <= key <= len(node):
                if key == len(node):
                    node.append(child)
                elif final:
                    raise ValueError("Forget review overwrites an array item")
                node = node[key]
            else:
                raise ValueError("Forget review path has a gap or incompatible container")

    for fragment in fragments:
        raw_path = fragment.get("path")
        if not isinstance(raw_path, list) or any(
            not isinstance(key, str) and not (type(key) is int and key >= 0) for key in raw_path
        ):
            raise ValueError("Forget review fragment path is malformed")
        path = tuple(raw_path)
        if set(fragment) == {"path", "value"}:
            if partial is not None:
                raise ValueError("Forget review interrupted an unfinished text field")
            assign(path, fragment["value"])
            continue
        if set(fragment) != {"path", "text", "offset", "total_characters"}:
            raise ValueError("Forget review fragment shape is malformed")
        offset = _review_integer(fragment["offset"])
        total = _review_integer(fragment["total_characters"])
        text = fragment["text"]
        if not isinstance(text, str) or not text or total == 0:
            raise ValueError("Forget review text fragment is malformed")
        prior = ""
        if partial is not None:
            if partial[:2] != (path, total):
                raise ValueError("Forget review changed an unfinished text field")
            prior = partial[2]
        if offset != len(prior) or offset + len(text) > total:
            raise ValueError("Forget review text has a gap or overlap")
        joined = prior + text
        if len(joined) == total:
            assign(path, joined)
            partial = None
        else:
            partial = (path, total, joined)
    if partial is not None:
        raise ValueError("Forget review text is incomplete")
    return root


def _complete_native_review(
    matching: list[tuple[int, Mapping[str, Any]]], calls: Sequence[Mapping[str, Any]]
) -> tuple[int, Mapping[str, Any]]:
    if len(matching) == 1 and "review" not in matching[0][1]:
        return matching[0]
    if not matching:
        raise ValueError("Forget lacks one matching complete native preview")
    fragments: list[Mapping[str, Any]] = []
    snapshot_digest: str | None = None
    total: int | None = None
    expected_cursor: str | None = None
    first_args = calls[matching[0][0]]["args"]
    limit = _forget_limit(first_args.get("limit", 100), "preview argument limit")
    digest = _forget_digest(matching[0][1].get("preview_digest"))
    for number, (call_index, page) in enumerate(matching):
        _forget_result(page)
        args = calls[call_index]["args"]
        review = page.get("review")
        if (
            not isinstance(review, Mapping)
            or review.get("format") != "json_fragments_v1"
            or page.get("status") != "preview"
            or page.get("complete") is not True
            or page.get("preview_digest") != digest
            or args.get("cursor") != expected_cursor
            or args.get("fact_id") != first_args.get("fact_id")
            or _forget_version(args.get("expected_version"), "page version")
            != first_args.get("expected_version")
            or _forget_limit(args.get("limit", 100), "page argument limit") != limit
        ):
            raise ValueError("Forget review page is unbound or inconsistent")
        page_digest = _forget_digest(review.get("snapshot_digest"))
        page_total = _review_integer(review.get("total_fragments"))
        if snapshot_digest is None:
            snapshot_digest, total = page_digest, page_total
        if page_digest != snapshot_digest or page_total != total or page_total == 0:
            raise ValueError("Forget review changed its snapshot or fragment count")
        if _review_integer(review.get("first_fragment")) != len(fragments):
            raise ValueError("Forget review page sequence has a gap or overlap")
        values = page.get("fragments")
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, Mapping) for v in values)
        ):
            raise ValueError("Forget review has no valid fragments")
        fragments.extend(values)
        if len(fragments) > page_total:
            raise ValueError("Forget review has extra fragments")
        last = len(fragments) == page_total
        if review.get("last_page") is not last:
            raise ValueError("Forget review last-page marker is inconsistent")
        expected_cursor = None if last else f"{snapshot_digest}:{len(fragments)}"
        if review.get("next_cursor") != expected_cursor or last != (number == len(matching) - 1):
            raise ValueError("Forget review is incomplete or has trailing pages")
        if not last and (
            page.get("apply_preconditions") is not None
            or any(v.get("path", [])[:1] == ["apply_preconditions"] for v in values)
        ):
            raise ValueError("Forget review exposes apply preconditions before the last page")
    if total != len(fragments) or expected_cursor is not None:
        raise ValueError("Forget review is incomplete")
    snapshot = _review_snapshot(fragments)
    encoded = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != snapshot_digest:
        raise ValueError("Forget review snapshot digest does not match its fragments")
    preconditions = {
        "fact_id": first_args["fact_id"],
        "expected_version": first_args["expected_version"],
        "preview_digest": digest,
        "preview_limit": limit,
    }
    if (
        snapshot.get("apply_preconditions") != preconditions
        or matching[-1][1].get("apply_preconditions") != preconditions
    ):
        raise ValueError("Forget review apply preconditions do not match its snapshot")
    return matching[0][0], snapshot


def _review_selection_refs(
    raw: Any, ids: list[str], records: list[Any], prefix: str, id_key: str
) -> list[str]:
    refs = [] if raw is None else _forget_ids(raw, "short selections")
    if not refs:
        return ids
    if ids:
        raise ValueError("Forget mixed short references and IDs")
    resolved = []
    for ref in refs:
        if re.fullmatch(prefix + r"[1-9][0-9]{0,2}", ref, re.ASCII) is None:
            raise ValueError("Forget short reference is malformed")
        index = int(ref[1:]) - 1
        if index >= len(records) or records[index].get("selection_ref") != ref:
            raise ValueError("Forget short reference was not in the reviewed snapshot")
        resolved.append(records[index][id_key])
    return _forget_ids(resolved, "resolved short selections")


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
    preview_index, preview = _complete_native_review(matching, calls)
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

    selected_facts = _review_selection_refs(
        args.get("selected_fact_refs"), selected_facts, candidates_value, "f", "fact_id"
    )
    selected_sources = _review_selection_refs(
        args.get("selected_source_refs"), selected_sources, sources_value, "s", "source_event_id"
    )
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


def require_learning_phase_observations(
    calls: Sequence[Mapping[str, Any]], *, phase: str, fact_id: str | None = None
) -> dict[str, Any]:
    """Apply the prospective pilot outcome policy without rewriting observations.

    Callers must supply the complete inventory-bound observation stream. This
    checks permitted operations and outcomes, not the remaining lifecycle,
    database, source replay or restart requirements.
    """
    recall = {"mnemos_retrieve", "mnemos_inspect"}
    allowed = {
        "teach": recall
        | {"mnemos_store", "mnemos_consolidate", "mnemos_job_status", "mnemos_list"},
        "recall-v1": recall,
        "project-2-recall": recall,
        "correct": recall | {"mnemos_correct", "mnemos_consolidate", "mnemos_job_status"},
        "recall-v2": recall,
        "forget": recall | {"mnemos_forget_preview", "mnemos_forget_related"},
        "recall-forgotten": recall,
    }
    if phase not in allowed or not calls:
        raise ValueError("Unknown or empty learning phase")
    if (phase == "forget" and (not isinstance(fact_id, str) or not fact_id.strip())) or (
        phase != "forget" and fact_id is not None
    ):
        raise ValueError("Learning phase has an invalid deletion target")
    sessions: set[str] = set()
    identities: set[str] = set()
    previous_start = 0
    mcp: list[Mapping[str, Any]] = []
    discovery: list[str] = []
    missing: list[Mapping[str, Any]] = []
    seen_discovery: set[str] = set()
    bad_status = {"rejected", "degraded", "error", "failed", "failure"}
    expected_missing = {
        "schema_version": 1,
        "status": "degraded",
        "committed": False,
        "degraded": True,
        "error": "Mnemos service returned HTTP 404 (fact_not_found)",
    }
    for call in calls:
        start, end = call.get("started_sequence"), call.get("completed_sequence")
        session, identity = call.get("session_id"), call.get("call_id")
        if (
            type(start) is not int
            or type(end) is not int
            or start <= previous_start
            or end <= start
            or not isinstance(session, str)
            or not session.strip()
            or not isinstance(identity, str)
            or not identity.strip()
            or identity in identities
            or call.get("model_call_correlated") is not True
        ):
            raise ValueError("Learning observation lacks ordered native identity")
        previous_start = start
        sessions.add(session)
        identities.add(identity)
        if len(sessions) != 1:
            raise ValueError("Learning observations cross native sessions")
        name, args, result = call.get("name"), call.get("args"), call.get("result")
        outcome, kind = call.get("outcome"), call.get("dispatch_kind")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(args, Mapping)
            or not isinstance(result, Mapping)
        ):
            raise ValueError("Learning observation arguments or result are malformed")
        if outcome not in {"success", "unsuccessful"}:
            raise ValueError("Learning observation has an unknown outcome")
        is_bad = (
            result.get("isError") is True
            or result.get("is_error") is True
            or bool(result.get("error"))
            or result.get("degraded") is True
            or (
                "status" in result
                and (
                    not isinstance(result["status"], str) or result["status"].lower() in bad_status
                )
            )
        )
        if (outcome == "success") == is_bad:
            raise ValueError("Learning outcome label contradicts its result")
        if kind == "codex_discovery":
            key = {
                "list_mcp_resources": "resources",
                "list_mcp_resource_templates": "resourceTemplates",
            }.get(name)
            if (
                key is None
                or mcp
                or name in seen_discovery
                or outcome != "success"
                or dict(args) not in ({}, {"server": "mnemos"})
                or set(result) != {key}
                or not isinstance(result[key], list)
                or any(
                    not isinstance(entry, Mapping) or entry.get("server") != "mnemos"
                    for entry in result[key]
                )
            ):
                raise ValueError("Learning discovery exceeds the bounded Mnemos listing")
            seen_discovery.add(name)
            discovery.append(identity)
            continue
        if kind != "mnemos_mcp" or name not in allowed[phase]:
            raise ValueError("Learning phase issued an unapproved operation")
        mcp.append(call)
        if outcome == "unsuccessful":
            ids = {key: value for key, value in args.items() if key in {"fact_id", "chunk_id"}}
            if (
                phase != "forget"
                or name != "mnemos_inspect"
                or not ids
                or set(args) != set(ids)
                or any(value != fact_id for value in ids.values())
                or dict(result) != expected_missing
                or type(result.get("schema_version")) is not int
                or result.get("committed") is not False
                or result.get("degraded") is not True
                or missing
            ):
                raise ValueError("Learning phase has an unexpected unsuccessful operation")
            missing.append(call)
    if not mcp:
        raise ValueError("Learning phase contains no memory operation")
    if any(
        c["dispatch_kind"] == "codex_discovery"
        and c["completed_sequence"] >= mcp[0]["started_sequence"]
        for c in calls
    ):
        raise ValueError("Discovery did not complete before memory operations")
    if phase == "forget":
        assert isinstance(fact_id, str)
        require_forget(mcp, fact_id)
        apply_index, apply = next(
            (i, c) for i, c in enumerate(mcp) if c["name"] == "mnemos_forget_related"
        )
        if any(c["completed_sequence"] >= apply["started_sequence"] for c in mcp[:apply_index]):
            raise ValueError("Forget started before its review operations completed")
        if missing and apply["completed_sequence"] >= missing[0]["started_sequence"]:
            raise ValueError("Missing-fact inspection did not follow completed deletion")
    return {
        "status": "phase_policy_satisfied",
        "phase": phase,
        "observation_count": len(calls),
        "mcp_call_count": len(mcp),
        "discovery_call_ids": discovery,
        "expected_missing_fact_call_ids": [c["call_id"] for c in missing],
        "outcomes_rewritten": False,
        "lifecycle_complete": False,
    }


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
