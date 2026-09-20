"""Read-only diagnostics for answerable retrieval misses.

This module reports evidence already present in a retrieval run. It does not
rejudge an answer, infer semantic entailment, or change the benchmark metric.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

STRATEGY_ORDER = ("lexical", "full", "vector", "hybrid")
BOUNDARY_LABELS = (
    "exact_stored_but_not_returned",
    "source_linked_returned_different_text",
    "source_linked_not_returned",
    "no_source_linked_runtime_fact",
    "unverified",
)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _raw_outputs(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in _list(report.get("raw_outputs")) if _mapping(item)]


def _session_id(item: Mapping[str, Any]) -> str | None:
    value = item.get("session_id")
    return value if isinstance(value, str) else None


def _find_session(
    raw: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    wanted = _session_id(raw)
    if wanted is None:
        return None
    return next((item for item in sessions if _session_id(item) == wanted), None)


def _find_query(
    session: Mapping[str, Any] | None, query_id: str | None
) -> Mapping[str, Any] | None:
    if session is None or query_id is None:
        return None
    for item in _list(session.get("queries")):
        query = _mapping(item)
        if query is not None and query.get("query_id") == query_id:
            return query
    return None


def _facts(session: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if session is None:
        return {}
    return {
        str(item["fact_id"]): item
        for item in (_mapping(value) for value in _list(session.get("facts")))
        if item is not None and isinstance(item.get("fact_id"), str)
    }


def _gold_facts(
    session: Mapping[str, Any] | None, query: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    all_facts = _facts(session)
    if query is None:
        return []
    return [
        fact
        for fact_id in _list(query.get("relevant_fact_ids"))
        if isinstance(fact_id, str)
        for fact in [all_facts.get(fact_id)]
        if fact is not None and fact.get("gold_status") == "supported"
    ]


def _expected_event_ids(gold_facts: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        event_id
        for fact in gold_facts
        for event_id in _list(fact.get("evidence_event_ids"))
        if isinstance(event_id, str)
    }


def _returned_facts(detail: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = _mapping(detail.get("result"))
    if result is None:
        return []
    return [item for item in (_mapping(value) for value in _list(result.get("facts"))) if item]


def _source_judgements(raw_output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in (_mapping(value) for value in _list(raw_output.get("extraction_judgements")))
        if item is not None
    ]


def _runtime_ids(judgements: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        value
        for item in judgements
        for value in [item.get("runtime_fact_id")]
        if isinstance(value, str) and value
    }


def _observed_source_ids(judgements: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        source_id
        for item in judgements
        for source_id in _list(item.get("observed_source_event_ids"))
        if isinstance(source_id, str)
    }


def _remember_match(
    session: Mapping[str, Any] | None,
    expected_ids: set[str],
    gold_content: str | None,
) -> bool | None:
    if session is None or gold_content is None:
        return None
    events = [
        event
        for event in (_mapping(value) for value in _list(session.get("events")))
        if event is not None
        and event.get("event_id") in expected_ids
        and (
            event.get("source_kind") == "explicit_remember"
            or event.get("lifecycle_operation") == "remember"
        )
    ]
    if not events:
        return None
    return any(event.get("content") == gold_content for event in events)


def _scope_status(
    returned: Iterable[Mapping[str, Any]],
    session: Mapping[str, Any] | None,
    gold: Mapping[str, Any],
) -> str:
    items = list(returned)
    if not items:
        return "unavailable"
    expected_scope = gold.get("scope")
    expected_scope_id = session.get("project_id") if session is not None else None
    if not isinstance(expected_scope, str):
        return "unverified"
    if expected_scope != "project" or not isinstance(expected_scope_id, str):
        return "unverified"
    for item in items:
        scope = item.get("scope")
        scope_id = item.get("scope_id")
        if not isinstance(scope, str) or not isinstance(scope_id, str):
            return "unverified"
        if scope != expected_scope or scope_id != expected_scope_id:
            return "mismatch"
    return "match"


def _target_context_status(
    query: Mapping[str, Any] | None, session: Mapping[str, Any] | None
) -> str | None:
    if query is None or session is None:
        return None
    expected_project = session.get("project_id")
    for key in ("project_id", "scope_id", "target_project_id"):
        if key in query:
            if not isinstance(expected_project, str):
                return "unverified"
            if query.get(key) != expected_project:
                return "mismatch"
    expected_profile = session.get("user_profile_id") or session.get("profile")
    for key in ("user_profile_id", "target_user_profile_id", "profile"):
        if key in query:
            if not isinstance(expected_profile, str):
                return "unverified"
            if query.get(key) != expected_profile:
                return "mismatch"
    return "match"


def _original_hit(
    raw: Mapping[str, Any], comparison: Mapping[str, Any], detail: Mapping[str, Any], index: int
) -> bool | None:
    value = detail.get("quality_hit")
    if isinstance(value, bool):
        return value
    return None


def _critical_presence(raw_output: Mapping[str, Any], gold_ids: set[str]) -> bool | None:
    for value in _list(raw_output.get("critical_retention_judgements")):
        item = _mapping(value)
        if item is None:
            continue
        if gold_ids and item.get("fact_id") not in gold_ids:
            continue
        if "observed_present" in item:
            observed = _bool(item.get("observed_present"))
            if observed is not None:
                return observed
    return None


def _failure_boundary(
    *,
    source_matches: set[str],
    returned: list[Mapping[str, Any]],
    accepted_returned: list[Mapping[str, Any]],
    exact_matches: list[str],
    literal_span: bool,
    scope_status: str,
    context_status: str | None,
    remember_equal: bool | None,
    critical_exact_fact_present: bool | None,
    runtime_ids: set[str],
    returned_ids: set[str],
    linked_runtime_ids: set[str],
    gold_content: str | None,
) -> str | None:
    if context_status != "match" or scope_status in {"mismatch", "unverified"}:
        return "unverified"
    if any(
        item.get("assertion_status") in {"conflict", "negated", "rejected"} for item in returned
    ):
        return "unverified"
    if not gold_content or not returned and not source_matches:
        return "unverified"
    linked_returned_ids = returned_ids & linked_runtime_ids
    if accepted_returned and linked_returned_ids:
        if exact_matches:
            return None
        return "source_linked_returned_different_text"
    if source_matches and linked_runtime_ids - returned_ids:
        return (
            "exact_stored_but_not_returned"
            if critical_exact_fact_present is True
            else "source_linked_not_returned"
        )
    if accepted_returned and literal_span and not source_matches:
        return "no_source_linked_runtime_fact"
    if returned and literal_span and not source_matches:
        return "no_source_linked_runtime_fact"
    return "unverified"


def _row(
    raw: Mapping[str, Any],
    raw_output: Mapping[str, Any],
    comparison: Mapping[str, Any],
    strategy: str,
    detail: Mapping[str, Any],
    session: Mapping[str, Any] | None,
    query: Mapping[str, Any] | None,
    answer_index: int,
) -> dict[str, Any]:
    gold = _gold_facts(session, query)
    gold_content = next(
        (value.get("content") for value in gold if isinstance(value.get("content"), str)), None
    )
    expected_ids = _expected_event_ids(gold)
    judgements = _source_judgements(raw_output)
    source_ids = _observed_source_ids(judgements)
    runtime_ids = _runtime_ids(judgements)
    linked_runtime_ids = {
        runtime_id
        for item in judgements
        for runtime_id in [item.get("runtime_fact_id")]
        if isinstance(runtime_id, str)
        and bool(
            {
                source_id
                for source_id in _list(item.get("observed_source_event_ids"))
                if isinstance(source_id, str)
            }
            & expected_ids
        )
    }
    returned = _returned_facts(detail)
    returned_ids = {
        value for item in returned for value in [item.get("id")] if isinstance(value, str)
    }
    scope_status = _scope_status(returned, session, gold[0]) if gold else "unverified"
    context_status = _target_context_status(query, session)
    accepted_returned = [
        item
        for item in returned
        if item.get("assertion_status") == "asserted" and scope_status == "match"
    ]
    allowed_contents = {
        content.strip().casefold()
        for content in [
            *(item.get("content") for item in gold),
            query.get("expected_answer") if query is not None else None,
        ]
        if isinstance(content, str) and content
    }
    exact_matches = [
        str(item["id"])
        for item in accepted_returned
        if isinstance(item.get("id"), str)
        and isinstance(item.get("content"), str)
        and item["content"].strip().casefold() in allowed_contents
    ]
    literal_span = bool(
        gold_content
        and any(
            isinstance(item.get("content"), str) and gold_content in item["content"]
            for item in returned
        )
    )
    matching_sources = source_ids & expected_ids
    remember_equal = _remember_match(session, expected_ids, gold_content)
    result: dict[str, Any] = {
        "session_id": raw.get("session_id"),
        "query_id": comparison.get("query_id"),
        "strategy": strategy,
        "original_quality_hit": _original_hit(raw, comparison, detail, answer_index),
        "gold_fact_ids": [
            str(item["fact_id"]) for item in gold if isinstance(item.get("fact_id"), str)
        ],
        "returned_fact_ids": sorted(returned_ids),
        "runtime_fact_ids": sorted(runtime_ids),
        "linked_runtime_fact_ids": sorted(linked_runtime_ids),
        "omitted_runtime_fact_ids": sorted(runtime_ids - returned_ids),
        "source_event_ids_observed": sorted(source_ids),
        "source_event_ids_expected": sorted(expected_ids),
        "source_event_ids_matching_gold": sorted(matching_sources),
        "exact_content_matches": exact_matches,
        "expected_fact_literal_substring": literal_span,
        "expected_answer_literal_substring": bool(
            isinstance(query, Mapping)
            and isinstance(query.get("expected_answer"), str)
            and any(query["expected_answer"] in str(item.get("content", "")) for item in returned)
        ),
        "critical_exact_fact_present": _critical_presence(
            raw_output,
            {str(item["fact_id"]) for item in gold if isinstance(item.get("fact_id"), str)},
        ),
        "remember_event_content_equal_gold": remember_equal,
        "remember_event_content_status": (
            "equal"
            if remember_equal is True
            else "different" if remember_equal is False else "unverified"
        ),
        "repetition": raw.get("repetition"),
        "scope_status": scope_status,
        "target_context_status": context_status,
    }
    result["failure_boundary"] = _failure_boundary(
        source_matches=matching_sources,
        returned=returned,
        accepted_returned=accepted_returned,
        exact_matches=exact_matches,
        literal_span=literal_span,
        scope_status=scope_status,
        context_status=context_status,
        remember_equal=remember_equal,
        critical_exact_fact_present=result["critical_exact_fact_present"],
        runtime_ids=runtime_ids,
        returned_ids=returned_ids,
        linked_runtime_ids=linked_runtime_ids,
        gold_content=gold_content,
    )
    return result


def analyze(report: dict[str, Any], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return conservative per-strategy diagnostics without changing metrics."""
    session_map = {
        session_id: session
        for session in sessions
        for session_id in [_session_id(session)]
        if session_id is not None
    }
    observations: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    strategy_order: list[str] = []
    strategy_summary: dict[str, dict[str, int]] = {
        strategy: {"complete": 0, "partial": 0, "original_quality_hits": 0}
        for strategy in STRATEGY_ORDER
    }
    strategy_hits = 0
    strategy_denominator = 0
    original_hits = 0
    original_denominator = 0
    for raw in _raw_outputs(report):
        raw_output = _mapping(raw.get("raw_output"))
        if raw_output is None:
            continue
        session = session_map.get(_session_id(raw) or "")
        answerable_index = 0
        for comparison_value in _list(raw_output.get("retrieval_comparisons")):
            comparison = _mapping(comparison_value)
            if comparison is None or comparison.get("quality_kind") != "answerable":
                continue
            raw_hit_values = _list(raw.get("answerable_hits"))
            raw_hit = (
                raw_hit_values[answerable_index]
                if answerable_index < len(raw_hit_values)
                and isinstance(raw_hit_values[answerable_index], bool)
                else None
            )
            answerable_index += 1
            query = _find_query(session, _text(comparison.get("query_id")))
            strategies = _mapping(comparison.get("strategies")) or {}
            declared = [
                value
                for value in _list(comparison.get("strategy_order"))
                if isinstance(value, str) and value in strategies
            ]
            order = declared or [name for name in STRATEGY_ORDER if name in strategies]
            order.extend(name for name in strategies if name not in order)
            for index, strategy in enumerate(order):
                detail = _mapping(strategies.get(strategy)) or {}
                row = _row(
                    raw,
                    raw_output,
                    comparison,
                    strategy,
                    detail,
                    session,
                    query,
                    answerable_index - 1,
                )
                complete = (
                    raw.get("execution_status") == "complete" and detail.get("status") == "complete"
                )
                if complete:
                    observations.append(row)
                    if row["original_quality_hit"] is not None:
                        strategy_denominator += 1
                    strategy_summary.setdefault(
                        strategy, {"complete": 0, "partial": 0, "original_quality_hits": 0}
                    )["complete"] += 1
                    if row["original_quality_hit"] is True:
                        strategy_hits += 1
                        strategy_summary[strategy]["original_quality_hits"] += 1
                else:
                    partial.append(row)
                    strategy_summary.setdefault(
                        strategy, {"complete": 0, "partial": 0, "original_quality_hits": 0}
                    )["partial"] += 1
                if strategy not in strategy_order:
                    strategy_order.append(strategy)
            if raw.get("execution_status") == "complete" and raw_hit is not None:
                original_denominator += 1
                if raw_hit:
                    original_hits += 1
    for strategy in STRATEGY_ORDER:
        if strategy not in strategy_order:
            strategy_order.append(strategy)
    boundary_counts = {label: 0 for label in BOUNDARY_LABELS}
    for row in observations:
        label = row.get("failure_boundary")
        if isinstance(label, str) and label in boundary_counts:
            boundary_counts[label] += 1
    return {
        "schema_version": 1,
        "strategy_order": strategy_order,
        "strategy_summary": strategy_summary,
        "observations": observations,
        "partial_observations": partial,
        "denominators": {
            "complete_answerable": len(observations),
            "partial_answerable": len(partial),
        },
        "original_quality_hits": {"count": original_hits, "denominator": original_denominator},
        "strategy_quality_hits": {"count": strategy_hits, "denominator": strategy_denominator},
        "failure_boundary_counts": boundary_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    sessions = json.loads(args.sessions.read_text(encoding="utf-8"))
    result = analyze(report, sessions)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
