from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from mnemos.scopes import ScopeContext, scope_allows

MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "local_memory"
    / "datasets"
    / "source-association-development-v1.json"
)


def _manifest() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def _context(case: dict[str, Any]) -> ScopeContext:
    context = case["context"]
    return ScopeContext(
        user_profile_id=context["user_profile_id"],
        project_id=context["project_id"],
        workspace_id=context["workspace_id"],
        session_id=context["session_id"],
        host=context["host"],
    )


def test_manifest_has_exact_five_disjoint_development_cases_and_fixed_options() -> None:
    manifest = _manifest()
    assert manifest["version"] == "source-association-development-v1"
    assert [case["id"] for case in manifest["graph_cases"]] == [
        "shared_capture_triplet",
        "shared_extraction_batch",
        "semantic_only_chain",
        "scope_isolation_pair",
        "revision_forget_replay",
    ]
    assert manifest["options"] == {
        "top_k": 5,
        "context_budget_tokens": 4096,
        "max_candidates": 100,
        "neighbor_budget": 4,
        "relevance_floor": 0.2,
        "semantic_edge_threshold": 0.6,
        "activation_threshold": 0.3,
        "decay": 0.2,
        "max_hops": 3,
        "max_visited": 1000,
        "graph_weight": 0.1,
        "source_edge_weight": 1.0,
        "fanout_limit": 16,
        "repetitions": 3,
    }
    assert all(case["split"] == "development" for case in manifest["graph_cases"])
    assert all(case["query_eligible_count"] == 36 for case in manifest["graph_cases"])

    contexts = [case["context"] for case in manifest["graph_cases"]]
    assert len({context["user_profile_id"] for context in contexts}) == 5
    assert len({context["project_id"] for context in contexts}) == 5
    assert len({case["template_id"] for case in manifest["graph_cases"]}) == 5
    assert len({case["query"] for case in manifest["graph_cases"]}) == 5


def test_facts_are_scope_eligible_and_controls_are_real_negative_cases() -> None:
    manifest = _manifest()
    fact_ids: set[str] = set()
    evidence_ids: set[str] = set()
    event_ids: set[str] = set()
    for case in manifest["graph_cases"]:
        context = _context(case)
        assert len(case["facts"]) == 36
        events = {event["id"]: event for event in case["events"]}
        assert len(events) == len(case["events"])
        assert not event_ids.intersection(events)
        event_ids.update(events)
        for event in events.values():
            assert event["role"] in {"user", "assistant", "tool", "system"}
            assert event["source_kind"]
            assert event["scope"] in {"project", "workspace", "global"}
            assert event["scope_id"] is not None or event["scope"] == "global"
            assert event["user_profile_id"]
        for fact in case["facts"]:
            assert fact["query_eligible"] is True
            assert scope_allows(fact["scope"], context, (fact["scope"]["scope"],))
            assert fact["source_event_id"] in events
            assert "embedding" not in fact
            assert "rank" not in fact
            assert "score" not in fact
            assert fact["id"] not in fact_ids
            assert fact["evidence_id"] not in evidence_ids
            fact_ids.add(fact["id"])
            evidence_ids.add(fact["evidence_id"])

        for control in case["scope_controls"]:
            allowed = scope_allows(control["scope"], context, (control["scope"]["scope"],))
            event = events[control["source_event_id"]]
            assert event["scope"] == control["scope"]["scope"]
            assert event["scope_id"] == control["scope"]["scope_id"]
            assert event["user_profile_id"] == control["scope"]["user_profile_id"]
            assert control["query_eligible"] is False
            assert control["expected_scope_allowed"] is False
            assert allowed is control["expected_scope_allowed"]
            assert control["id"] not in fact_ids
            assert control["evidence_id"] not in evidence_ids
            fact_ids.add(control["id"])
            evidence_ids.add(control["evidence_id"])

        eligible_records = sum(
            scope_allows(record["scope"], context, (record["scope"]["scope"],))
            for record in [*case["facts"], *case["scope_controls"]]
        )
        assert eligible_records == 36

    scoped = next(case for case in manifest["graph_cases"] if case["id"] == "scope_isolation_pair")
    controls = scoped["scope_controls"]
    assert {control["control_kind"] for control in controls} == {
        "rejected_foreign_project",
        "rejected_foreign_profile",
    }
    for kind in (
        "rejected_foreign_project",
        "rejected_foreign_profile",
    ):
        group = [control for control in controls if control["control_kind"] == kind]
        assert len(group) == 2
        assert len({control["source_event_id"] for control in group}) == 1


def test_shared_source_patterns_have_supported_focal_content_and_semantic_control() -> None:
    manifest = _manifest()
    cases = {case["id"]: case for case in manifest["graph_cases"]}
    for case_id in (
        "shared_capture_triplet",
        "shared_extraction_batch",
        "scope_isolation_pair",
        "revision_forget_replay",
    ):
        case = cases[case_id]
        events = {event["id"]: event for event in case["events"]}
        pattern = case["source_pattern"]
        assert pattern["shared_event_ids"]
        for group in pattern["expected_shared_fact_ids"]:
            event_id = pattern["shared_event_ids"][0]
            event_content = events[event_id]["content"].lower()
            for fact_id in group:
                fact = next(item for item in case["facts"] if item["id"] == fact_id)
                assert fact["content"].lower() in event_content

    orchard = cases["scope_isolation_pair"]
    orchard_group = orchard["source_pattern"]["expected_shared_fact_ids"][0]
    orchard_scopes = {
        next(fact for fact in orchard["facts"] if fact["id"] == fact_id)["scope"]["scope"]
        for fact_id in orchard_group
    }
    assert orchard_scopes == {"project", "workspace"}

    semantic = cases["semantic_only_chain"]
    assert semantic["source_pattern"]["shared_event_ids"] == []
    assert len({fact["source_event_id"] for fact in semantic["facts"]}) == 36


def test_revision_actions_are_fact_level_and_reference_authored_state() -> None:
    manifest = _manifest()
    lifecycle = next(
        case for case in manifest["graph_cases"] if case["id"] == "revision_forget_replay"
    )
    fact_ids = {fact["id"] for fact in lifecycle["facts"]}
    event_ids = {event["id"] for event in lifecycle["events"]}
    actions = lifecycle["lifecycle_actions"]
    assert [action["action"] for action in actions] == [
        "correct",
        "forget",
        "replay",
        "restart",
    ]
    for action in actions:
        if "fact_id" in action:
            assert action["fact_id"] in fact_ids
        if "source_event_id" in action:
            assert action["source_event_id"] in event_ids
    correction = actions[0]
    assert correction["new_content"]
    assert correction["new_source_event_id"] in event_ids
    forget = actions[1]
    assert forget["expect_fact_absent"] is True
    assert set(forget["expect_surviving_fact_ids"]) == {
        "revision_forget_replay-cooler",
        "revision_forget_replay-seal",
    }
    assert forget["expect_shared_source_retained"] is True
    replay = actions[2]
    assert replay["expect_suppressed_fact_id"] == forget["fact_id"]
    assert replay["expect_surviving_fact_ids"] == forget["expect_surviving_fact_ids"]
