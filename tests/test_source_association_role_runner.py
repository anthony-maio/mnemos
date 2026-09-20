"""The role experiment requires complete input coverage and preserves failed controls."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from benchmarks.local_memory.embedding_prefix_development_v1 import load_cases
from benchmarks.local_memory.run_source_association_roles_v1 import summarize, validate_role_vectors
from benchmarks.local_memory.source_association_role_preparation_v1 import PREFIXES


def prepared() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _, _ = load_cases()
    queries = {case["query"]: [1.0] + [0.0] * 767 for case in manifest["graph_cases"]}
    documents = {
        fact["content"]: [0.0, 1.0] + [0.0] * 766
        for case in manifest["graph_cases"]
        for fact in [*case["facts"], *case.get("scope_controls", [])]
    }
    for case in manifest["graph_cases"]:
        for action in case.get("lifecycle_actions", []):
            if action["action"] == "correct":
                documents[action["new_content"]] = [0.0, 1.0] + [0.0] * 766
    return manifest, {
        label: {
            "queries": copy.deepcopy(queries),
            "documents": copy.deepcopy(documents),
            "prefixes": prefixes,
        }
        for label, prefixes in PREFIXES.items()
    }


def test_complete_document_and_query_roles_include_lifecycle_input() -> None:
    manifest, roles = prepared()
    validate_role_vectors(manifest, roles)
    assert all(len(arm["documents"]) == 185 and len(arm["queries"]) == 5 for arm in roles.values())


@pytest.mark.parametrize("role", ["queries", "documents"])
def test_missing_vector_cannot_be_hidden_by_another_role(role: str) -> None:
    manifest, roles = prepared()
    arm = roles["nomic_task"]
    key = next(iter(arm[role]))
    other = "documents" if role == "queries" else "queries"
    arm[other][key] = arm[role].pop(key)
    with pytest.raises(ValueError, match="vector coverage"):
        validate_role_vectors(manifest, roles)


def rows_for(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "embedding_contract": role,
            "case_id": case["id"],
            "arm": arm,
            "repetition": rep,
            "controls": {
                "scope_pass": True,
                **({"lifecycle_pass": False} if case.get("lifecycle_actions") else {}),
            },
            "target": {
                "direct_candidate": False,
                "structural_path": False,
                "activation_path": False,
                "in_top_k": False,
            },
            "source_diagnostic": {
                "relations": (
                    []
                    if case["id"] == "semantic_only_chain"
                    else [{"fact_ids": [case["seed_id"], case["target_id"]]}]
                )
            },
        }
        for role in PREFIXES
        for case in manifest["graph_cases"]
        for rep in range(3)
        for arm in ("semantic", "source_association", "combined")
    ]


def test_completed_grid_keeps_all_eighteen_lifecycle_failures() -> None:
    manifest, _ = prepared()
    result = summarize(rows_for(manifest), manifest)
    assert result["execution_complete"] is True
    assert result["controls_pass"] is result["coverage_pass"] is False
    assert len(result["failed_controls"]) == 18
    assert {row["control"] for row in result["failed_controls"]} == {"lifecycle_pass"}
    assert all(
        summary["rows"] == 15
        for roles in result["summary_by_embedding"].values()
        for summary in roles.values()
    )


def test_duplicate_row_does_not_substitute_for_missing_scheduled_row() -> None:
    manifest, _ = prepared()
    rows = rows_for(manifest)
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="scheduled grid"):
        summarize(rows, manifest)
