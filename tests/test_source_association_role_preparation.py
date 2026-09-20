"""Role preparation preserves the original graph inputs and lifecycle content."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.embedding_prefix_development_v1 import load_cases
from benchmarks.local_memory.source_association_role_preparation_v1 import (
    corrected_contents,
    maps_from_paired,
)


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _, cases = load_cases()
    arms = []
    for label, prefixes in (
        ("unprefixed", {"query_prefix": "", "document_prefix": "", "symmetric_prefix": ""}),
        (
            "nomic_task",
            {
                "query_prefix": "search_query: ",
                "document_prefix": "search_document: ",
                "symmetric_prefix": "",
            },
        ),
    ):
        rows = []
        for case in cases:
            rows.append(
                {
                    "case_id": case["id"],
                    "query_vector": [1.0] + [0.0] * 767,
                    "document_vectors": {
                        fact["id"]: [0.0, 1.0] + [0.0] * 766
                        for fact in [*case["facts"], *case.get("scope_controls", [])]
                    },
                }
            )
        arms.append({"label": label, "prefixes": prefixes, "dimension": 768, "cases": rows})
    return manifest, {"arms": arms}


def test_same_text_in_two_roles_retains_two_different_vectors() -> None:
    manifest, paired = fixture()
    case = manifest["graph_cases"][0]
    case["query"] = case["facts"][0]["content"]
    result = maps_from_paired(manifest, paired)
    for arm in result.values():
        assert len(arm["queries"]) == 5 and len(arm["documents"]) == 184
        assert arm["queries"][case["query"]][:2] == [1.0, 0.0]
        assert arm["documents"][case["query"]][:2] == [0.0, 1.0]


@pytest.mark.parametrize("fault", ["missing_control", "duplicate_case", "wrong_role", "nan"])
def test_rejects_incomplete_or_misbound_paired_vectors(fault: str) -> None:
    manifest, paired = fixture()
    arm = paired["arms"][0]
    if fault == "missing_control":
        orchard = next(c for c in manifest["graph_cases"] if c.get("scope_controls"))
        row = next(c for c in arm["cases"] if c["case_id"] == orchard["id"])
        del row["document_vectors"][orchard["scope_controls"][0]["id"]]
    elif fault == "duplicate_case":
        arm["cases"][1] = copy.deepcopy(arm["cases"][0])
    elif fault == "wrong_role":
        arm["prefixes"]["query_prefix"] = "search_document: "
    else:
        arm["cases"][0]["query_vector"][0] = float("nan")
    with pytest.raises(ValueError):
        maps_from_paired(manifest, paired)


def test_lifecycle_requires_exact_authored_corrected_content() -> None:
    manifest, _ = fixture()
    case = next(c for c in manifest["graph_cases"] if c.get("lifecycle_actions"))
    correction = case["lifecycle_actions"][0]["new_content"]
    assert corrected_contents(manifest) == [correction]
    assert all(correction != f["content"] for c in manifest["graph_cases"] for f in c["facts"])
