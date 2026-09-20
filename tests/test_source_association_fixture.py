from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any, cast

import pytest

from benchmarks.local_memory.source_association_diagnostic import (
    inspect_source_associations,
)
from benchmarks.local_memory.source_association_fixture import (
    populate_source_association_fixture,
)
from mnemos.utils.storage import SQLiteStore

_MANIFEST = (
    Path(__file__).parents[1]
    / "benchmarks/local_memory/datasets/source-association-development-v1.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_MANIFEST.read_text(encoding="utf-8")))


def _case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in manifest["graph_cases"] if case["id"] == case_id)


def _vectors(case: dict[str, Any]) -> dict[str, list[float]]:
    items = case["facts"] + case.get("scope_controls", [])
    return {
        item["content"]: [1.0, float(index + 1), float((index % 3) + 1)]
        for index, item in enumerate(items)
    }


def _relation_signature(result: dict[str, Any]) -> set[tuple[str, tuple[str, str]]]:
    return {
        (str(relation["source_event_id"]), tuple(sorted(relation["fact_ids"])))
        for relation in result["relations"]
    }


def test_shared_capture_materializes_real_relations_and_durable_jobs(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    case = _case(manifest, "shared_capture_triplet")
    destination = tmp_path / "shared.db"
    result = populate_source_association_fixture(case, _vectors(case), destination)

    assert result["status"] == "materialized"
    assert result["counts"] == {
        "episodic_events": 34,
        "facts": 36,
        "scope_controls": 0,
        "query_eligible_facts": 36,
        "evidence": 36,
        "jobs": 34,
        "completed_jobs": 34,
    }
    diagnostics = inspect_source_associations(destination)
    focal = "shared_capture_triplet-source-shared"
    focal_relations = [r for r in diagnostics["relations"] if r["source_event_id"] == focal]
    assert len(focal_relations) == 3
    assert {fact_id for relation in focal_relations for fact_id in relation["fact_ids"]} == {
        "shared_capture_triplet-crate",
        "shared_capture_triplet-contact",
        "shared_capture_triplet-dock",
    }

    reopened = SQLiteStore(db_path=str(destination))
    try:
        assert len(reopened.list_jobs(status="completed")) == 34
        assert len(reopened.list_episodic_events(status="consolidated")) == 34
    finally:
        reopened.close()
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT COUNT(*) FROM fact_evidence").fetchone()[0] == 36


def test_semantic_only_case_has_no_source_relations(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    case = _case(manifest, "semantic_only_chain")
    destination = tmp_path / "semantic-only.db"
    result = populate_source_association_fixture(case, _vectors(case), destination)

    assert result["counts"]["query_eligible_facts"] == 36
    diagnostics = inspect_source_associations(destination)
    assert diagnostics["relations"] == []
    assert all(group["status"] != "eligible" for group in diagnostics["groups"])


def test_scope_controls_are_persisted_but_excluded_by_primary_scope_query(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    case = _case(manifest, "scope_isolation_pair")
    destination = tmp_path / "scope.db"
    populate_source_association_fixture(case, _vectors(case), destination)

    controls = {item["id"] for item in case["scope_controls"]}
    store = SQLiteStore(db_path=str(destination))
    try:
        selected = store.retrieve(
            [1.0, 1.0, 1.0],
            top_k=100,
            filter_fn=lambda chunk: chunk.metadata.get("user_profile_id") == "assoc-profile-orchard"
            and (
                (
                    chunk.metadata.get("scope") == "project"
                    and chunk.metadata.get("scope_id") == "assoc-project-orchard-east"
                )
                or (
                    chunk.metadata.get("scope") == "workspace"
                    and chunk.metadata.get("scope_id") == "assoc-workspace-orchard-maintenance"
                )
            ),
        )
    finally:
        store.close()
    selected_ids = {chunk.id for chunk in selected}
    assert not selected_ids & controls
    assert len(selected_ids) == 36
    assert all(item["query_eligible"] is False for item in case["scope_controls"])

    diagnostics = inspect_source_associations(destination)
    foreign_relations = [
        relation for relation in diagnostics["relations"] if set(relation["fact_ids"]) & controls
    ]
    assert foreign_relations
    assert all(
        relation["source_event_id"] in {item["source_event_id"] for item in case["scope_controls"]}
        for relation in foreign_relations
    )


def test_forward_reverse_insertion_and_restart_have_same_relations(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    case = _case(manifest, "shared_extraction_batch")
    forward_path = tmp_path / "forward.db"
    reverse_path = tmp_path / "reverse.db"
    populate_source_association_fixture(
        case, _vectors(case), forward_path, insertion_order="forward"
    )
    populate_source_association_fixture(
        case, _vectors(case), reverse_path, insertion_order="reverse"
    )

    forward = inspect_source_associations(forward_path)
    reverse = inspect_source_associations(reverse_path)
    assert _relation_signature(forward) == _relation_signature(reverse)
    reopened = SQLiteStore(db_path=str(forward_path))
    try:
        assert reopened.list_jobs(status="completed")
    finally:
        reopened.close()
    assert _relation_signature(inspect_source_associations(forward_path)) == _relation_signature(
        forward
    )


@pytest.mark.parametrize(
    "bad_kind",
    [
        "missing_vector",
        "bad_source_ref",
        "zero_vector",
        "forged_eligibility",
        "invalid_role",
        "missing_project",
    ],
)
def test_invalid_inputs_fail_before_destination_creation(
    manifest: dict[str, Any], tmp_path: Path, bad_kind: str
) -> None:
    case = deepcopy(_case(manifest, "shared_capture_triplet"))
    vectors = _vectors(case)
    destination = tmp_path / f"{bad_kind}.db"
    if bad_kind == "missing_vector":
        vectors.pop(case["facts"][0]["content"])
    elif bad_kind == "bad_source_ref":
        case["facts"][0]["source_event_id"] = "missing-event"
    elif bad_kind == "zero_vector":
        vectors[case["facts"][0]["content"]] = [0.0, 0.0, 0.0]
    elif bad_kind == "forged_eligibility":
        case["facts"][0]["scope"]["user_profile_id"] = "foreign-profile"
    elif bad_kind == "invalid_role":
        case["events"][0]["role"] = "invalid"
    else:
        case["facts"][0]["scope"]["scope_id"] = None
        case["facts"][0]["scope"]["project_id"] = None

    with pytest.raises(ValueError):
        populate_source_association_fixture(case, vectors, destination)
    assert not destination.exists()


def test_destination_is_refused_without_mutation(manifest: dict[str, Any], tmp_path: Path) -> None:
    case = _case(manifest, "shared_capture_triplet")
    destination = tmp_path / "existing.db"
    destination.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError):
        populate_source_association_fixture(case, _vectors(case), destination)
    assert destination.read_bytes() == b"sentinel"
