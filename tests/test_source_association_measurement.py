from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.source_association_fixture import populate_source_association_fixture
from benchmarks.local_memory.source_association_measurement import (
    observe_retrieval,
    open_measurement_engine,
)
from benchmarks.local_memory.run_source_association import measure_repetition


def _synthetic_case() -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any]]:
    path = (
        Path(__file__).parents[1]
        / "benchmarks/local_memory/datasets/source-association-development-v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    case = copy.deepcopy(manifest["graph_cases"][0])
    # Synthetic geometry, never the real Nomic artifact. The target is outside
    # the direct 30-candidate pool but shares real persisted source evidence.
    case["query"] = "synthetic-query-with-no-lexical-overlap"
    vectors = {fact["content"]: [0.7, 0.714] for fact in case["facts"]}
    vectors[case["facts"][0]["content"]] = [1.0, 0.0]
    vectors[case["facts"][1]["content"]] = [0.8, 0.6]
    vectors[case["facts"][2]["content"]] = [0.25, 0.968]
    vectors[case["query"]] = [1.0, 0.0]
    return case, vectors, manifest["options"]


@pytest.mark.asyncio
async def test_source_observation_records_outside_pool_path_without_claiming_top_five(
    tmp_path: Path,
) -> None:
    case, vectors, options = _synthetic_case()
    database = tmp_path / "source.sqlite"
    populate_source_association_fixture(case, vectors, database)
    engine, store = open_measurement_engine(database, case, vectors, options)
    try:
        result = await observe_retrieval(
            engine, store, case, vectors, options, arm="source_association"
        )
        assert result["target"]["direct_candidate"] is False
        assert result["target"]["structural_path"] is True
        assert result["target"]["activation_path"] is True
        assert result["target"]["in_top_k"] is False
        assert result["target"]["rank"] > 5
        assert result["candidate_pool"]["actual_selected_seed_id"] == case["seed_id"]
        assert all(result["controls"].values())
        target = next(
            item for item in result["candidate_scores"] if item["id"] == case["target_id"]
        )
        assert target["graph_energy"] == pytest.approx(0.8)
        assert target["graph_contribution"] == pytest.approx(0.08)
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_repetition_keeps_all_three_arms_and_real_control_observations(
    tmp_path: Path,
) -> None:
    case, vectors, options = _synthetic_case()
    rows = await measure_repetition(case, vectors, options, tmp_path / "repetition", 0)
    assert [row["arm"] for row in rows] == ["semantic", "source_association", "combined"]
    for row in rows:
        assert all(value is True for value in row["controls"].values() if value is not None)
        assert row["control_observations"]["edge_cut"]["target"]["energy"] == 0.0
        assert row["control_observations"]["edge_cut"]["candidate_pool"] == row["candidate_pool"]
        saved = json.loads((tmp_path / "repetition" / (row["arm"] + ".json")).read_text())
        assert saved["candidate_scores"] == row["candidate_scores"]


@pytest.mark.asyncio
async def test_observation_uses_configured_visit_budget_beyond_legacy_default(
    tmp_path: Path,
) -> None:
    case, vectors, options = _synthetic_case()
    source_id = case["facts"][0]["source_event_id"]
    for fact in case["facts"][:15]:
        fact["source_event_id"] = source_id
    source = next(event for event in case["events"] if event["id"] == source_id)
    source["content"] = " ".join(fact["content"] for fact in case["facts"][:15])
    database = tmp_path / "wide-source.sqlite"
    populate_source_association_fixture(case, vectors, database)
    engine, store = open_measurement_engine(database, case, vectors, options)
    try:
        result = await observe_retrieval(
            engine, store, case, vectors, options, arm="source_association"
        )
        assert len(result["independent_activation"]) == 15
        assert len(result["activation"]) == 15
        assert result["controls"]["activation_oracle_pass"] is True
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_semantic_view_matches_untouched_production_retrieval_and_restart(
    tmp_path: Path,
) -> None:
    case, vectors, options = _synthetic_case()
    database = tmp_path / "semantic.sqlite"
    populate_source_association_fixture(case, vectors, database)
    engine, store = open_measurement_engine(database, case, vectors, options)
    try:
        untouched = await observe_retrieval(engine, store, case, vectors, options, arm=None)
        projected = await observe_retrieval(engine, store, case, vectors, options, arm="semantic")
        assert untouched["returned_ids"] == projected["returned_ids"]
        assert untouched["candidate_scores"] == projected["candidate_scores"]
        assert all(untouched["controls"].values())
        assert all(projected["controls"].values())
    finally:
        await engine.aclose()
    engine, store = open_measurement_engine(database, case, vectors, options)
    try:
        restarted = await observe_retrieval(engine, store, case, vectors, options, arm="semantic")
        assert restarted["candidate_scores"] == projected["candidate_scores"]
        assert restarted["edges"] == projected["edges"]
        assert all(restarted["controls"].values())
    finally:
        await engine.aclose()
