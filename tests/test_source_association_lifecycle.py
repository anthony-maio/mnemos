import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.source_association_fixture import (
    populate_source_association_fixture,
)
from benchmarks.local_memory.source_association_lifecycle import run_lifecycle_control

_MANIFEST = (
    Path(__file__).parents[1]
    / "benchmarks/local_memory/datasets/source-association-development-v1.json"
)


def _case() -> dict[str, Any]:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return next(item for item in manifest["graph_cases"] if item["id"] == "revision_forget_replay")


def _vectors(case: dict[str, Any]) -> dict[str, list[float]]:
    vectors = {
        item["content"]: [1.0, float(index + 1), 2.0] for index, item in enumerate(case["facts"])
    }
    correction = case["lifecycle_actions"][0]["new_content"]
    vectors[correction] = [2.0, 3.0, 5.0]
    return vectors


def test_revision_forget_replay_uses_real_leased_transitions(tmp_path: Path) -> None:
    case = _case()
    db_path = tmp_path / "lifecycle.db"
    populate_source_association_fixture(case, _vectors(case), db_path)

    report = run_lifecycle_control(case, _vectors(case), db_path)

    assert report["controls"]["correction_pass"] is True
    assert report["controls"]["forget_pass"] is True
    assert report["controls"]["replay_suppression_pass"] is True
    assert report["controls"]["forgotten_fact_absent_after_replay"] is True
    assert report["controls"]["surviving_facts_preserved"] is True
    assert report["controls"]["restart_pass"] is True
    assert (
        report["steps"]["correct"]["revisions"]["revision_forget_replay-seal"][-1]["version"] == 2
    )
    assert (
        report["steps"]["correct"]["facts"][2]["content"]
        != report["steps"]["initial"]["facts"][2]["content"]
    )
    assert report["controls"]["source_preserved_pass"] is False
    assert report["controls"]["stale_old_source_relation_absent"] is True
    assert report["status"] == "completed_with_control_failures"


def test_replay_control_reports_suppression_without_reinsertion(tmp_path: Path) -> None:
    case = _case()
    db_path = tmp_path / "lifecycle.db"
    populate_source_association_fixture(case, _vectors(case), db_path)

    report = run_lifecycle_control(case, _vectors(case), db_path)
    replay = report["steps"]["replay"]
    forgotten_id = "revision_forget_replay-handoff"

    assert replay["facts"]
    assert next(item for item in replay["facts"] if item["id"] == forgotten_id)["present"] is False
    assert forgotten_id in replay["suppressions"]["fact_ids"]
    assert report["controls"]["shared_source_no_longer_relates_forgotten_fact"] is True


def test_authored_version_conflict_is_preserved_without_substitution(tmp_path: Path) -> None:
    case = _case()
    db_path = tmp_path / "lifecycle.db"
    populate_source_association_fixture(case, _vectors(case), db_path)
    case["lifecycle_actions"][0]["expected_version"] = 7
    with pytest.raises(ValueError, match="authored expected_version"):
        run_lifecycle_control(case, _vectors(case), db_path)
    failure = json.loads(db_path.with_suffix(".lifecycle-failed.json").read_text())
    assert failure["status"] == "failed"
    assert list(failure["steps"]) == ["initial"]
