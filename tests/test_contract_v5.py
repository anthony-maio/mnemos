from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.contracts import load_dataset

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl"
)


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    result = [json.loads(line) for line in SOURCE.read_bytes().splitlines()]
    result[0]["dataset_version"] = "local-memory-heldout-v5"
    result[0][
        "description"
    ] = "Contract tests reuse v4 rows; these are not a fresh evaluation corpus."
    return result


def _load(tmp_path: Path, rows: list[dict[str, Any]]) -> Any:
    path = tmp_path / "contract.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return load_dataset(path)


def test_v5_reuses_strict_checks_without_v4_query_names(
    rows: list[dict[str, Any]], tmp_path: Path
) -> None:
    session = next(row for row in rows[1:] if row["session_id"] == "heldout-061")
    for index, query in enumerate(session["queries"]):
        query["query_id"] = f"new-canary-name-{index}"
    dataset = _load(tmp_path, rows)
    assert len(dataset.heldout_sessions) == 200
    assert len(dataset.state_pairs) == 50
    assert dataset.version == "local-memory-heldout-v5"
    rows[0]["dataset_version"] = "local-memory-heldout-v4"
    with pytest.raises(ValueError, match="missing paired scope canary queries"):
        _load(tmp_path, rows)


@pytest.mark.parametrize(
    "field,id_field", [("events", "event_id"), ("queries", "query_id"), ("state_pairs", "pair_id")]
)
def test_v5_rejects_cross_split_identity_collision(
    rows: list[dict[str, Any]], tmp_path: Path, field: str, id_field: str
) -> None:
    development = next(row for row in rows[1:] if row["split"] == "development")
    heldout = next(row for row in rows[1:] if row["split"] == "heldout" and row[field])
    if not development[field]:
        development[field] = [dict(heldout[field][0])]
    else:
        development[field][0][id_field] = heldout[field][0][id_field]
    with pytest.raises(ValueError, match="globally unique"):
        _load(tmp_path, rows)


@pytest.mark.parametrize("operation", ["correct", "forget"])
def test_v5_requires_lifecycle_targets(
    rows: list[dict[str, Any]], tmp_path: Path, operation: str
) -> None:
    event = next(
        event
        for row in rows[1:]
        for event in row["events"]
        if event["lifecycle_operation"] == operation
    )
    event["target_source_event_id"] = None
    with pytest.raises(ValueError, match="lifecycle target metadata is required"):
        _load(tmp_path, rows)


def test_v5_rejects_foreign_answerable_label(rows: list[dict[str, Any]], tmp_path: Path) -> None:
    query = next(q for row in rows[1:] for q in row["queries"] if q["kind"] == "answerable")
    query["target_user_profile_id"] = "foreign-profile"
    with pytest.raises(ValueError, match="outside source scope"):
        _load(tmp_path, rows)


def test_v5_rejects_explicit_fact_body_mismatch(rows: list[dict[str, Any]], tmp_path: Path) -> None:
    for row in rows[1:]:
        facts = {fact["fact_id"]: fact for fact in row["facts"]}
        for event in row["events"]:
            if event["lifecycle_operation"] == "remember":
                supported = [
                    facts[i]
                    for i in event["evidence_fact_ids"]
                    if facts[i]["gold_status"] == "supported"
                ]
                if supported:
                    supported[0]["content"] = "A different unsupported body."
                    with pytest.raises(ValueError, match="must equal its explicit remember body"):
                        _load(tmp_path, rows)
                    return
    pytest.fail("fixture needs an explicit supported fact")


def test_v5_rejects_unmatched_state_pair(rows: list[dict[str, Any]], tmp_path: Path) -> None:
    pair = next(pair for row in rows[1:] for pair in row["state_pairs"])
    pair["matched_content"] = "Unrelated unsupported state target."
    with pytest.raises(ValueError, match="state target must match"):
        _load(tmp_path, rows)


def test_v5_retains_minimum_coverage(rows: list[dict[str, Any]], tmp_path: Path) -> None:
    index = next(i for i, row in enumerate(rows) if row.get("split") == "heldout")
    rows.pop(index)
    with pytest.raises(ValueError, match="held-out sessions below minimum"):
        _load(tmp_path, rows)
