"""Check the comparison controls without executing its retrieval cohort."""

import copy
from datetime import datetime
import json
from pathlib import Path

from benchmarks.legacy_chronology import MatchedEmbedder, SCOPE, projection
from mnemos.engine import _build_scope_filter
from mnemos.types import MemoryChunk
from mnemos.utils.storage import InMemoryStore

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks/development/legacy-chronology-v1.json"


def test_query_vectors_do_not_drift_with_order_or_repetition():
    fixture = json.loads(FIXTURE.read_bytes())
    provider = MatchedEmbedder(fixture)
    for queries in [fixture["queries"], list(reversed(fixture["queries"]))]:
        for query in queries:
            assert provider.embed(query["text"]) == query["embedding"]
    assert provider.calls == 16


def test_scope_controls_are_present_but_only_nine_records_are_eligible():
    fixture = json.loads(FIXTURE.read_bytes())
    allowed = _build_scope_filter(
        current_scope="project", scope_id=SCOPE, allowed_scopes=("project", "global")
    )
    expected = {doc["id"] for doc in fixture["source_documents"]} | {"global-control"}
    for rows in fixture["conditions"].values():
        assert len(rows) == 18 and len({row["id"] for row in rows}) == 18
        assert {row["id"] for row in rows if allowed(MemoryChunk.model_validate(row))} == expected
        assert sum(row["id"].startswith("foreign-") for row in rows) == 8


def test_access_projection_changes_only_activity_fields():
    fixture = json.loads(FIXTURE.read_bytes())
    store = InMemoryStore()
    for row in fixture["conditions"]["ordered"]:
        store.store(MemoryChunk.model_validate(row))
    before = projection(store)
    assert store.touch(
        "editor-old", updated_at=datetime.fromisoformat(fixture["prior_access_clock"])
    )
    after = projection(store)
    assert before["common_sha256"] != after["common_sha256"]
    assert before["revisions"] == after["revisions"]
    normalized = copy.deepcopy(after["common"])
    original = {row["id"]: row for row in before["common"]}
    for row in normalized:
        row["updated_at"] = original[row["id"]]["updated_at"]
        row["access_count"] = original[row["id"]]["access_count"]
    assert normalized == before["common"]


def test_fixture_withholds_temporal_correctness_for_indeterminate_conditions():
    fixture = json.loads(FIXTURE.read_bytes())
    assert fixture["determinate_conditions"] == ["ordered", "known-rewrite"]
    assert len(fixture["queries"]) == 8
    assert all(query["original_expected_ids"] for query in fixture["queries"])
    for row in fixture["conditions"]["unknown-rewrite"][:8]:
        assert row["content_updated_at"] is None
        if row["id"].endswith("-new"):
            assert row["version"] == 2
    assert len({row["content_updated_at"] for row in fixture["conditions"]["tied"][:8]}) == 1
