from copy import deepcopy
import json

import pytest

from benchmarks.local_memory.joint_support_baseline import baseline_requests, select_joint
from benchmarks.local_memory.joint_support_corpus import assemble
from benchmarks.local_memory.joint_support_contrasts import build_contrasts


def test_mapping_round_trips_all_owned_sources_and_preserves_order():
    examples, _ = assemble()
    added, _ = build_contrasts()
    for example in examples + added:
        request = example["request"]
        before = deepcopy(request)
        rows = baseline_requests(request, "fixed instruction")
        expected = [(c["id"], s["id"]) for c in request["candidates"] for s in c["sources"]]
        assert [(r["fact_id"], r["evidence_id"]) for r in rows] == expected
        for row in rows:
            document = json.loads(row["document"])
            assert row["query"] == request["query"]
            assert document["reference_date"] == request["reference_date"]
            assert [c["id"] for c in document["candidates"]] == [
                c["id"] for c in request["candidates"]
            ]
            for mapped, original in zip(document["candidates"], request["candidates"], strict=True):
                assert {k: mapped[k] for k in ("id", "version", "content", "assertion_status")} == {
                    k: original[k] for k in ("id", "version", "content", "assertion_status")
                }
                for actual, source in zip(mapped["sources"], original["sources"], strict=True):
                    assert actual == dict(
                        evidence_id=source["id"],
                        fact_id=original["id"],
                        source_event_id=source["source_event_id"],
                        excerpt=source["excerpt"],
                    )
            target = next(c for c in document["candidates"] if c["id"] == row["fact_id"])
            assert document["target_evidence"] in target["sources"]
        assert request == before


def request():
    return deepcopy(assemble()[0][0]["request"])


def test_missing_source_is_not_replaced_by_fact_text():
    value = request()
    for c in value["candidates"]:
        c["sources"] = []
    assert baseline_requests(value, "instruction") == []
    assert (
        select_joint(
            value, [c["id"] for c in value["candidates"]], [99.0] * len(value["candidates"])
        )
        == []
    )


def test_cross_owner_source_collision_is_preserved():
    value = request()
    for c in value["candidates"]:
        c["sources"][0]["id"] = "shared-id"
    rows = baseline_requests(value, "instruction")
    assert len(rows) == len(value["candidates"])
    assert len({(r["fact_id"], r["evidence_id"]) for r in rows}) == len(rows)


def test_overflow_rejects_entire_question():
    value = request()
    for c in value["candidates"]:
        c["sources"] = [dict(id=str(i), source_event_id=str(i), excerpt="source") for i in range(8)]
    with pytest.raises(ValueError, match="Entire baseline"):
        baseline_requests(value, "instruction")


def test_joint_decoder_rejects_wrong_order_nonfinite_and_uses_fixed_boundary():
    value = request()
    ids = [c["id"] for c in value["candidates"]]
    with pytest.raises(ValueError, match="ordered"):
        select_joint(value, ids[::-1], [0.0] * len(ids))
    with pytest.raises(ValueError, match="Nonfinite"):
        select_joint(value, ids, [float("nan")] * len(ids))
    assert select_joint(value, ids, [0.0] * len(ids)) == ids
    assert select_joint(value, ids, [-0.001] * len(ids)) == []
