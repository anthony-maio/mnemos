from copy import deepcopy

import pytest

from benchmarks.local_memory.joint_support_critical import apply_critical_registry, fact_binding
from benchmarks.local_memory.joint_support_data import request_digest
from benchmarks.local_memory.joint_support_expansion import build_expansion


def fixtures():
    examples, labels = build_expansion()
    examples = [e for e in examples if e["id"].startswith("expansion-negative-prerequisite")]
    labels = {e["id"]: labels[e["id"]] for e in examples}
    candidate = examples[0]["request"]["candidates"][0]
    registry = dict(
        criterion="critical-action-constraint-v1",
        entries=[
            dict(
                target_id=candidate["id"],
                fact_sha256=fact_binding(candidate),
                justification="Preserves the interlock condition on transmission.",
            )
        ],
    )
    return examples, labels, registry


def test_counts_unique_fact_separately_from_applicable_queries():
    examples, labels, registry = fixtures()
    saved = deepcopy(labels)
    annotated, report = apply_critical_registry(examples, labels, registry)
    assert labels == saved
    assert report["registered_facts_by_partition"] == {"train": 1}
    assert report["supported_critical_opportunities_by_partition"] == {"train": 3}
    unknown = "expansion-negative-prerequisite-unknown"
    assert annotated[unknown]["targets"][0]["critical"]
    assert unknown not in report["facts"]["relay-rule"]["supported_opportunities"]
    assert not report["measured"]


@pytest.mark.parametrize("change", ["evidence", "version", "content", "status"])
def test_registry_cannot_transfer_review_after_fact_or_evidence_changes(change):
    examples, labels, registry = fixtures()
    candidate = examples[0]["request"]["candidates"][0]
    if change == "evidence":
        candidate["sources"][0]["excerpt"] = "The earlier interlock rule was withdrawn."
    elif change == "version":
        candidate["version"] = 2
    elif change == "content":
        candidate["content"] = "Relay R may transmit without an interlock."
    else:
        candidate["assertion_status"] = "unknown"
    labels[examples[0]["id"]]["request_sha256"] = request_digest(examples[0]["request"])
    with pytest.raises(ValueError, match="identity/content/evidence changed"):
        apply_critical_registry(examples, labels, registry)


def test_missing_registered_fact_cannot_silently_reduce_denominator():
    examples, labels, registry = fixtures()
    registry["entries"][0]["target_id"] = "missing"
    with pytest.raises(ValueError, match="lacks an applicable"):
        apply_critical_registry(examples, labels, registry)


@pytest.mark.parametrize("review_status", ["pending", "disputed"])
def test_corpus_manifest_cannot_claim_reviewed_when_any_case_is_unreviewed(
    tmp_path, monkeypatch, review_status
):
    from benchmarks.local_memory import joint_support_corpus as corpus

    examples, labels = corpus.assemble()
    labels[examples[0]["id"]]["review_status"] = review_status
    monkeypatch.setattr(corpus, "assemble", lambda: (examples, labels))
    manifest = corpus.write_corpus(tmp_path / "corpus")
    assert manifest["status"] == "unreviewed_draft_not_training_ready"
