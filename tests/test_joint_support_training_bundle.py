from copy import deepcopy

import pytest

from benchmarks.local_memory.joint_support_contrasts import build_contrasts
from benchmarks.local_memory.joint_support_corpus import assemble
from benchmarks.local_memory.joint_support_training_bundle import build_bundle


def corpus():
    examples, labels = assemble()
    added, annotations = build_contrasts()
    return examples + added, {**labels, **annotations}


def test_bundle_contains_all_training_targets_but_no_reserved_labels():
    examples, labels = corpus()
    bundle = build_bundle(examples, labels, "fixed instruction")
    train = {e["id"] for e in examples if e["split"] == "train"}
    assert {r["id"] for r in bundle["training"]} == train
    assert sum(len(r["targets"]) for r in bundle["training"]) == 282
    assert sum(t["weight"] for r in bundle["training"] for t in r["targets"]) == pytest.approx(1)
    assert all(set(e) == {"id", "split", "request"} for e in bundle["examples"])
    assert len(bundle["baseline"]) == 48
    assert sum(len(q["requests"]) for q in bundle["baseline"]) == 109


def test_changing_reserved_decisions_does_not_change_bundle():
    examples, labels = corpus()
    original = build_bundle(examples, labels, "fixed instruction")
    changed = deepcopy(labels)
    for example in examples:
        if example["split"] != "train":
            for target in changed[example["id"]]["targets"]:
                target["supported"] = not target["supported"]
                target["reason"] = "supported" if target["supported"] else "wrong_subject_relation"
    assert build_bundle(examples, changed, "fixed instruction") == original


def test_label_order_cannot_reassign_targets_or_weights():
    examples, labels = corpus()
    original = build_bundle(examples, labels, "fixed instruction")
    for annotation in labels.values():
        annotation["targets"].reverse()
    assert build_bundle(examples, labels, "fixed instruction") == original
