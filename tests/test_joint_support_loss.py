from collections import defaultdict
from copy import deepcopy
import math

import pytest

from benchmarks.local_memory.joint_support_corpus import assemble
from benchmarks.local_memory.joint_support_data import audit_groups
from benchmarks.local_memory.joint_support_loss import training_weights


def test_real_corpus_family_and_class_mass_and_target_accounting():
    examples, labels = assemble()
    weights = training_weights(examples, labels)
    training = {e["id"] for e in examples if e["split"] == "train"}
    expected = {(i, t["id"]) for i in training for t in labels[i]["targets"]}
    assert set(weights) == expected
    assert all(math.isfinite(w) and w > 0 for w in weights.values())
    groups = [g for g in audit_groups(examples, labels)["groups"] if g[0] in training]
    for group in groups:
        masses = defaultdict(float)
        for identity in group:
            for target in labels[identity]["targets"]:
                masses[target["supported"]] += weights[(identity, target["id"])]
        assert sum(masses.values()) == pytest.approx(1 / len(groups))
        for mass in masses.values():
            assert mass == pytest.approx(1 / len(groups) / len(masses))


def test_reserved_labels_cannot_change_training_mass():
    examples, labels = assemble()
    original = training_weights(examples, labels)
    changed = deepcopy(labels)
    for example in examples:
        if example["split"] != "train":
            for target in changed[example["id"]]["targets"]:
                target["supported"] = not target["supported"]
                target["reason"] = "supported" if target["supported"] else "wrong_subject_relation"
    assert training_weights(examples, changed) == original


def test_input_order_does_not_change_weights():
    examples, labels = assemble()
    assert training_weights(examples[::-1], labels) == training_weights(examples, labels)


def test_pending_label_is_rejected():
    examples, labels = assemble()
    labels[examples[0]["id"]]["review_status"] = "pending"
    with pytest.raises(ValueError, match="reviewed"):
        training_weights(examples, labels)


def test_negative_reasons_and_queries_have_equal_mass_within_family():
    examples, labels = assemble()
    before = deepcopy((examples, labels))
    weights = training_weights(examples, labels)
    train = {e["id"] for e in examples if e["split"] == "train"}
    for group in audit_groups(examples, labels)["groups"]:
        if group[0] not in train:
            continue
        by_reason = defaultdict(lambda: defaultdict(float))
        for identity in group:
            for target in labels[identity]["targets"]:
                if not target["supported"]:
                    by_reason[target["reason"]][identity] += weights[(identity, target["id"])]
        totals = [sum(queries.values()) for queries in by_reason.values()]
        assert totals == pytest.approx([totals[0]] * len(totals))
        for queries in by_reason.values():
            values = list(queries.values())
            assert values == pytest.approx([values[0]] * len(values))
    assert (examples, labels) == before
