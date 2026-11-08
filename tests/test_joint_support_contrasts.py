from copy import deepcopy

from benchmarks.local_memory.joint_support_contrasts import build_contrasts
from benchmarks.local_memory.joint_support_corpus import assemble
from benchmarks.local_memory.joint_support_data import audit_groups, request_digest


def test_all_new_cases_bind_labels_and_keep_cohort_limit():
    examples, labels = build_contrasts()
    assert len(examples) == len(labels) == 15
    for example in examples:
        assert example["split"] == "development"
        assert len(example["request"]["candidates"]) <= 5
        assert labels[example["id"]]["request_sha256"] == request_digest(example["request"])


def test_new_families_do_not_cross_original_partitions():
    old, original = assemble()
    before = deepcopy((old, original))
    added, annotations = build_contrasts()
    audit = audit_groups(old + added, {**original, **annotations})
    assert audit["examples"] == {"train": 67, "tuning": 12, "development": 48}
    assert audit["independent_groups"] == {"train": 3, "tuning": 3, "development": 9}
    assert audit["training_ready"] is False
    assert (old, original) == before


def test_quorum_rule_content_and_owned_sources_stay_fixed_across_queries():
    examples, _ = build_contrasts()
    rules = [
        e["request"]["candidates"][0] for e in examples if e["id"].startswith("contrast-quorum")
    ]
    assert len(rules) == 9
    assert all(rule == rules[0] for rule in rules)


def test_permutation_keeps_labels_attached_to_exact_target():
    examples, labels = build_contrasts()
    rows = {e["id"]: e for e in examples}
    base = "contrast-negated-report-denied-positive-report"
    permuted = "contrast-negated-report-denial-permutation"
    assert rows[base]["request"]["candidates"] == rows[permuted]["request"]["candidates"][::-1]
    assert labels[base]["targets"] == labels[permuted]["targets"][::-1]
