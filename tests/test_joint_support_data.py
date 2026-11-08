from copy import deepcopy

import pytest

from benchmarks.local_memory.joint_support_data import audit_groups, request_digest
from tests.test_joint_support_frame import request


def corpus():
    rows, labels = [], {}
    for i, split in enumerate(("train", "tuning", "development")):
        value = request(1)
        value["query"] = f"Which region for project {i}?"
        value["candidates"][0]["content"] = f"Project {i} uses region {i}."
        value["candidates"][0]["sources"][0]["excerpt"] = f"Project {i} uses region {i}."
        identity = f"example-{i}"
        rows.append(
            dict(
                id=identity,
                split=split,
                known_material=False,
                request=value,
                lineage={
                    k: [f"{k}-{i}"]
                    for k in ("scenario", "source", "template", "paraphrase", "counterfactual")
                },
            )
        )
        labels[identity] = dict(
            request_sha256=request_digest(value),
            review_status="pending",
            targets=[
                dict(
                    id="fact-0",
                    supported=True,
                    reason="supported",
                    critical=True,
                    rationale="The target's own source states the requested relation.",
                )
            ],
        )
    return rows, labels


def test_integrity_pass_does_not_claim_semantic_quality_or_training_readiness():
    rows, labels = corpus()
    before = deepcopy((rows, labels))
    result = audit_groups(rows, labels)
    assert result["independent_groups"] == {"train": 1, "tuning": 1, "development": 1}
    assert result["training_ready"] is False and result["semantic_labels_verified"] is False
    assert result["minimum_partition_checks"] is False
    assert (rows, labels) == before


@pytest.mark.parametrize("kind", ["scenario", "source", "template", "paraphrase", "counterfactual"])
def test_declared_family_cannot_cross_split(kind):
    rows, labels = corpus()
    rows[1]["lineage"][kind] = rows[0]["lineage"][kind]
    with pytest.raises(ValueError, match="crosses splits"):
        audit_groups(rows, labels)


def test_transitive_family_connections_stay_in_one_partition():
    rows, labels = corpus()
    rows[1]["split"] = "train"
    rows[1]["lineage"]["source"] = rows[0]["lineage"]["source"]
    rows[2]["lineage"]["template"] = rows[1]["lineage"]["template"]
    with pytest.raises(ValueError, match="example-0, example-1, example-2"):
        audit_groups(rows, labels)


def test_renamed_source_with_spacing_and_case_changes_still_connects():
    rows, labels = corpus()
    rows[1]["request"]["candidates"][0]["sources"][0]["excerpt"] = "  PROJECT 0  USES REGION 0. "
    labels[rows[1]["id"]]["request_sha256"] = request_digest(rows[1]["request"])
    with pytest.raises(ValueError, match="crosses splits"):
        audit_groups(rows, labels)


def test_same_split_augmentation_counts_one_independent_group():
    rows, labels = corpus()
    rows[1]["split"] = "train"
    rows[1]["lineage"]["scenario"] = rows[0]["lineage"]["scenario"]
    result = audit_groups(rows, labels)
    assert result["examples"]["train"] == 2
    assert result["independent_groups"]["train"] == 1


def test_identical_inputs_cannot_receive_contradictory_gold_labels():
    rows, labels = corpus()
    rows[1]["split"] = "train"
    rows[1]["request"] = deepcopy(rows[0]["request"])
    labels["example-1"]["request_sha256"] = request_digest(rows[1]["request"])
    labels["example-1"]["targets"][0].update(supported=False, reason="wrong_subject_relation")
    with pytest.raises(ValueError, match="Contradictory labels"):
        audit_groups(rows, labels)


@pytest.mark.parametrize(
    "defect", ["stale", "orphan", "missing", "duplicate_target", "reason", "known", "acceptance"]
)
def test_label_and_partition_contracts_fail_closed(defect):
    rows, labels = corpus()
    if defect == "stale":
        rows[0]["request"]["query"] += " Changed."
    elif defect == "orphan":
        labels["extra"] = deepcopy(labels["example-0"])
    elif defect == "missing":
        del labels["example-0"]
    elif defect == "duplicate_target":
        labels["example-0"]["targets"] *= 2
    elif defect == "reason":
        labels["example-0"]["targets"][0]["reason"] = "unresolved_condition"
    elif defect == "known":
        rows[1]["known_material"] = True
    else:
        rows[1]["split"] = "acceptance"
    with pytest.raises(ValueError):
        audit_groups(rows, labels)
