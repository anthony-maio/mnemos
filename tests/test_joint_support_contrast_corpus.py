import hashlib
import json

import pytest

from benchmarks.local_memory.joint_support_contrast_corpus import (
    BASE_BINDINGS,
    write_contrast_corpus,
)
from benchmarks.local_memory.joint_support_corpus import write_corpus


def test_extension_preserves_base_and_registers_one_new_critical_fact(tmp_path):
    baseline, destination = tmp_path / "baseline", tmp_path / "new"
    write_corpus(baseline)
    saved = {name: (baseline / name).read_bytes() for name in BASE_BINDINGS}
    manifest = write_contrast_corpus(baseline, destination)
    assert manifest["examples"] == 127 and manifest["targets"] == 400
    assert manifest["status"] == "reviewed_draft_not_training_ready"
    for name, raw in saved.items():
        assert (baseline / name).read_bytes() == raw
        assert hashlib.sha256(raw).hexdigest() == BASE_BINDINGS[name]
    original_labels = json.loads(saved["labels.json"])
    new_labels = json.loads((destination / "labels.json").read_bytes())
    assert all(new_labels[k] == v for k, v in original_labels.items())
    inventory = json.loads((destination / "critical-inventory.json").read_bytes())
    assert inventory["registered_facts_by_partition"]["development"] == 3
    assert inventory["supported_critical_opportunities_by_partition"]["development"] == 11
    with pytest.raises(FileExistsError):
        write_contrast_corpus(baseline, destination)


def test_modified_baseline_fails_before_output(tmp_path):
    baseline, destination = tmp_path / "baseline", tmp_path / "new"
    write_corpus(baseline)
    with (baseline / "inputs.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="binding changed"):
        write_contrast_corpus(baseline, destination)
    assert not destination.exists()
