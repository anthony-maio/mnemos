import json

import pytest

from benchmarks.local_memory.joint_support_training_audit import rows


def test_duplicate_outputs_cannot_be_hidden_by_dictionary_conversion(tmp_path):
    path = tmp_path / "output.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(dict(id="same", status=status)) for status in ("unavailable", "assessed")
        )
    )
    with pytest.raises(ValueError, match="Duplicate"):
        rows(path, {"same"})


def test_unknown_output_cannot_replace_missing_case(tmp_path):
    path = tmp_path / "output.jsonl"
    path.write_text(json.dumps(dict(id="unknown", status="assessed")))
    with pytest.raises(ValueError, match="unknown"):
        rows(path, {"expected"})


def test_absent_output_file_remains_empty_for_unavailable_accounting(tmp_path):
    assert rows(tmp_path / "missing.jsonl", {"expected"}) == {}
