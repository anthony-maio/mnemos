import copy

import pytest

from benchmarks.local_memory.inspection_batch_codec import decode, encode, serialized


def test_full_inspections_round_trip_without_mutating_input():
    text = 'Repeated source text with quotes " and Unicode \u2603' * 8
    original = [
        dict(
            fact=dict(id="fact-1", version=2, content=text),
            evidence=[dict(excerpt=text)],
            revisions=[dict(content=text, version=1)],
        )
    ]
    before = copy.deepcopy(original)
    encoded = encode(original)
    assert decode(encoded) == original == before
    assert len(serialized(encoded)) < len(serialized(original))
    assert encoded["strings"] == [text]


def test_reserved_user_metadata_is_literal_not_an_injected_reference():
    original = {
        "$ref": 0,
        "$object": [["x", "y"]],
        "nested": [{"$ref": "outside"}],
        "types": [True, False, None, 1, 1.5],
    }
    assert decode(encode(original)) == original


@pytest.mark.parametrize("value", [-1, True, 1, "0", None])
def test_malformed_references_rejected(value):
    with pytest.raises(ValueError):
        decode(dict(format="inspection_strings_v1", strings=["text"], results={"$ref": value}))


def test_escaped_duplicate_keys_cannot_overwrite_fields():
    with pytest.raises(ValueError):
        decode(
            dict(
                format="inspection_strings_v1",
                strings=[],
                results={"$object": [["id", "a"], ["id", "b"]]},
            )
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "not a JSON key"}, {1, 2}])
def test_non_json_input_rejected(value):
    with pytest.raises(ValueError):
        encode(value)


def test_distinct_identical_length_strings_are_not_combined():
    original = ["original" * 20, "corrected" * 20]
    assert encode(original)["strings"] == []
    assert decode(encode(original)) == original
