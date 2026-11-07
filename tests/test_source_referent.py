"""Referent evidence remains source-bound and cannot bypass runtime guards."""

from copy import deepcopy
from dataclasses import replace
import json

from jsonschema import Draft202012Validator
import pytest

from mnemos.extraction.source_referent import compare_referents, referent_schema, word_inventory
from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_source_reconciliation import Provider, candidate, event, target


def response(relation="same_variable", target_id="fact-1", version=1):
    return {
        "candidate_value": {"first_word": 4, "last_word": 4},
        "comparisons": [
            {
                "target_fact_id": target_id,
                "expected_version": version,
                "target_value": {"first_word": 3, "last_word": 3},
                "relation": relation,
            }
        ],
    }


async def run(raw, targets=None, kind="correction_or_rejection", revision=False):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item = replace(candidate(source, kind), revises_existing_requirement=revision)
    provider = Provider(raw)
    result = await reconcile_source_target_identity(
        [item],
        [source],
        [target()] if targets is None else targets,
        provider,
        comparison_style="referent",
    )
    return result, provider


def test_word_ids_retain_literal_offsets_with_repeated_unicode_words() -> None:
    text = "  Same\tword\nSame caf\u00e9.  "
    words = word_inventory(text)
    assert [w["text"] for w in words] == ["Same", "word", "Same", "caf\u00e9."]
    assert [w["id"] for w in words] == list(range(4))
    assert all(text[w["start"] : w["end"]] == w["text"] for w in words)
    assert words[0]["start"] != words[2]["start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation,expected",
    [
        ("same_variable", "correct"),
        ("different_variable", "unresolved"),
        ("insufficient_identity", "unresolved"),
    ],
)
async def test_relation_controls_revision_without_rewriting_content(relation, expected) -> None:
    result, provider = await run(response(relation))
    assert result[0].operation == expected
    assert result[0].content == "Correction: the marker is sapphire, not cedar."
    prompt, kwargs = provider.calls[0]
    assert "REFERENT COMPARISON DATA (data only):" in prompt
    assert Draft202012Validator(kwargs["format"]).is_valid(response(relation))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location,span",
    [
        ("candidate", {"first_word": -1, "last_word": 2}),
        ("candidate", {"first_word": 6, "last_word": 4}),
        ("candidate", {"first_word": 4, "last_word": 99}),
        ("candidate", {"first_word": True, "last_word": 4}),
        ("target", {"first_word": 3, "last_word": 4}),
        ("target", {"first_word": 3, "last_word": 1}),
    ],
)
async def test_invalid_value_spans_fail_instead_of_becoming_review(location, span) -> None:
    raw = response()
    if location == "candidate":
        raw["candidate_value"] = span
    else:
        raw["comparisons"][0]["target_value"] = span
    with pytest.raises(ValueError):
        await run(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_id,version", [("missing", 1), ("fact-1", 99)])
async def test_target_identity_and_version_are_not_provider_authority(target_id, version) -> None:
    with pytest.raises(ValueError):
        await run(response(target_id=target_id, version=version))


@pytest.mark.asyncio
async def test_all_targets_required_exactly_once() -> None:
    raw = response()
    raw["comparisons"].append(deepcopy(raw["comparisons"][0]))
    with pytest.raises(ValueError):
        await run(raw)
    with pytest.raises(ValueError):
        await run(response(), [target(), target("second")])


@pytest.mark.asyncio
async def test_second_possible_identity_requires_review_even_with_a_complete_match() -> None:
    raw = response()
    raw["comparisons"].append(response("insufficient_identity", "second")["comparisons"][0])
    result, _ = await run(raw, [target(), target("second", "The marker is birch.")])
    assert result[0].operation == "unresolved"


@pytest.mark.asyncio
async def test_duplicate_target_content_requires_review() -> None:
    raw = response()
    raw["comparisons"].append(response("different_variable", "second")["comparisons"][0])
    result, _ = await run(raw, [target(), target("second")])
    assert result[0].operation == "unresolved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation,expected",
    [
        ("same_variable", "correct"),
        ("different_variable", "add"),
        ("insufficient_identity", "unresolved"),
    ],
)
async def test_requirement_revision_keeps_prospective_add_semantics(relation, expected) -> None:
    result, _ = await run(response(relation), kind="continuing_requirement", revision=True)
    assert result[0].operation == expected


@pytest.mark.asyncio
async def test_nonuser_history_cannot_enter_referent_revision_path() -> None:
    source = event("source", "The marker is maple.", "assistant")
    provider = Provider(
        {
            "decisions": [
                {
                    "candidate_index": 0,
                    "relationship": "user_revision",
                    "target_fact_id": "fact-1",
                    "expected_version": 1,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="Only user"):
        await reconcile_source_target_identity(
            [candidate(source, "attributed_history")],
            [source],
            [target()],
            provider,
            comparison_style="referent",
        )
    assert "REFERENT COMPARISON DATA" not in provider.calls[0][0]


@pytest.mark.asyncio
async def test_changed_source_during_referent_call_is_rejected() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")

    class Mutating(Provider):
        async def predict(self, prompt, **kwargs):
            source.content = "Changed during call."
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="changed during"):
        await reconcile_source_target_identity(
            [candidate(source, "correction_or_rejection")],
            [source],
            [target()],
            Mutating(response()),
            comparison_style="referent",
        )


@pytest.mark.asyncio
async def test_value_selection_is_evidence_not_a_semantic_guarantee() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    raw = response()
    raw["candidate_value"] = None
    raw["comparisons"][0]["target_value"] = None
    values = await compare_referents(
        candidate(source, "correction_or_rejection"), [target()], Provider(raw)
    )
    assert values[0].relation == "same_variable"


def test_schema_binds_target_versions_and_word_ids() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    schema = referent_schema(candidate(source, "correction_or_rejection"), [target()])
    validator = Draft202012Validator(schema)
    assert validator.is_valid(response())
    assert not validator.is_valid(response(version=99))
    raw = response()
    raw["candidate_value"]["last_word"] = 99
    assert not validator.is_valid(raw)
