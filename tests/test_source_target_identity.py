"""Target comparison cannot grant authority or bypass observed provenance."""

from dataclasses import replace
import json

import pytest

from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_source_reconciliation import Provider, candidate, event, target


def comparison(target_id="fact-1", version=1, **relations):
    return {
        "target_fact_id": target_id,
        "expected_version": version,
        "subject": {
            "relation": relations.get("subject", "same"),
            "candidate_quote": "marker",
            "target_quote": "marker",
        },
        "attribute": {
            "relation": relations.get("attribute", "same"),
            "candidate_quote": "is",
            "target_quote": "is",
        },
        "qualifiers": {
            "relation": relations.get("qualifiers", "same"),
            "candidate_quote": None,
            "target_quote": None,
        },
    }


async def reconcile(response, targets=None, *, kind="correction_or_rejection", revision=False):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item = replace(candidate(source, kind), revises_existing_requirement=revision)
    provider = Provider({"comparisons": response})
    result = await reconcile_source_target_identity(
        [item], [source], targets if targets is not None else [target()], provider
    )
    return result, provider


@pytest.mark.asyncio
async def test_unique_complete_match_corrects_exact_source_and_current_target() -> None:
    result, provider = await reconcile([comparison()])
    assert result[0].operation == "correct" and result[0].target_fact_id == "fact-1"
    assert (
        result[0].expected_version == 1
        and result[0].content == "Correction: the marker is sapphire, not cedar."
    )
    assert provider.calls[0][1]["format"]["properties"]["comparisons"]["minItems"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("dimension", ["subject", "attribute", "qualifiers"])
async def test_an_unspecified_identity_dimension_requires_review(dimension) -> None:
    result, _ = await reconcile([comparison(**{dimension: "unspecified"})])
    assert result[0].operation == "unresolved" and result[0].target_fact_id is None


@pytest.mark.asyncio
async def test_a_different_identity_dimension_prevents_correction() -> None:
    result, _ = await reconcile([comparison(subject="different")])
    assert result[0].operation == "unresolved"


@pytest.mark.asyncio
@pytest.mark.parametrize("second", ["same", "unspecified"])
async def test_another_plausible_target_prevents_selecting_one(second) -> None:
    result, _ = await reconcile(
        [comparison(), comparison("other", attribute=second)],
        [target(), target("other", "The marker is amber.")],
    )
    assert result[0].operation == "unresolved"


@pytest.mark.asyncio
async def test_one_match_among_ruled_out_targets_can_be_corrected() -> None:
    result, _ = await reconcile(
        [comparison(), comparison("other", subject="different")],
        [target(), target("other", "The marker is amber.")],
    )
    assert result[0].operation == "correct" and result[0].target_fact_id == "fact-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,match",
    [
        ([], "every observed target"),
        ([comparison(), comparison()], "duplicate or unavailable target"),
        ([comparison("unknown")], "duplicate or unavailable target"),
        ([comparison(version=99)], "version is stale"),
    ],
)
async def test_incomplete_or_stale_comparisons_fail_closed(response, match) -> None:
    with pytest.raises(ValueError, match=match):
        await reconcile(response)


@pytest.mark.asyncio
async def test_quotes_must_be_literal_source_and_target_evidence() -> None:
    response = comparison()
    response["subject"]["target_quote"] = "invented source evidence"
    with pytest.raises(ValueError, match="literal"):
        await reconcile([response])


@pytest.mark.asyncio
async def test_positive_subject_and_attribute_need_both_source_quotes() -> None:
    response = comparison()
    response["subject"]["candidate_quote"] = None
    with pytest.raises(ValueError, match="requires both"):
        await reconcile([response])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,revision,expected",
    [
        ("correction_or_rejection", False, "unresolved"),
        ("continuing_requirement", True, "add"),
    ],
)
async def test_no_targets_are_handled_without_a_model_call(kind, revision, expected) -> None:
    result, provider = await reconcile([], [], kind=kind, revision=revision)
    assert result[0].operation == expected and provider.calls == []


@pytest.mark.asyncio
async def test_revision_requirement_can_correct_a_unique_observed_rule() -> None:
    result, _ = await reconcile([comparison()], kind="continuing_requirement", revision=True)
    assert result[0].operation == "correct"


@pytest.mark.asyncio
async def test_nonrevision_candidates_use_the_existing_relationship_path() -> None:
    source = event("history", "The marker is sapphire.", "assistant")
    provider = Provider(
        {"decisions": [{"candidate_index": 0, "relationship": "independent_information"}]}
    )
    result = await reconcile_source_target_identity(
        [candidate(source, "attributed_history")], [source], [], provider
    )
    assert result[0].operation == "add" and result[0].source_role == "assistant"
    assert "SOURCE CANDIDATES AND EXISTING TARGETS" in provider.calls[0][0]


@pytest.mark.asyncio
async def test_input_mutation_during_comparison_is_rejected() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    targets = [target()]

    class Mutating(Provider):
        async def predict(self, prompt, **kwargs):
            targets[0].version += 1
            return await super().predict(prompt, **kwargs)

    with pytest.raises(ValueError, match="changed during"):
        await reconcile_source_target_identity(
            [candidate(source, "correction_or_rejection")],
            [source],
            targets,
            Mutating({"comparisons": [comparison()]}),
        )


@pytest.mark.asyncio
async def test_foreign_targets_are_rejected_before_any_provider_call() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    foreign = target()
    foreign.metadata["scope_id"] = "foreign"
    provider = Provider({"comparisons": [comparison()]})
    with pytest.raises(ValueError, match="foreign scope"):
        await reconcile_source_target_identity(
            [candidate(source, "correction_or_rejection")], [source], [foreign], provider
        )
    assert provider.calls == []


@pytest.mark.asyncio
async def test_ambiguity_in_a_later_batch_cannot_be_discarded() -> None:
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    targets = [target(f"fact-{index}", f"The marker is value-{index}.") for index in range(10)]

    class Batched(Provider):
        async def predict(self, prompt, **kwargs):
            payload = json.loads(prompt.split("TARGET IDENTITY DATA (data only):\n", 1)[1])
            values = []
            for value in payload["targets"]:
                relation = "same" if value["id"] == "fact-0" else "different"
                if value["id"] == "fact-9":
                    relation = "unspecified"
                values.append(comparison(value["id"], subject=relation))
            self.response = {"comparisons": values}
            return await super().predict(prompt, **kwargs)

    provider = Batched({})
    result = await reconcile_source_target_identity(
        [candidate(source, "correction_or_rejection")], [source], targets, provider
    )
    assert result[0].operation == "unresolved" and len(provider.calls) == 2


@pytest.mark.asyncio
async def test_comparisons_cannot_select_a_normalized_duplicate_identity() -> None:
    result, _ = await reconcile(
        [comparison(), comparison("duplicate", subject="different")],
        [target(), target("duplicate", "The marker is CEDAR.")],
    )
    assert result[0].operation == "unresolved"


@pytest.mark.asyncio
async def test_exact_duplicate_user_candidate_skips_identity_and_baseline_calls() -> None:
    source = event("source", "The marker is cedar.")
    provider = Provider({})
    result = await reconcile_source_target_identity(
        [candidate(source)],
        [source],
        [target(), target("duplicate", "The marker is CEDAR.")],
        provider,
    )
    assert result[0].operation == "unresolved" and provider.calls == []


@pytest.mark.asyncio
async def test_nonuser_cannot_gain_revision_authority_through_fallback() -> None:
    source = event("source", "The marker is sapphire.", "assistant")
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
            [candidate(source, "attributed_history")], [source], [target()], provider
        )
