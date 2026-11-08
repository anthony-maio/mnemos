import json

import pytest

from mnemos.extraction.source_qualifier_coverage import compare_referents_with_qualifier_coverage
from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_pairwise_referent import response
from tests.test_source_reconciliation import Provider, candidate, event, target


class CoverageProvider(Provider):
    def __init__(self, relation="same_variable", coverage="covered"):
        super().__init__({})
        self.relation, self.coverage = relation, coverage

    async def predict(self, prompt, **kwargs):
        self.response = (
            {"coverage": self.coverage}
            if "QUALIFIER COVERAGE DATA" in prompt
            else response(self.relation)
        )
        return await super().predict(prompt, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation", ["same_variable", "different_variable", "insufficient_identity"]
)
@pytest.mark.parametrize("coverage", ["covered", "incompatible", "unspecified"])
async def test_qualifier_result_can_only_withhold_complete_matches(relation, coverage):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    provider = CoverageProvider(relation, coverage)
    values = await compare_referents_with_qualifier_coverage(
        candidate(source), [target()], provider
    )
    expected = (
        "insufficient_identity"
        if relation == "same_variable" and coverage != "covered"
        else relation
    )
    assert values[0].relation == expected
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_missing_condition_yields_review_not_addition_or_correction():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    values = await reconcile_source_target_identity(
        [candidate(source, "correction_or_rejection")],
        [source],
        [target()],
        CoverageProvider(coverage="unspecified"),
        comparison_style="qualifier_coverage",
    )
    assert len(values) == 1 and values[0].operation == "unresolved"


@pytest.mark.asyncio
async def test_every_pair_checked_and_storage_ids_kept_out_of_qualifier_prompt():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    first, second = target("private-1"), target("private-2", "The marker is birch.")
    first.version = 19
    provider = CoverageProvider(relation="different_variable")
    values = await compare_referents_with_qualifier_coverage(
        candidate(source), [first, second], provider
    )
    assert [(v.target_fact_id, v.expected_version) for v in values] == [
        ("private-1", 19),
        ("private-2", 1),
    ]
    assert len(provider.calls) == 4
    prompts = [p for p, _ in provider.calls if "QUALIFIER COVERAGE DATA" in p]
    assert len(prompts) == 2 and all("private-" not in p for p in prompts)
    assert all(
        set(json.loads(p.split("QUALIFIER COVERAGE DATA (data only):\n")[1]))
        == {"candidate", "target"}
        for p in prompts
    )


@pytest.mark.asyncio
async def test_failed_later_coverage_request_returns_no_partial_result():
    class Failing(CoverageProvider):
        async def predict(self, prompt, **kwargs):
            if len(self.calls) == 3:
                raise RuntimeError("second coverage unavailable")
            return await super().predict(prompt, **kwargs)

    source = event("source", "Correction: the marker is sapphire, not cedar.")
    with pytest.raises(RuntimeError, match="second coverage"):
        await compare_referents_with_qualifier_coverage(
            candidate(source), [target(), target("two")], Failing()
        )


@pytest.mark.asyncio
async def test_foreign_target_rejected_before_any_provider_call():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    foreign = target()
    foreign.metadata["scope_id"] = "another-project"
    provider = CoverageProvider()
    with pytest.raises(ValueError):
        await reconcile_source_target_identity(
            [candidate(source)],
            [source],
            [foreign],
            provider,
            comparison_style="qualifier_coverage",
        )
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw", ["{}", '{"coverage":"yes"}', '{"coverage":"covered","target":"invented"}']
)
async def test_malformed_coverage_never_accepts_match(raw):
    class Invalid(CoverageProvider):
        async def predict(self, prompt, **kwargs):
            if "QUALIFIER COVERAGE DATA" in prompt:
                return raw
            return await super().predict(prompt, **kwargs)

    source = event("source", "Correction: the marker is sapphire, not cedar.")
    with pytest.raises(ValueError):
        await compare_referents_with_qualifier_coverage(candidate(source), [target()], Invalid())
