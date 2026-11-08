import json

import pytest

from benchmarks.local_memory.referent_evaluation import CapturedReferents, grade_pairwise_relations
from mnemos.extraction.source_qualifier_coverage import compare_referents_with_identity_coverage
from mnemos.extraction.source_target_identity import reconcile_source_target_identity
from tests.test_pairwise_referent import response
from tests.test_source_reconciliation import Provider, candidate, event, target


class IdentityProvider(Provider):
    def __init__(self, relation="same_variable", coverage="covered"):
        super().__init__({})
        self.relation, self.coverage = relation, coverage

    async def predict(self, prompt, **kwargs):
        self.response = (
            {"coverage": self.coverage}
            if "IDENTITY COVERAGE DATA" in prompt
            else response(self.relation)
        )
        return await super().predict(prompt, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation", ["same_variable", "different_variable", "insufficient_identity"]
)
@pytest.mark.parametrize("coverage", ["covered", "incompatible", "unspecified"])
async def test_identity_guard_only_retains_relation_or_requires_review(relation, coverage):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    provider = IdentityProvider(relation, coverage)
    values = await compare_referents_with_identity_coverage(candidate(source), [target()], provider)
    expected = (
        "insufficient_identity"
        if coverage == "unspecified" or (relation == "same_variable" and coverage != "covered")
        else relation
    )
    assert values[0].relation == expected
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_false_unrelated_requirement_becomes_review_instead_of_new_fact():
    from dataclasses import replace

    source = event("source", "From now on the count must be eighteen instead of twelve.")
    item = replace(candidate(source, "continuing_requirement"), revises_existing_requirement=True)
    provider = IdentityProvider("different_variable", "unspecified")
    result = await reconcile_source_target_identity(
        [item], [source], [target()], provider, comparison_style="identity_coverage"
    )
    assert result[0].operation == "unresolved"


@pytest.mark.asyncio
async def test_benchmark_retains_both_raw_layers_and_grades_effective_relation():
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item, observed = candidate(source), target()
    captured = CapturedReferents(IdentityProvider("different_variable", "unspecified"))
    await compare_referents_with_identity_coverage(item, [observed], captured)
    assert len(captured.responses) == len(captured.coverage_responses) == 1
    case = {
        "targets": [{"id": observed.id, "version": observed.version}],
        "expected_relations": {observed.id: "insufficient_identity"},
    }
    graded = grade_pairwise_relations(
        case,
        item,
        [observed],
        captured.responses,
        error=None,
        coverage_responses=captured.coverage_responses,
    )
    assert graded["relations_correct"] == 1 and graded["relation_case_exact"]


@pytest.mark.parametrize(
    "coverage",
    [[], ["{}"], ['{"coverage":"covered","extra":1}'], ['{"coverage":"unspecified"}'] * 2],
)
def test_missing_malformed_or_extra_coverage_gets_no_relation_credit(coverage):
    source = event("source", "Correction: the marker is sapphire, not cedar.")
    item, observed = candidate(source), target()
    case = {
        "targets": [{"id": observed.id, "version": observed.version}],
        "expected_relations": {observed.id: "same_variable"},
    }
    grade = grade_pairwise_relations(
        case, item, [observed], [json.dumps(response())], error=None, coverage_responses=coverage
    )
    assert not grade["response_valid"] and grade["relations_correct"] == 0
