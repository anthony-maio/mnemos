from dataclasses import replace
from datetime import date

import pytest

from benchmarks.local_memory.relational_support import AnnotationSet, Claim, QueryFrame, Span, evaluate


def fixture_claim(fact, relation, value, *, start=None, end=None):
    interval = f"from {start} until {end}"
    source = f"Project {relation} {value}; {interval}."
    def span(text):
        i = source.index(text)
        return Span(i, i + len(text), text)
    return source, Claim(fact, fact + '-source', 'project', relation, value,
                         span('Project'), span(relation), span(value), start, end,
                         span(interval) if start or end else None,
                         'bounded' if start or end else 'unbounded')


def run(items, relation='duration', at=date(2026, 11, 8), **changes):
    sources = {c.fact_id: {c.evidence_id: text} for text, c in items}
    annotations = AnnotationSet('snapshot', 'policy', tuple(c for _, c in items), tuple(sources))
    return evaluate(QueryFrame('project', relation, at), replace(annotations, **changes),
                    sources=sources, snapshot_fingerprint='snapshot', policy_identity='policy')


def test_conflict_only_applies_to_requested_relation():
    items = [fixture_claim('a', 'duration', '7 days'), fixture_claim('b', 'duration', '9 days'),
             fixture_claim('c', 'reviewer', 'Taylor')]
    assert run(items).status == 'conflict'
    assert run(items, 'reviewer').fact_ids == ('c',)


def test_historical_interval_and_replacement_boundary():
    items = [fixture_claim('old', 'owner', 'Taylor', start=date(2025, 1, 1), end=date(2026, 1, 1)),
             fixture_claim('new', 'owner', 'Morgan', start=date(2026, 1, 1))]
    assert run(items, 'owner', date(2025, 12, 31)).fact_ids == ('old',)
    assert run(items, 'owner', date(2026, 1, 1)).fact_ids == ('new',)
    assert run(items, 'owner', date(2024, 12, 31)).status == 'absent'


@pytest.mark.parametrize('changes', [dict(snapshot_fingerprint='stale'), dict(policy_identity='other'),
                                    dict(assessed_fact_ids=()), dict(assessed_fact_ids=('a', 'a'))])
def test_incomplete_or_stale_annotations_are_unavailable(changes):
    assert run([fixture_claim('a', 'duration', '7 days')], **changes).status == 'unavailable'


def test_foreign_evidence_and_changed_span_rejected():
    source, claim = fixture_claim('a', 'duration', '7 days')
    args = dict(sources={'a': {'a-source': source}}, snapshot_fingerprint='snapshot', policy_identity='policy')
    for changed in (replace(claim, evidence_id='foreign'),
                    replace(claim, value_span=Span(0, 7, 'changed'))):
        with pytest.raises(ValueError):
            evaluate(QueryFrame('project', 'duration', date.today()),
                     AnnotationSet('snapshot', 'policy', (changed,), ('a',)), **args)


def test_temporal_claim_requires_provenance_and_valid_order():
    source, claim = fixture_claim('a', 'owner', 'Taylor', start=date(2025, 1, 1))
    for changed in (replace(claim, temporal_span=None), replace(claim, valid_until=date(2025, 1, 1))):
        with pytest.raises(ValueError):
            run([(source, changed)], 'owner')


def test_equal_values_preserve_all_supporting_witnesses():
    result = run([fixture_claim('a', 'duration', '7 days'), fixture_claim('b', 'duration', '7 days')])
    assert result.fact_ids == ('a', 'b')
    assert result.evidence_ids == ('a-source', 'b-source')


def test_unresolved_applicability_is_not_timeless_support_or_successful_absence():
    source, claim = fixture_claim('a', 'owner', 'Taylor')
    assert run([(source, replace(claim, applicability='unresolved'))], 'owner').status == 'unavailable'
