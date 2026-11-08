from dataclasses import replace
import json

import pytest

from benchmarks.local_memory.packed_query_probe import build_cases, prepare_probe
from benchmarks.local_memory.source_qualification import QualificationIdentity


IDENTITY = QualificationIdentity('a' * 64, 'b' * 64, 'c' * 64)


def test_probe_preserves_reordered_evidence_and_separates_grading():
    cases = build_cases(IDENTITY, IDENTITY)
    execution, grading = prepare_probe(cases)
    assert len(execution['requests']) == len(grading['labels']) == 16
    for request in execution['requests']:
        assert not {'expected_text', 'expected_status', 'expected_fact_ids'} & request.keys()
        payload = json.loads(request['prompt'].split('\nINPUT\n')[1])
        assert payload['claim_order'] == ['c0', 'c1']
        assert len(payload['facts']) == 2
    first, second = cases[:2]
    assert first.request.snapshot.candidates == tuple(reversed(second.request.snapshot.candidates))
    assert first.expected_fact_ids == second.expected_fact_ids == ('owner',)
    assert first.expected_text != second.expected_text
    assert {row['expected_status'] for row in grading['labels']} == {
        'selected', 'conflict', 'absent', 'unavailable'}


@pytest.mark.parametrize('mutation', ['reverse', 'duplicate', 'wrong_status', 'wrong_fact', 'wrong_position'])
def test_invalid_frozen_expectations_rejected_before_execution(mutation):
    cases = build_cases(IDENTITY, IDENTITY)
    if mutation == 'reverse': cases = tuple(reversed(cases))
    if mutation == 'duplicate': cases = (cases[0], cases[0])
    if mutation == 'wrong_status': cases = (replace(cases[0], expected_status='absent'),)
    if mutation == 'wrong_fact': cases = (replace(cases[0], expected_fact_ids=('region',)),)
    if mutation == 'wrong_position': cases = (replace(cases[0], expected_text='"S|XAMA"'),)
    with pytest.raises(ValueError): prepare_probe(cases)
