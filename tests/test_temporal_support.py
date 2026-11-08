from datetime import date, datetime

import pytest

from benchmarks.local_memory.temporal_support import SourceTime, RequestedTime, evaluate_time


TEXT = 'Mira currently owns Project.'


def test_present_assertion_answers_source_state_without_claiming_timeless_truth():
    result = evaluate_time(SourceTime('present_assertion', TEXT), RequestedTime('source_state'), TEXT)
    assert result.status == 'applicable' and result.basis == 'source_assertion'


@pytest.mark.parametrize('at', [date(2020, 1, 1), date(2026, 11, 8), date(2030, 1, 1)])
def test_unanchored_present_assertion_never_borrows_query_date(at):
    result = evaluate_time(SourceTime('present_assertion', TEXT), RequestedTime('at_date', at), TEXT)
    assert result.status == 'unresolved' and result.basis is None


@pytest.mark.parametrize('at,expected', [(date(2026, 11, 7), 'unresolved'),
                                      (date(2026, 11, 8), 'applicable'),
                                      (date(2026, 11, 9), 'unresolved')])
def test_explicit_as_of_assertion_does_not_extend_beyond_its_date(at, expected):
    text = 'As of November 8, 2026, Mira owns Project.'
    annotation = SourceTime('present_assertion', text, assertion_date=date(2026, 11, 8))
    assert evaluate_time(annotation, RequestedTime('at_date', at), text).status == expected


@pytest.mark.parametrize('at,expected', [(date(2025, 1, 1), 'applicable'), (date(2026, 1, 1), 'excluded')])
def test_explicit_half_open_interval(at, expected):
    text = 'During 2025, Mira owned Project.'
    annotation = SourceTime('interval', text, date(2025, 1, 1), date(2026, 1, 1))
    assert evaluate_time(annotation, RequestedTime('at_date', at), text).status == expected


@pytest.mark.parametrize('annotation', [SourceTime('present_assertion'),
    SourceTime('present_assertion', TEXT, valid_from=date(2026, 1, 1)),
    SourceTime('interval', TEXT), SourceTime('timeless', TEXT, assertion_date=date(2026, 1, 1)),
    SourceTime('present_assertion', 'other source'),
    SourceTime('present_assertion', TEXT, assertion_date=datetime(2026, 1, 1))])
def test_incompatible_or_unbound_temporal_fields_are_rejected(annotation):
    with pytest.raises(ValueError): evaluate_time(annotation, RequestedTime('source_state'), TEXT)


def test_unresolved_source_cannot_become_source_state_support():
    result = evaluate_time(SourceTime('unresolved'), RequestedTime('source_state'), TEXT)
    assert result.status == 'unresolved'


def test_source_state_must_not_smuggle_a_date():
    with pytest.raises(ValueError):
        evaluate_time(SourceTime('present_assertion', TEXT),
                      RequestedTime('source_state', date(2026, 11, 8)), TEXT)
