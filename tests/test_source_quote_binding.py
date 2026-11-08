import pytest

from benchmarks.local_memory.source_quote_binding import bind_own_source_quotes, bind_unique_quote


def test_exact_quote_derives_python_character_offsets():
    source = 'A\U0001f600 owner is Mira.'
    result = bind_unique_quote(source, 'Mira')
    assert result.status == 'bound'
    assert result.span.start == source.index('Mira')
    result.span.validate(source)


@pytest.mark.parametrize('source,quote', [('Mira and Mira', 'Mira'), ('aaa', 'aa')])
def test_repeated_or_overlapping_quote_never_selects_first_occurrence(source, quote):
    result = bind_unique_quote(source, quote)
    assert result.status == 'ambiguous' and result.span is None


@pytest.mark.parametrize('quote', ['mira', 'Mira ', 'Mi ra', 'Morgan'])
def test_no_case_whitespace_or_semantic_normalization(quote):
    assert bind_unique_quote('Mira', quote).status == 'absent'


@pytest.mark.parametrize('quote', ['', ' ', None, 3, 'x' * 64001])
def test_invalid_quote_is_contract_error(quote):
    with pytest.raises(ValueError):
        bind_unique_quote('source', quote)


def test_own_source_binding_never_searches_other_evidence():
    sources = {'a': {'own': 'Project owner is Mira.'}, 'b': {'foreign': 'Project owner is Lee.'}}
    quotes = dict(subject_span='Project', relation_span='owner', value_span='Lee',
                  temporal_span=None, whole_fact_span=None)
    result = bind_own_source_quotes('a', 'own', quotes, sources=sources)
    assert result['value_span'].status == 'absent'
    with pytest.raises(ValueError, match='foreign'):
        bind_own_source_quotes('a', 'foreign', quotes, sources=sources)


def test_changed_or_forgotten_source_cannot_reuse_binding():
    quotes = dict(subject_span='Project', relation_span='owner', value_span='Mira',
                  temporal_span=None, whole_fact_span='Project owner is Mira.')
    original = {'a': {'own': 'Project owner is Mira.'}}
    assert bind_own_source_quotes('a', 'own', quotes, sources=original)['value_span'].status == 'bound'
    changed = {'a': {'own': 'Project owner is Lee.'}}
    assert bind_own_source_quotes('a', 'own', quotes, sources=changed)['value_span'].status == 'absent'
    with pytest.raises(ValueError):
        bind_own_source_quotes('a', 'own', quotes, sources={})


def test_missing_quote_field_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match='explicit'):
        bind_own_source_quotes('a', 'own', {'value_span': 'Mira'}, sources={'a': {'own': 'Mira'}})
