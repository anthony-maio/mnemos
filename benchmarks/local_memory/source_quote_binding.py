"""Development-only exact quotation binding, without model-generated offsets.

Uniqueness proves a location in the supplied evidence, not semantic support.
No normalization, fuzzy matching or first-occurrence fallback is permitted.
"""

from dataclasses import dataclass
from typing import Literal, Mapping

from .relational_support import Span


@dataclass(frozen=True)
class QuoteBinding:
    status: Literal['bound', 'absent', 'ambiguous']
    span: Span | None = None


def bind_unique_quote(source: str, quote: str) -> QuoteBinding:
    if (not isinstance(source, str) or not isinstance(quote, str)
            or not quote.strip() or len(source.encode('utf-8')) > 64000
            or len(quote.encode('utf-8')) > 64000):
        raise ValueError('Require bounded source text and a nonblank exact quotation')
    start = source.find(quote)
    if start < 0:
        return QuoteBinding('absent')
    # Start + 1 catches overlapping occurrences such as aa within aaa.
    if source.find(quote, start + 1) >= 0:
        return QuoteBinding('ambiguous')
    span = Span(start, start + len(quote), quote)
    span.validate(source)
    return QuoteBinding('bound', span)


def bind_own_source_quotes(
    fact_id: str,
    evidence_id: str,
    quotes: Mapping[str, str | None],
    *,
    sources: Mapping[str, Mapping[str, str]],
) -> dict[str, QuoteBinding | None]:
    """Bind all supplied claim quotations to one eligible fact's own evidence.

    Callers must retain non-bound results as unresolved; they cannot drop an
    inconvenient quote and treat the remaining fragments as whole-fact support.
    Null temporal/whole-fact quotes are permitted for unqualified annotations,
    but the existing evaluator still requires whole-fact and temporal witnesses
    whenever their respective assessments claim support or bounded validity.
    """
    fields = ('subject_span', 'relation_span', 'value_span', 'temporal_span', 'whole_fact_span')
    if set(quotes) != set(fields):
        raise ValueError('Every quote field must be explicit, with no extra fields')
    own = sources.get(fact_id)
    if own is None or evidence_id not in own:
        raise ValueError('Quotation cites unavailable or foreign evidence')
    results: dict[str, QuoteBinding | None] = {}
    for field in fields:
        quote = quotes[field]
        if quote is None:
            if field not in {'temporal_span', 'whole_fact_span'}:
                raise ValueError('Subject, relation and value quotations are required')
            results[field] = None
        else:
            results[field] = bind_unique_quote(own[evidence_id], quote)
    return results
