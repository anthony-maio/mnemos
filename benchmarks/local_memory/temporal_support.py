"""Query-independent temporal meaning, separate from source freshness claims.

This is a development contract for the next inventory version. It does not
reinterpret frozen v1 inventories or connect to production answer selection.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .source_quote_binding import bind_unique_quote


@dataclass(frozen=True)
class SourceTime:
    meaning: Literal['interval', 'present_assertion', 'timeless', 'unresolved']
    quotation: str | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    # An assertion date established by the source, never ingestion time or the
    # current query's reference date. A receipt time is not proof of validity.
    assertion_date: date | None = None


@dataclass(frozen=True)
class RequestedTime:
    meaning: Literal['source_state', 'at_date', 'unresolved']
    at: date | None = None


@dataclass(frozen=True)
class TemporalDecision:
    status: Literal['applicable', 'excluded', 'unresolved']
    basis: Literal['source_assertion', 'dated_support', 'timeless_support'] | None = None


def validate_source_time(annotation: SourceTime, excerpt: str) -> None:
    if not isinstance(annotation, SourceTime) or annotation.meaning not in {
            'interval', 'present_assertion', 'timeless', 'unresolved'}:
        raise ValueError('Invalid source temporal meaning')
    for boundary in (annotation.valid_from, annotation.valid_until, annotation.assertion_date):
        if boundary is not None and type(boundary) is not date:
            raise ValueError('Temporal bounds require exact dates')
    if annotation.quotation is not None:
        if bind_unique_quote(excerpt, annotation.quotation).status != 'bound':
            raise ValueError('Temporal quotation must bind uniquely to own evidence')
    if annotation.meaning != 'unresolved' and annotation.quotation is None:
        raise ValueError('Temporal assessment requires a source quotation')
    bounded = annotation.valid_from is not None or annotation.valid_until is not None
    if annotation.meaning == 'interval':
        if not bounded or annotation.assertion_date is not None:
            raise ValueError('Interval needs validity bounds, not an assertion date')
        if (annotation.valid_from is not None and annotation.valid_until is not None
                and annotation.valid_from >= annotation.valid_until):
            raise ValueError('Interval must be nonempty and half-open')
    elif bounded:
        raise ValueError('Only explicit intervals carry validity bounds')
    if annotation.meaning != 'present_assertion' and annotation.assertion_date is not None:
        raise ValueError('Only present assertions carry an assertion date')


def evaluate_time(annotation: SourceTime, requested: RequestedTime, excerpt: str) -> TemporalDecision:
    """Assess temporal scope only; relation, entailment and conditions still apply.

    source_state means explicitly asking what retained evidence states, with no
    demand for current, historical or future validity. It must not be used as a
    fallback when a dated/current question lacks sufficient temporal evidence.
    """
    validate_source_time(annotation, excerpt)
    if not isinstance(requested, RequestedTime) or requested.meaning not in {
            'source_state', 'at_date', 'unresolved'}:
        raise ValueError('Invalid requested temporal meaning')
    if ((requested.meaning == 'at_date' and type(requested.at) is not date)
            or (requested.meaning != 'at_date' and requested.at is not None)):
        raise ValueError('Requested date contradicts temporal meaning')
    if requested.meaning == 'unresolved' or annotation.meaning == 'unresolved':
        return TemporalDecision('unresolved')
    if requested.meaning == 'source_state':
        return TemporalDecision('applicable', 'source_assertion')
    if annotation.meaning == 'timeless':
        return TemporalDecision('applicable', 'timeless_support')
    if annotation.meaning == 'present_assertion':
        if annotation.assertion_date == requested.at:
            return TemporalDecision('applicable', 'dated_support')
        # A dated assertion does not establish truth or falsity on another day.
        return TemporalDecision('unresolved')
    if ((annotation.valid_from is None or annotation.valid_from <= requested.at)
            and (annotation.valid_until is None or requested.at < annotation.valid_until)):
        return TemporalDecision('applicable', 'dated_support')
    return TemporalDecision('excluded')
