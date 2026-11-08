from dataclasses import replace
from itertools import product

import pytest

from benchmarks.local_memory.condition_applicability import (
    Atom,
    Circumstance,
    Expression,
    Predicate,
    QueryCircumstances,
    SourceConditions,
    Span,
    evaluate_conditions,
    text_digest,
)

SOURCE = "Only when maintenance is active and approval is granted."
QUERY = "For maintenance active and approval granted, who handles the task?"
MAINTENANCE = Predicate("task", "maintenance", "active")
APPROVAL = Predicate("task", "approval", "granted")


def span(text, quote):
    start = text.index(quote)
    return Span(start, start + len(quote), quote)


A = Atom(MAINTENANCE, span(SOURCE, "maintenance is active"))
B = Atom(APPROVAL, span(SOURCE, "approval is granted"))


def evaluate(expr, states=(True, True), *, complete=True):
    items = tuple(
        Circumstance(pred, span(QUERY, quote), value, "case_constraint")
        for pred, quote, value in zip(
            (MAINTENANCE, APPROVAL), ("maintenance active", "approval granted"), states
        )
        if value is not None
    )
    source = SourceConditions(text_digest(SOURCE), "authored-test-v1", complete, expr)
    query = QueryCircumstances(text_digest(QUERY), "authored-test-v1", True, items)
    return evaluate_conditions(source, query, source_text=SOURCE, query_text=QUERY)


@pytest.mark.parametrize("left,right", list(product([True, False, None], repeat=2)))
def test_conjunction_and_disjunction_keep_unknown_distinct_from_false(left, right):
    expected_all = (
        "excluded"
        if False in (left, right)
        else "unresolved" if None in (left, right) else "applies"
    )
    expected_any = (
        "applies"
        if True in (left, right)
        else "unresolved" if None in (left, right) else "excluded"
    )
    assert evaluate(Expression("all", (A, B)), (left, right)) == expected_all
    assert evaluate(Expression("any", (A, B)), (left, right)) == expected_any


@pytest.mark.parametrize(
    "value,expected", [(True, "excluded"), (False, "applies"), (None, "unresolved")]
)
def test_negation_does_not_turn_absence_into_evidence(value, expected):
    assert evaluate(Expression("not", (A,)), (value, None)) == expected


def fixtures():
    source = SourceConditions(text_digest(SOURCE), "authored-test-v1", True, A)
    witness = Circumstance(MAINTENANCE, span(QUERY, "maintenance active"), True, "case_constraint")
    query = QueryCircumstances(text_digest(QUERY), "authored-test-v1", True, (witness,))
    return source, query, witness


@pytest.mark.parametrize("mode", ["mention", "unresolved"])
def test_mentions_and_uncertain_interpretations_do_not_establish_a_case(mode):
    source, query, witness = fixtures()
    query = replace(query, items=(replace(witness, mode=mode),))
    assert evaluate_conditions(source, query, source_text=SOURCE, query_text=QUERY) == "unresolved"


def test_foreign_subject_and_conflicting_polarities_are_not_assumed_true():
    source, query, witness = fixtures()
    foreign = replace(witness, predicate=replace(MAINTENANCE, subject="another-task"))
    for items in [(foreign,), (witness, replace(witness, positive=False))]:
        assert (
            evaluate_conditions(
                source, replace(query, items=items), source_text=SOURCE, query_text=QUERY
            )
            == "unresolved"
        )


def test_incomplete_interpretation_never_becomes_unconditional_support():
    source, query, _ = fixtures()
    assert evaluate(None, complete=False) == "unresolved"
    assert evaluate(None) == "applies"
    assert (
        evaluate_conditions(
            source, replace(query, complete=False), source_text=SOURCE, query_text=QUERY
        )
        == "unresolved"
    )


@pytest.mark.parametrize(
    "mutation",
    ["source", "query", "policy", "span", "offset", "predicate", "empty", "arity", "hidden"],
)
def test_invalid_or_stale_bindings_are_rejected(mutation):
    source, query, witness = fixtures()
    if mutation == "source":
        source = replace(source, source_sha256="old")
    if mutation == "query":
        query = replace(query, query_sha256="old")
    if mutation == "policy":
        query = replace(query, policy_revision="different")
    if mutation == "span":
        query = replace(
            query, items=(replace(witness, span=replace(witness.span, text="invented")),)
        )
    if mutation == "offset":
        source = replace(source, expression=replace(A, span=replace(A.span, start=True)))
    if mutation == "predicate":
        source = replace(source, expression=replace(A, predicate=replace(MAINTENANCE, subject="")))
    if mutation == "empty":
        source = replace(source, expression=Expression("all", ()))
    if mutation == "arity":
        source = replace(source, expression=Expression("not", (A, B)))
    if mutation == "hidden":
        source = replace(
            source, expression=Expression("any", (A, replace(B, span=Span(0, 3, "bad"))))
        )
    with pytest.raises(ValueError):
        evaluate_conditions(source, query, source_text=SOURCE, query_text=QUERY)


def test_excessive_expression_depth_and_circumstance_count_fail():
    source, query, witness = fixtures()
    expr = A
    for _ in range(18):
        expr = Expression("not", (expr,))
    with pytest.raises(ValueError):
        evaluate_conditions(
            replace(source, expression=expr), query, source_text=SOURCE, query_text=QUERY
        )
    with pytest.raises(ValueError):
        evaluate_conditions(
            source, replace(query, items=(witness,) * 129), source_text=SOURCE, query_text=QUERY
        )
