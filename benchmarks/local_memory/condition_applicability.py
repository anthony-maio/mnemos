"""Development-only condition logic over separately interpreted, bound evidence.

Span binding proves location, not semantic correctness. Source completeness and
query circumstance interpretation must be evaluated independently before use.
"""

from dataclasses import dataclass
import hashlib
from typing import Literal

Truth = Literal["true", "false", "unknown"]


@dataclass(frozen=True)
class Predicate:
    subject: str
    relation: str
    value: str


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Atom:
    predicate: Predicate
    span: Span


@dataclass(frozen=True)
class Expression:
    operator: Literal["all", "any", "not"]
    children: tuple["Atom | Expression", ...]


@dataclass(frozen=True)
class SourceConditions:
    source_sha256: str
    policy_revision: str
    complete: bool
    expression: Atom | Expression | None


@dataclass(frozen=True)
class Circumstance:
    predicate: Predicate
    span: Span
    positive: bool
    mode: Literal["case_constraint", "mention", "unresolved"]


@dataclass(frozen=True)
class QueryCircumstances:
    query_sha256: str
    policy_revision: str
    complete: bool
    items: tuple[Circumstance, ...]


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _predicate(predicate: Predicate) -> None:
    if not isinstance(predicate, Predicate) or any(
        not isinstance(value, str) or not value.strip()
        for value in (predicate.subject, predicate.relation, predicate.value)
    ):
        raise ValueError("Predicate requires explicit subject, relation and value")


def _span(span: Span, text: str) -> None:
    if (
        not isinstance(span, Span)
        or type(span.start) is not int
        or type(span.end) is not int
        or not 0 <= span.start < span.end <= len(text)
        or not isinstance(span.text, str)
        or not span.text.strip()
        or text[span.start : span.end] != span.text
    ):
        raise ValueError("Span must identify exact nonempty text in its own input")


def evaluate_conditions(
    source: SourceConditions, question: QueryCircumstances, *, source_text: str, query_text: str
) -> Literal["applies", "excluded", "unresolved"]:
    """Missing circumstances stay unknown, including under negation.

    A case_constraint describes the circumstances asked about, not a claim that
    those circumstances hold in the world. Mentioned or unresolved text cannot
    satisfy a prerequisite. No aliasing, inference or closed-world negation is
    performed here. Conflicting constraints produce unknown for that predicate.
    """
    if not isinstance(source, SourceConditions) or not isinstance(question, QueryCircumstances):
        raise ValueError("Expected separate source and question interpretations")
    if source.source_sha256 != text_digest(source_text) or question.query_sha256 != text_digest(
        query_text
    ):
        raise ValueError("Condition inputs changed")
    if (
        not isinstance(source.policy_revision, str)
        or not source.policy_revision.strip()
        or source.policy_revision != question.policy_revision
        or type(source.complete) is not bool
        or type(question.complete) is not bool
    ):
        raise ValueError("Condition interpretations require the same explicit policy")
    if not isinstance(question.items, tuple) or len(question.items) > 128:
        raise ValueError("Question circumstances exceed the bounded contract")
    states: dict[Predicate, set[bool]] = {}
    for item in question.items:
        if not isinstance(item, Circumstance) or type(item.positive) is not bool:
            raise ValueError("Invalid query circumstance")
        _predicate(item.predicate)
        _span(item.span, query_text)
        if item.mode not in {"case_constraint", "mention", "unresolved"}:
            raise ValueError("Unknown circumstance interpretation")
        if item.mode == "case_constraint":
            states.setdefault(item.predicate, set()).add(item.positive)

    visited = 0

    def evaluate(node: Atom | Expression, depth: int = 0) -> Truth:
        nonlocal visited
        visited += 1
        if visited > 128 or depth > 16:
            raise ValueError("Source condition expression exceeds the bounded contract")
        if isinstance(node, Atom):
            _predicate(node.predicate)
            _span(node.span, source_text)
            values = states.get(node.predicate, set())
            if values == {True}:
                return "true"
            if values == {False}:
                return "false"
            return "unknown"
        if (
            not isinstance(node, Expression)
            or node.operator not in {"all", "any", "not"}
            or not isinstance(node.children, tuple)
            or not node.children
            or (node.operator == "not" and len(node.children) != 1)
        ):
            raise ValueError("Invalid source condition expression")
        # Validate every branch, even when one value decides the logical result.
        values = [evaluate(child, depth + 1) for child in node.children]
        if node.operator == "not":
            return {"true": "false", "false": "true", "unknown": "unknown"}[values[0]]
        if node.operator == "all":
            return "false" if "false" in values else "unknown" if "unknown" in values else "true"
        return "true" if "true" in values else "unknown" if "unknown" in values else "false"

    value = evaluate(source.expression) if source.expression is not None else "true"
    if not source.complete or (source.expression is not None and not question.complete):
        return "unresolved"
    return {"true": "applies", "false": "excluded", "unknown": "unresolved"}[value]
