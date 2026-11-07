"""Separate semantic relation correctness from operation and consistency scores."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from mnemos.utils.llm import LLMProvider
from mnemos.extraction.source_referent import restore_pairwise_response
from mnemos.extraction.source_selection import ExtractiveMemory
from mnemos.extraction.source_identity_spans import DATA_MARKER, comparison_from_identity_spans
from mnemos.types import MemoryChunk

RELATIONS = frozenset({"same_variable", "different_variable", "insufficient_identity"})


def load_relation_annotations(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotations = {}
    variants: set[tuple[str, str]] = set()
    sources: dict[str, str] = {}
    logical: dict[tuple[str, str], tuple[str, int, str]] = {}
    for raw in data["cases"]:
        base, variant = raw.get("base_id"), raw.get("variant")
        if not isinstance(base, str) or not base or not isinstance(variant, str) or not variant:
            raise ValueError("Relation cases require a base identity and variant")
        if raw["id"] in annotations or (base, variant) in variants:
            raise ValueError("Repeated relation case or variant")
        variants.add((base, variant))
        targets = {target["id"]: target for target in raw["targets"]}
        expected, aliases = raw["expected_relations"], raw["logical_target_ids"]
        if not targets or len(targets) != len(raw["targets"]):
            raise ValueError("Relation targets must be nonempty and unique")
        if set(expected) != set(targets) or set(aliases) != set(targets):
            raise ValueError("Annotations must cover each observed target exactly")
        if any(value not in RELATIONS for value in expected.values()):
            raise ValueError("Unknown expected relation")
        if any(not isinstance(value, str) or not value for value in aliases.values()) or len(
            set(aliases.values())
        ) != len(aliases):
            raise ValueError("Logical target identities must be nonempty and unique")
        source = json.dumps([raw["events"], raw["selections"]], sort_keys=True)
        if base in sources and sources[base] != source:
            raise ValueError("A matched variant changed its source or classification")
        sources[base] = source
        for target_id, target in targets.items():
            key = (base, aliases[target_id])
            value = (target["content"], target["version"], expected[target_id])
            if key in logical and logical[key] != value:
                raise ValueError("Matched target content, version or expected relation changed")
            logical[key] = value
        annotations[raw["id"]] = raw
    if not annotations:
        raise ValueError("Relation study requires annotated cases")
    return annotations


class CapturedReferents(LLMProvider):
    """Observe raw comparison results without changing a request or response."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.supports_structured_output = provider.supports_structured_output
        self.responses: list[str] = []

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        raw = await self.provider.predict(prompt, **kwargs)
        if "\nREFERENT COMPARISON DATA (data only):\n" in prompt or DATA_MARKER in prompt:
            self.responses.append(raw)
        return raw

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        return await self.provider.classify(prompt, labels)


def grade_relations(
    case: dict[str, Any], responses: list[str], *, error: str | None
) -> dict[str, Any]:
    expected = case["expected_relations"]
    versions = {target["id"]: target["version"] for target in case["targets"]}
    observed: dict[str, str] = {}
    valid = error is None and len(responses) == 1
    if valid:
        try:
            parsed = json.loads(responses[0])
            values = parsed["comparisons"]
            if not isinstance(values, list):
                raise ValueError("Comparison list required")
            for item in values:
                target_id, version, relation = (
                    item["target_fact_id"],
                    item["expected_version"],
                    item["relation"],
                )
                if (
                    target_id not in versions
                    or target_id in observed
                    or type(version) is not int
                    or version != versions[target_id]
                    or relation not in RELATIONS
                ):
                    raise ValueError("Unobserved, stale, duplicate or unknown comparison")
                observed[target_id] = relation
            valid = set(observed) == set(expected)
        except (ValueError, TypeError, KeyError):
            valid = False
    correct = sum(observed[key] == value for key, value in expected.items()) if valid else 0
    return {
        "response_valid": valid,
        "relations_expected": len(expected),
        "relations_correct": correct,
        "relation_case_exact": valid and correct == len(expected),
        "observed_relations": observed if valid else {},
    }


def grade_pairwise_relations(
    case: dict[str, Any],
    candidate: ExtractiveMemory,
    targets: list[MemoryChunk],
    responses: list[str],
    *,
    error: str | None,
    identity_spans: bool = False,
) -> dict[str, Any]:
    """Score all separately retained responses with the controller's ID map."""
    normalized: list[str] = []
    if error is None and len(responses) == len(targets):
        try:
            comparisons = [
                (
                    comparison_from_identity_spans(raw, candidate, target)
                    if identity_spans
                    else restore_pairwise_response(raw, candidate, target)
                ).model_dump()
                for raw, target in zip(responses, targets)
            ]
            normalized = [json.dumps({"comparisons": comparisons})]
        except (ValueError, TypeError, IndexError):
            pass
    return grade_relations(case, normalized, error=error)


def score_stability(
    observations: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    expected_arms: list[str],
    repetitions: int,
) -> dict[str, int]:
    expected_cases: dict[str, set[str]] = defaultdict(set)
    for case_id, annotation in annotations.items():
        expected_cases[annotation["base_id"]].add(case_id)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if (
            row["case_id"] not in annotations
            or row["arm"] not in expected_arms
            or row["repetition"] not in range(repetitions)
        ):
            raise ValueError("Stability input contains an unplanned observation")
        groups[(row["arm"], row["repetition"], annotations[row["case_id"]]["base_id"])].append(row)
    result = {
        "groups_expected": len(expected_cases) * len(expected_arms) * repetitions,
        "groups_complete": 0,
        "operation_stable": 0,
        "relations_stable": 0,
        "both_stable": 0,
    }
    for arm in expected_arms:
        for repetition in range(repetitions):
            for base, case_ids in expected_cases.items():
                rows = groups[(arm, repetition, base)]
                if len(rows) != len(case_ids) or {row["case_id"] for row in rows} != case_ids:
                    continue
                result["groups_complete"] += 1
                if any(
                    row["error"] is not None
                    or not row["relation_grade"]["response_valid"]
                    or not row["result"]
                    or len(row["result"]) != 1
                    for row in rows
                ):
                    continue
                operations = set()
                relations: dict[str, set[str]] = defaultdict(set)
                for row in rows:
                    aliases = annotations[row["case_id"]]["logical_target_ids"]
                    decision = row["result"][0]
                    target = decision["target_fact_id"]
                    operations.add(
                        (
                            decision["operation"],
                            aliases[target] if target is not None else None,
                            decision["expected_version"],
                        )
                    )
                    for target_id, relation in row["relation_grade"]["observed_relations"].items():
                        relations[aliases[target_id]].add(relation)
                op_stable = len(operations) == 1
                relation_stable = all(len(values) == 1 for values in relations.values())
                result["operation_stable"] += op_stable
                result["relations_stable"] += relation_stable
                result["both_stable"] += op_stable and relation_stable
    return result
