"""Matched-evidence adapter; preserves the existing per-witness reranker policy."""

import json
import math


def eligible(candidate):
    return candidate["assertion_status"] == "asserted" and bool(candidate["sources"])


def baseline_requests(request, instruction):
    """Map validated joint inputs without inventing witnesses or metadata."""
    cohort = []
    seen = set()
    for candidate in request["candidates"]:
        if candidate["id"] in seen:
            raise ValueError("Duplicate candidate identity")
        seen.add(candidate["id"])
        evidence = []
        for source in candidate["sources"]:
            evidence.append(
                dict(
                    evidence_id=source["id"],
                    fact_id=candidate["id"],
                    source_event_id=source["source_event_id"],
                    excerpt=source["excerpt"],
                )
            )
        if len({s["evidence_id"] for s in evidence}) != len(evidence):
            raise ValueError("Duplicate owned witness")
        cohort.append(
            dict(
                id=candidate["id"],
                version=candidate["version"],
                content=candidate["content"],
                assertion_status=candidate["assertion_status"],
                sources=evidence,
            )
        )
    result = []
    for original, candidate in zip(request["candidates"], cohort, strict=True):
        if not eligible(original):
            continue
        for source in candidate["sources"]:
            document = dict(
                reference_date=request["reference_date"],
                target_fact_id=candidate["id"],
                target_fact=candidate["content"],
                target_evidence=source,
                candidates=cohort,
            )
            result.append(
                dict(
                    fact_id=candidate["id"],
                    evidence_id=source["evidence_id"],
                    instruction=instruction,
                    query=request["query"],
                    document=json.dumps(document, ensure_ascii=False, separators=(",", ":")),
                )
            )
    if len(result) > 32:
        raise ValueError("Entire baseline question exceeds witness budget")
    return result


def select_joint(request, target_ids, logits):
    expected = [c["id"] for c in request["candidates"]]
    if list(target_ids) != expected or len(logits) != len(expected):
        raise ValueError("Target logits do not bind the exact ordered cohort")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in logits
    ):
        raise ValueError("Nonfinite or malformed logit")
    return [
        c["id"]
        for c, value in zip(request["candidates"], logits, strict=True)
        if value >= 0 and eligible(c)
    ]
