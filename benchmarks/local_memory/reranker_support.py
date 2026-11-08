"""Fixed witness decoder for the prospective reranker development screen."""

from collections.abc import Mapping, Sequence
import math
from typing import Any


def select_witnesses(
    requests: Sequence[Mapping[str, Any]], logits: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Validate the entire question before selecting at the fixed 0.9 threshold.

    Logit order is no, yes. Scores normalize only those two model tokens and
    are not calibrated correctness probabilities. Empty input selects nothing.
    """
    if len(requests) != len(logits) or len(requests) > 32:
        raise ValueError("Invalid witness batch size")
    seen: set[tuple[str, str]] = set()
    rows = []
    for request, values in zip(requests, logits):
        fact, evidence = request.get("fact_id"), request.get("evidence_id")
        if not isinstance(fact, str) or not fact or not isinstance(evidence, str) or not evidence:
            raise ValueError("Missing witness identity")
        if (fact, evidence) in seen:
            raise ValueError("Duplicate witness")
        seen.add((fact, evidence))
        if len(values) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Expected finite no/yes logits")
        maximum = max(values)
        no, yes = (math.exp(value - maximum) for value in values)
        score = yes / (no + yes)
        rows.append({"fact_id": fact, "evidence_id": evidence, "score": score})
    witnesses: dict[str, list[str]] = {}
    for row in rows:
        if row["score"] >= 0.9:
            witnesses.setdefault(row["fact_id"], []).append(row["evidence_id"])
    return {"selected_ids": list(witnesses), "witnesses": witnesses, "scores": rows}
