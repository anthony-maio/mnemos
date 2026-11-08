"""Pure decision rule for the prospective discriminative development screen."""

from collections.abc import Mapping, Sequence
import math
from typing import Any


def select_supported(
    pairs: Sequence[Mapping[str, Any]], logits: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Apply fixed three-class support and contradiction thresholds.

    Label order is entailment, neutral, contradiction. The runner must verify
    the model configuration before invoking this rule. This is not a public
    memory API and does not perform source/version validation or inference.
    """
    if len(pairs) != len(logits) or len(pairs) > 128:
        raise ValueError("pair/logit count mismatch or pair limit exceeded")
    support: dict[str, list[str]] = {}
    scores = []
    witnessed: dict[str, list[str]] = {}
    vetoes = []
    for pair, values in zip(pairs, logits):
        if len(values) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("expected three finite numeric logits")
        peak = max(values)
        weights = [math.exp(value - peak) for value in values]
        probabilities = [weight / sum(weights) for weight in weights]
        scores.append(probabilities)
        if pair.get("kind") not in {"support", "contradiction"}:
            raise ValueError("unknown pair kind")
        for field in ("fact_id", "evidence_id"):
            if not isinstance(pair.get(field), str) or not pair[field]:
                raise ValueError("missing pair identity")
        if pair["kind"] == "support":
            witnesses = support.setdefault(pair["fact_id"], [])
            if pair["evidence_id"] in witnesses:
                raise ValueError("duplicate support evidence")
            witnesses.append(pair["evidence_id"])
            if probabilities[0] >= 0.8:
                witnessed.setdefault(pair["fact_id"], []).append(pair["evidence_id"])
    expected = {
        (left, le, right, re)
        for left, les in support.items()
        for right, res in support.items()
        if left != right
        for le in les
        for re in res
    }
    seen = set()
    for pair, probabilities in zip(pairs, scores):
        if pair["kind"] != "contradiction":
            continue
        key = (
            pair["fact_id"],
            pair["evidence_id"],
            pair.get("other_fact_id"),
            pair.get("other_evidence_id"),
        )
        if key not in expected or key in seen:
            raise ValueError("invalid or duplicate contradiction pair")
        seen.add(key)
        if probabilities[2] >= 0.8:
            vetoes.append(list(key))
    if seen != expected:
        raise ValueError("incomplete contradiction coverage")
    vetoed = {key[index] for key in vetoes for index in (0, 2)}
    return {
        "selected_ids": [fact for fact in support if fact in witnessed and fact not in vetoed],
        "support_witnesses": witnessed,
        "contradiction_vetoes": vetoes,
        "probabilities": scores,
    }
