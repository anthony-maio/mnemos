"""Development-only decoding of observed single-token choice probabilities.

No inference, renormalization, semantic judgment, or production activation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenChoice:
    choice: str
    probabilities: tuple[tuple[str, float], ...]
    remaining_mass: float


def decode_token_choice(response: Mapping[str, object], labels: Sequence[str]) -> TokenChoice:
    """Require all offered labels at one observed token position.

    Returned values preserve reported mass without normalization over labels.
    The caller must verify provider score semantics and semantic quality.
    Missing labels are not zero probabilities.
    """
    if isinstance(labels, (str, bytes)) or not labels:
        raise ValueError("Expected explicit choice labels")
    if any(
        not isinstance(x, str) or len(x) != 1 or not x.isascii() or not x.isupper() for x in labels
    ):
        raise ValueError("Labels must be single uppercase ASCII letters")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels")
    if response.get("done") is not True or response.get("thinking"):
        raise ValueError("Response is incomplete or contains thinking output")
    entries = response.get("logprobs")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Expected exactly one scored output token")

    def score(item: object) -> tuple[str, float]:
        if not isinstance(item, dict):
            raise ValueError("Invalid token score")
        token, logprob = item.get("token"), item.get("logprob")
        if not isinstance(token, str) or not token:
            raise ValueError("Missing token text")
        if type(logprob) not in (int, float) or not math.isfinite(logprob) or logprob > 0:
            raise ValueError("Invalid log probability")
        wire = item.get("bytes")
        if wire is not None and (
            not isinstance(wire, list)
            or any(type(b) is not int or not 0 <= b <= 255 for b in wire)
            or bytes(wire) != token.encode("utf-8")
        ):
            raise ValueError("Token bytes disagree with text")
        return token, float(logprob)

    chosen, chosen_logprob = score(entries[0])
    if chosen not in labels or response.get("response") != chosen:
        raise ValueError("Generated response is not exactly one offered label")
    alternatives = entries[0].get("top_logprobs")
    if not isinstance(alternatives, list):
        raise ValueError("Missing alternatives")
    scores: dict[str, float] = {}
    for item in alternatives:
        token, value = score(item)
        if token in scores:
            raise ValueError("Duplicate alternative")
        scores[token] = value
    if not set(labels) <= scores.keys():
        raise ValueError("Offered labels missing from observed alternatives")
    if not math.isclose(scores[chosen], chosen_logprob, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Chosen token score disagrees with alternatives")
    if math.fsum(math.exp(value) for value in scores.values()) > 1 + 1e-6:
        raise ValueError("Observed probability mass exceeds one")
    probabilities = tuple((label, math.exp(scores[label])) for label in labels)
    return TokenChoice(chosen, probabilities, max(0.0, 1 - math.fsum(p for _, p in probabilities)))
