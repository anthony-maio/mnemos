"""Uncertainty-aware, deterministic scoring helpers for local-memory runs."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95
    method: str = ""
    sample_count: int = 0
    cluster_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "method": self.method,
            "sample_count": self.sample_count,
            "cluster_count": self.cluster_count,
        }


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Return a Wilson score interval for a binomial proportion.

    This is used for retention, Recall@k, and abstention accuracy. The result
    includes the sample count so a small fixture cannot look like a production
    confidence claim.
    """
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(
            "successes and trials must satisfy 0 <= successes <= trials and trials > 0"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    z = _normal_quantile(0.5 + confidence / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return Interval(
        estimate=p,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence=confidence,
        method="wilson",
        sample_count=trials,
    )


def paired_bootstrap_interval(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 20260916,
    cluster_ids: Sequence[str] | None = None,
) -> Interval:
    """Bootstrap the paired treatment-minus-baseline difference.

    Resampling is paired by index, preserving the same session/query cases in
    both profiles. A fixed seed makes reports reproducible and is recorded by
    the harness. With cluster IDs, each draw selects whole clusters with
    replacement, retaining every observation in each selected cluster. The
    estimand remains the observation-weighted mean, so unequal cluster sizes
    can produce resamples of different lengths. This function performs no
    model work.
    """
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired samples must have equal non-zero lengths")
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if cluster_ids is not None and len(cluster_ids) != len(baseline):
        raise ValueError("cluster_ids must have one ID per paired observation")
    differences = [float(after) - float(before) for before, after in zip(baseline, treatment)]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("paired samples must contain finite numbers")
    rng = random.Random(seed)
    n = len(differences)
    if cluster_ids is None:
        clusters = [differences]
        cluster_labels = None
    else:
        cluster_labels = list(dict.fromkeys(cluster_ids))
        clusters = [
            [
                difference
                for difference, cluster in zip(differences, cluster_ids)
                if cluster == label
            ]
            for label in cluster_labels
        ]
    cluster_count = len(clusters)
    draws = []
    for _ in range(repetitions):
        if cluster_labels is None:
            sampled = [differences[rng.randrange(n)] for _ in range(n)]
        else:
            sampled = [
                difference
                for _ in range(cluster_count)
                for difference in clusters[rng.randrange(cluster_count)]
            ]
        draws.append(sum(sampled) / len(sampled))
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        estimate=sum(differences) / n,
        lower=_quantile(draws, alpha),
        upper=_quantile(draws, 1.0 - alpha),
        confidence=confidence,
        method="paired_bootstrap",
        sample_count=n,
        cluster_count=None if cluster_labels is None else cluster_count,
    )


def binary_interval(values: Iterable[bool], confidence: float = 0.95) -> Interval:
    """Score booleans and attach the required Wilson interval."""
    values_list = list(values)
    if not values_list:
        raise ValueError("at least one binary observation is required")
    return wilson_interval(sum(bool(value) for value in values_list), len(values_list), confidence)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile for an empty sequence")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * fraction)


def _normal_quantile(probability: float) -> float:
    """Acklam's rational approximation, avoiding a scipy dependency."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
    coefficients_a = (
        -39.6968302866538,
        220.946098424521,
        -275.928510446969,
        138.357751867269,
        -30.6647980661472,
        2.50662827745924,
    )
    coefficients_b = (
        -54.4760987982241,
        161.585836858041,
        -155.698979859887,
        66.8013118877197,
        -13.2806815528857,
    )
    coefficients_c = (
        -0.00778489400243029,
        -0.322396458041136,
        -2.40075827716184,
        -2.54973253934373,
        4.37466414146497,
        2.93816398269878,
    )
    coefficients_d = (0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742)
    if probability < 0.02425:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            (
                (
                    ((coefficients_c[0] * q + coefficients_c[1]) * q + coefficients_c[2]) * q
                    + coefficients_c[3]
                )
                * q
                + coefficients_c[4]
            )
            * q
            + coefficients_c[5]
        ) / (
            (
                ((coefficients_d[0] * q + coefficients_d[1]) * q + coefficients_d[2]) * q
                + coefficients_d[3]
            )
            * q
            + 1.0
        )
    if probability > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(
            (
                (
                    ((coefficients_c[0] * q + coefficients_c[1]) * q + coefficients_c[2]) * q
                    + coefficients_c[3]
                )
                * q
                + coefficients_c[4]
            )
            * q
            + coefficients_c[5]
        ) / (
            (
                ((coefficients_d[0] * q + coefficients_d[1]) * q + coefficients_d[2]) * q
                + coefficients_d[3]
            )
            * q
            + 1.0
        )
    q = probability - 0.5
    r = q * q
    return (
        (
            (
                (
                    ((coefficients_a[0] * r + coefficients_a[1]) * r + coefficients_a[2]) * r
                    + coefficients_a[3]
                )
                * r
                + coefficients_a[4]
            )
            * r
            + coefficients_a[5]
        )
        * q
        / (
            (
                (
                    ((coefficients_b[0] * r + coefficients_b[1]) * r + coefficients_b[2]) * r
                    + coefficients_b[3]
                )
                * r
                + coefficients_b[4]
            )
            * r
            + 1.0
        )
    )
