"""Diagnostic selection grading; unavailable output never earns abstention credit."""

from collections import Counter, defaultdict
import json

from .joint_support_data import audit_groups


def grade_selections(examples, labels, observations, split):
    audit = audit_groups(examples, labels)
    if audit["review_status"] != {"reviewed": len(examples)}:
        raise ValueError("Unreviewed labels")
    rows = [e for e in examples if e["split"] == split]
    if not rows:
        raise ValueError("Empty grading partition")
    ids = {e["id"] for e in rows}
    if set(observations) != ids:
        raise ValueError("Every partition example needs exactly one observation")
    counts = Counter(questions=len(rows))
    errors, category_errors = [], Counter()
    critical_outcomes = defaultdict(
        lambda: dict(applicable_cases=[], retained_cases=[], missing_cases=[], unavailable_cases=[])
    )
    permutations = defaultdict(list)
    for example in rows:
        identity = example["id"]
        annotation = labels[identity]["targets"]
        allowed = {t["id"] for t in annotation}
        wanted = {t["id"] for t in annotation if t["supported"]}
        critical = {t["id"] for t in annotation if t["supported"] and t["critical"]}
        observation = observations[identity]
        selected = observation.get("selected_ids")
        valid = (
            observation.get("status") == "assessed"
            and isinstance(selected, list)
            and all(isinstance(s, str) for s in selected)
            and len(set(selected)) == len(selected)
            and set(selected) <= allowed
        )
        chosen = set(selected) if valid else set()
        counts["invalid_or_unavailable"] += not valid
        counts["supported"] += len(chosen & wanted)
        counts["unsupported"] += len(chosen - wanted)
        counts["expected"] += len(wanted)
        counts["critical_expected"] += len(critical)
        counts["critical_retained"] += len(chosen & critical)
        for target in critical:
            outcome = critical_outcomes[target]
            outcome["applicable_cases"].append(identity)
            outcome["retained_cases" if target in chosen else "missing_cases"].append(identity)
            if not valid:
                outcome["unavailable_cases"].append(identity)
        counts["answerable"] += bool(wanted)
        counts["hits"] += bool(chosen & wanted)
        counts["no_answer"] += not wanted
        counts["correct_abstention"] += valid and not wanted and not chosen
        counts["exact_cohorts"] += valid and chosen == wanted
        for target in annotation:
            if target["id"] in chosen - wanted:
                category_errors[target["reason"]] += 1
        if not valid or chosen != wanted:
            errors.append(
                dict(
                    id=identity,
                    valid=valid,
                    expected=sorted(wanted),
                    selected=sorted(chosen),
                    missing=sorted(wanted - chosen),
                    unsupported=sorted(chosen - wanted),
                )
            )
        request = example["request"]
        key = json.dumps(
            {**request, "candidates": sorted(request["candidates"], key=lambda c: c["id"])},
            sort_keys=True,
        )
        permutations[key].append((identity, valid, chosen))

    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    predicted = counts["supported"] + counts["unsupported"]
    metrics = dict(
        precision=counts["supported"] / predicted if predicted else None,
        supported_retention=ratio("supported", "expected"),
        critical_retention=ratio("critical_retained", "critical_expected"),
        answerable_recall=ratio("hits", "answerable"),
        abstention=ratio("correct_abstention", "no_answer"),
        exact_cohorts=ratio("exact_cohorts", "questions"),
    )
    gates = {
        key: metrics[key] is not None and metrics[key] >= minimum
        for key, minimum in {
            "precision": 0.95,
            "supported_retention": 0.95,
            "critical_retention": 0.95,
            "answerable_recall": 0.90,
            "abstention": 0.90,
        }.items()
    }
    gates["all_observations_available"] = counts["invalid_or_unavailable"] == 0
    permutation_groups = [
        dict(ids=[r[0] for r in group], consistent=all(r[1] and r[2] == group[0][2] for r in group))
        for group in permutations.values()
        if len(group) > 1
    ]
    counts["distinct_critical_facts"] = len(critical_outcomes)
    return dict(
        split=split,
        status="diagnostic_thresholds_met" if all(gates.values()) else "diagnostic_failed",
        counts=dict(counts),
        metrics=metrics,
        gates=gates,
        errors=errors,
        false_support_by_reason=dict(category_errors),
        critical_facts={
            identity: outcome for identity, outcome in sorted(critical_outcomes.items())
        },
        permutations=permutation_groups,
        fresh_acceptance=False,
        full_retrieval_timing=False,
    )
