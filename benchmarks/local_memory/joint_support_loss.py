"""Training-only mass allocation; no inference, threshold fitting or sampling."""

from collections import defaultdict
import math

from .joint_support_data import audit_groups


def training_weights(examples, labels):
    """Balance connected families, then classes, reasons, queries and targets.

    Each family receives equal total mass. Within each family, present binary
    classes receive equal mass. Negative reasons then share the negative mass;
    each reason is divided equally across queries containing it, then targets
    within each query. No weights depend on tuning or development decisions.
    """
    audit = audit_groups(examples, labels)
    if audit["review_status"] != {"reviewed": len(examples)}:
        raise ValueError("Every label must be reviewed")
    training = {e["id"] for e in examples if e["split"] == "train"}
    groups = [sorted(g) for g in audit["groups"] if g[0] in training]
    if not groups:
        raise ValueError("No training families")
    result = {}
    for group in groups:
        # category -> reason -> query -> target IDs
        buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for identity in group:
            for target in labels[identity]["targets"]:
                buckets[target["supported"]][target["reason"]][identity].append(target["id"])
        if not buckets:
            raise ValueError("Training family contains no targets")
        for reasons in buckets.values():
            for queries in reasons.values():
                for identity, targets in queries.items():
                    weight = 1 / (
                        len(groups) * len(buckets) * len(reasons) * len(queries) * len(targets)
                    )
                    for target in targets:
                        result[(identity, target)] = weight
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise ValueError("Training mass must sum to one")
    return result
