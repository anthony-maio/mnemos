"""Source-bound critical action constraints, separate from model inputs."""

from collections import Counter
from copy import deepcopy
import hashlib
import json

from .joint_support_data import audit_groups

CRITERION = "critical-action-constraint-v1"


def fact_binding(candidate):
    """Bind identity, version, exact content, status and complete owned evidence."""
    return hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def apply_critical_registry(examples, labels, registry):
    audit_groups(examples, labels)
    if set(registry) != {"criterion", "entries"} or registry["criterion"] != CRITERION:
        raise ValueError("Unknown critical criterion")
    entries = {}
    for entry in registry["entries"]:
        if (
            set(entry) != {"target_id", "fact_sha256", "justification"}
            or entry["target_id"] in entries
            or not entry["justification"].strip()
            or len(entry["fact_sha256"]) != 64
        ):
            raise ValueError("Invalid critical registry entry")
        entries[entry["target_id"]] = entry
    if not entries:
        raise ValueError("Empty critical registry")
    result = deepcopy(labels)
    facts = {
        key: dict(partitions=set(), occurrences=[], supported_opportunities=[]) for key in entries
    }
    for example in examples:
        candidates = {c["id"]: c for c in example["request"]["candidates"]}
        for target in result[example["id"]]["targets"]:
            identity = target["id"]
            target["critical"] = identity in entries
            if identity not in entries:
                continue
            if fact_binding(candidates[identity]) != entries[identity]["fact_sha256"]:
                raise ValueError("Critical fact identity/content/evidence changed: " + identity)
            record = facts[identity]
            record["partitions"].add(example["split"])
            record["occurrences"].append(example["id"])
            if target["supported"]:
                record["supported_opportunities"].append(example["id"])
    if any(not f["occurrences"] or not f["supported_opportunities"] for f in facts.values()):
        raise ValueError("Critical registry entry lacks an applicable supported occurrence")
    if any(len(f["partitions"]) != 1 for f in facts.values()):
        raise ValueError("Critical fact crosses partitions")
    counts, opportunities = Counter(), Counter()
    for record in facts.values():
        record["partitions"] = sorted(record["partitions"])
        split = record["partitions"][0]
        counts[split] += 1
        opportunities[split] += len(record["supported_opportunities"])
    audit_groups(examples, result)
    return result, dict(
        criterion=CRITERION,
        registered_facts_by_partition=dict(counts),
        supported_critical_opportunities_by_partition=dict(opportunities),
        facts=facts,
        metric_scope="Critical selection retention only; no extraction or durable-retention result",
        measured=False,
    )
