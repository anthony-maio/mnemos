"""Split/label integrity checks, not semantic validation of training labels."""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import unicodedata

from .joint_support_frame import serialize_joint_cohort

SPLITS = {"train", "tuning", "development"}
LINEAGES = {"scenario", "source", "template", "paraphrase", "counterfactual"}
REASONS = {
    "supported",
    "wrong_subject_relation",
    "incomplete_source_support",
    "attribution_mismatch",
    "unresolved_condition",
    "unmet_condition",
    "temporal_mismatch",
    "unresolved_conflict",
    "other_adjudicated",
}


def request_digest(request: Mapping) -> str:
    # Byte encoding validates the schema only; this is not a model token count.
    frame = serialize_joint_cohort(
        request,
        lambda s: list(s.encode("utf-8")),
        tokenizer_sha256="0" * 64,
        special_ids=frozenset(),
        max_tokens=100_000,
    )
    return hashlib.sha256(frame.payload_json.encode("utf-8")).hexdigest()


def _content_digest(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_groups(examples: Sequence[Mapping], labels: Mapping) -> dict:
    """Reject cross-split ancestry and report coverage without inventing gold.

    IDs and lineage are outside model requests. Exact normalized fact/source
    content adds overlap edges even when examples rename their ancestry.
    Similarity beyond normalization still requires an independent label audit.
    """
    if not examples or len(examples) > 10_000:
        raise ValueError("Expected a bounded nonempty corpus")
    parent = list(range(len(examples)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    ids, splits = [], []
    coverage = {split: Counter() for split in SPLITS}
    positive = Counter()
    critical = Counter()
    reviews = Counter()
    bindings = {}
    decisions = {}
    for index, example in enumerate(examples):
        if set(example) != {"id", "split", "lineage", "known_material", "request"}:
            raise ValueError("Invalid corpus example fields")
        identity, split, lineage = example["id"], example["split"], example["lineage"]
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or identity in ids
            or split not in SPLITS
        ):
            raise ValueError("Invalid example identity/split")
        if type(example["known_material"]) is not bool:
            raise ValueError("Known-material declaration required")
        if example["known_material"] and split != "train":
            raise ValueError("Known material cannot enter reserved audit partitions")
        if not isinstance(lineage, Mapping) or set(lineage) != LINEAGES:
            raise ValueError("Complete lineage inventory required")
        keys = []
        for kind, values in lineage.items():
            if (
                not isinstance(values, list)
                or len(values) != len(set(values))
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or (kind in {"scenario", "source", "template"} and not values)
            ):
                raise ValueError("Invalid lineage family")
            keys.extend((kind, v) for v in values)
        request = example["request"]
        digest = request_digest(request)
        keys.append(("request_digest", digest))
        for candidate in request["candidates"]:
            keys.append(("content", _content_digest(candidate["content"])))
            keys.extend(("content", _content_digest(s["excerpt"])) for s in candidate["sources"])
        for key in keys:
            if key in owners:
                parent[root(index)] = root(owners[key])
            else:
                owners[key] = index
        annotation = labels.get(identity)
        if not isinstance(annotation, Mapping) or set(annotation) != {
            "request_sha256",
            "review_status",
            "targets",
        }:
            raise ValueError("Separate bound label record required")
        if annotation["request_sha256"] != digest:
            raise ValueError("Labels do not bind the current request")
        if annotation["review_status"] not in {"pending", "reviewed", "disputed"}:
            raise ValueError("Invalid review status")
        reviews[annotation["review_status"]] += 1
        targets = annotation["targets"]
        if not isinstance(targets, list) or len(targets) != len(request["candidates"]):
            raise ValueError("Every target requires exactly one label")
        seen = set()
        for target in targets:
            if set(target) != {"id", "supported", "reason", "critical", "rationale"}:
                raise ValueError("Invalid target label fields")
            if (
                target["id"] in seen
                or target["id"] not in {c["id"] for c in request["candidates"]}
                or type(target["supported"]) is not bool
                or type(target["critical"]) is not bool
                or target["reason"] not in REASONS
                or (target["reason"] == "supported") != target["supported"]
                or not isinstance(target["rationale"], str)
                or not target["rationale"].strip()
            ):
                raise ValueError("Invalid or inconsistent target label")
            seen.add(target["id"])
            key = (digest, target["id"])
            decision = (target["supported"], target["reason"], target["critical"])
            if key in decisions and decisions[key] != decision:
                raise ValueError("Contradictory labels for identical target input")
            decisions[key] = decision
            coverage[split][target["reason"]] += 1
            positive[split] += target["supported"]
            critical[split] += target["supported"] and target["critical"]
        ids.append(identity)
        splits.append(split)
        bindings[identity] = digest
    if set(labels) != set(ids):
        raise ValueError("Orphan label records")
    groups = defaultdict(list)
    for index in range(len(examples)):
        groups[root(index)].append(index)
    for members in groups.values():
        if len({splits[i] for i in members}) != 1:
            raise ValueError(
                "Connected leakage family crosses splits: " + ", ".join(ids[i] for i in members)
            )
    counts = Counter(splits)
    independent = Counter(splits[members[0]] for members in groups.values())
    ready = all(
        counts[s] and positive[s] and sum(coverage[s].values()) > positive[s] for s in SPLITS
    )
    return dict(
        status="structural_audit_passed",
        examples=dict(counts),
        independent_groups=dict(independent),
        reason_counts={s: dict(v) for s, v in coverage.items()},
        supported_critical_targets=dict(critical),
        review_status=dict(reviews),
        minimum_partition_checks=ready and reviews["pending"] == reviews["disputed"] == 0,
        semantic_labels_verified=False,
        training_ready=False,
        request_bindings=bindings,
        groups=[[ids[i] for i in members] for members in groups.values()],
    )
