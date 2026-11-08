"""Reviewed audit constructions; population independence remains unproven."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .joint_support_data import audit_groups, request_digest
from .joint_support_expansion import build_expansion
from .joint_support_seed import build_seed


def build_partitions():
    examples, labels = [], {}

    def fact(identity, content, *sources):
        return dict(
            id=identity,
            version=1,
            content=content,
            assertion_status="asserted",
            sources=[
                dict(id=f"{identity}-s{i}", source_event_id=f"{identity}-e{i}", excerpt=s)
                for i, s in enumerate(sources)
            ],
        )

    def add(family, split, suffix, query, candidates, decisions):
        identity = family + "-" + suffix
        request = dict(query=query, reference_date="2026-11-08", candidates=deepcopy(candidates))
        examples.append(
            dict(
                id=identity,
                split=split,
                known_material=False,
                request=request,
                lineage={
                    kind: [family]
                    for kind in ("scenario", "source", "template", "paraphrase", "counterfactual")
                },
            )
        )
        labels[identity] = dict(
            request_sha256=request_digest(request),
            review_status="reviewed",
            targets=[
                dict(
                    id=c["id"], supported=r == "supported", reason=r, critical=False, rationale=why
                )
                for c, (r, why) in zip(candidates, decisions, strict=True)
            ],
        )

    family = "audit-reused-asset-alias"
    register = "Asset registry: label Finch referred to serial 41 until replacement on 2026-10-01; from that date Finch refers to serial 92."
    a = fact(
        "finch-current",
        "From October 1 through November 8, 2026, the asset labeled Finch has a 7 liter tank.",
        register,
        "Serial 92 specification, valid from 2026-10-01 through 2026-11-08: tank capacity 7 liters.",
    )
    b = fact(
        "finch-transferred",
        "The asset currently labeled Finch has a 4 liter tank.",
        register,
        "Serial 41 specification, valid throughout September 2026: tank capacity 4 liters.",
    )
    c = fact(
        "finch-old",
        "The asset labeled Finch in September 2026 had a 4 liter tank.",
        register,
        "Serial 41 specification, valid throughout September 2026: tank capacity 4 liters.",
    )
    add(
        family,
        "tuning",
        "current",
        "What is the tank capacity of the asset currently labeled Finch?",
        [a, b, c],
        [
            (
                "supported",
                "The current alias maps to serial 92, whose own specification establishes 7 liters.",
            ),
            (
                "wrong_subject_relation",
                "The 4 liter specification belongs to the replaced serial, not the current asset.",
            ),
            ("temporal_mismatch", "The September asset is not the currently requested asset."),
        ],
    )
    add(
        family,
        "tuning",
        "historical",
        "What was the tank capacity of the asset labeled Finch in September 2026?",
        [a, b, c],
        [
            (
                "temporal_mismatch",
                "The current serial's capacity does not answer the September identity.",
            ),
            (
                "wrong_subject_relation",
                "This target incorrectly assigns the old serial's capacity to the current identity.",
            ),
            (
                "supported",
                "The registry and serial 41 specification jointly establish the historical capacity.",
            ),
        ],
    )
    incomplete = deepcopy(a)
    incomplete["sources"] = incomplete["sources"][1:]
    add(
        family,
        "tuning",
        "unbound",
        "What is the tank capacity of the asset currently labeled Finch?",
        [incomplete],
        [
            (
                "incomplete_source_support",
                "The retained specification does not bind serial 92 to the requested label.",
            )
        ],
    )
    add(
        family,
        "tuning",
        "boundary",
        "What is the tank capacity of the asset labeled Finch on October 1, 2026?",
        [a, c],
        [
            (
                "supported",
                "The replacement date belongs to serial 92 under the explicit inclusive registry rule.",
            ),
            ("temporal_mismatch", "The September identity ceased to hold on the replacement date."),
        ],
    )

    family = "audit-specific-exception-precedence"
    rule = "Shipping rule for Box V: delivery is permitted to all zones except zone 6. A zone 6 delivery is permitted only with permit Q. No other delivery restrictions apply."
    a = fact(
        "box-rule",
        "Box V may be delivered outside zone 6; delivery to zone 6 requires permit Q.",
        rule,
    )
    b = fact("box-overgeneralized", "Box V may be delivered to every zone without a permit.", rule)
    for suffix, query, reason, why in [
        (
            "general",
            "May Box V be delivered to zone 2 without permit Q?",
            "supported",
            "The exception applies only to zone 6; the source explicitly excludes other restrictions.",
        ),
        (
            "exception",
            "May Box V be delivered to zone 6 without permit Q?",
            "supported",
            "The explicit exception requires a permit, supporting a negative answer.",
        ),
        (
            "unknown",
            "May Box V be delivered to zone 6? The permit status is unknown.",
            "unresolved_condition",
            "The exception makes unknown permit status decisive.",
        ),
        (
            "scope",
            "May Box W be delivered to zone 2?",
            "wrong_subject_relation",
            "The rule is limited to Box V and cannot establish permission for Box W.",
        ),
    ]:
        add(
            family,
            "tuning",
            suffix,
            query,
            [a, b],
            [
                (reason, why),
                (
                    "wrong_subject_relation" if suffix == "scope" else "incomplete_source_support",
                    "The target either addresses a different box or removes the source's explicit zone exception.",
                ),
            ],
        )

    family = "audit-choice-versus-conjunction"
    rule = "Build J packaging specification: a conforming package contains exactly one insert, either foam or felt. Neither material is preferred. A package containing both inserts is nonconforming."
    a = fact("build-foam", "A single foam insert is a conforming choice for Build J.", rule)
    b = fact("build-felt", "A single felt insert is a conforming choice for Build J.", rule)
    c = fact("build-both", "Build J requires both a foam and a felt insert in each package.", rule)
    d = fact("build-unique", "Foam is the only conforming insert material for Build J.", rule)
    add(
        family,
        "development",
        "alternatives",
        "Which single-insert materials conform to Build J's specification?",
        [a, b, c, d],
        [
            ("supported", "Foam is one explicitly allowed single-insert option."),
            (
                "supported",
                "Felt is another allowed option; it does not conflict with foam as an alternative.",
            ),
            ("incomplete_source_support", "The source expressly rejects the target's conjunction."),
            (
                "incomplete_source_support",
                "The source permits felt too, contradicting exclusive foam eligibility.",
            ),
        ],
    )
    add(
        family,
        "development",
        "preferred",
        "Which material does the Build J specification prefer?",
        [a, b],
        [
            (
                "wrong_subject_relation",
                "Being permitted does not establish preference; the source states neither is preferred.",
            ),
            (
                "wrong_subject_relation",
                "Being permitted does not establish preference; the source states neither is preferred.",
            ),
        ],
    )
    combined = fact(
        "build-no-both", "Using foam and felt together is nonconforming for Build J.", rule
    )
    add(
        family,
        "development",
        "both",
        "Would a package containing foam and felt together conform to Build J?",
        [combined, c],
        [
            ("supported", "The explicit prohibition establishes a negative answer."),
            ("incomplete_source_support", "The target reverses the specification's prohibition."),
        ],
    )
    add(
        family,
        "development",
        "reordered",
        "Which single-insert materials conform to Build J's specification?",
        [d, b, c, a],
        [
            (
                "incomplete_source_support",
                "Felt is also allowed, so the exclusivity claim is unsupported.",
            ),
            ("supported", "The source permits felt as a single insert."),
            (
                "incomplete_source_support",
                "The source expressly rejects using both inserts together.",
            ),
            ("supported", "The source permits foam as a single insert."),
        ],
    )

    family = "audit-nested-retraction"
    transcript = "Final hearing transcript: Iris said, 'Yesterday I said Sol approved Gate T. I retract that claim: I have no evidence Sol approved it.' Sol did not testify. The hearing records no approval decision."
    a = fact("gate-retraction", "Iris retracted her claim that Sol approved Gate T.", transcript)
    b = fact("gate-approval", "Sol approved Gate T.", transcript)
    c = fact("gate-denial", "Sol rejected Gate T.", transcript)
    add(
        family,
        "development",
        "current",
        "What approval decision did Sol actually make about Gate T?",
        [a, b, c],
        [
            ("wrong_subject_relation", "Iris's retraction is not a decision made by Sol."),
            (
                "attribution_mismatch",
                "A retracted nested report does not establish Sol's actual approval.",
            ),
            (
                "incomplete_source_support",
                "Absence of approval evidence does not establish a rejection decision.",
            ),
        ],
    )
    add(
        family,
        "development",
        "retraction",
        "What did Iris retract at the final hearing?",
        [a, b, c],
        [
            ("supported", "The target preserves the speaker and retraction of the nested claim."),
            (
                "attribution_mismatch",
                "The target asserts the retracted proposition as fact rather than identifying Iris's retraction.",
            ),
            ("incomplete_source_support", "No rejection decision appears in the transcript."),
        ],
    )
    historical = fact(
        "gate-earlier-report",
        "Iris previously claimed that Sol approved Gate T.",
        transcript,
        "Earlier hearing transcript, before the final hearing: Iris said, 'Sol approved Gate T.'",
    )
    add(
        family,
        "development",
        "history",
        "What did Iris claim before retracting it?",
        [historical],
        [
            (
                "supported",
                "The retraction does not erase evidence that Iris previously made this attributed claim.",
            )
        ],
    )
    add(
        family,
        "development",
        "unknown",
        "Does the final hearing establish that Sol rejected Gate T?",
        [c],
        [
            (
                "incomplete_source_support",
                "The target's rejection assertion is unsupported even though the source can explain why no decision is established.",
            )
        ],
    )
    return examples, labels


def write_partitions(destination: Path):
    destination.mkdir(exist_ok=False)
    examples, labels = build_seed()
    for build in (build_expansion, build_partitions):
        added, annotations = build()
        examples.extend(added)
        labels.update(annotations)
    audit = audit_groups(examples, labels)
    for name, value in (
        ("inputs.json", examples),
        ("labels.json", labels),
        ("structural-audit.json", audit),
    ):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest = dict(
        status="reviewed_partition_draft_not_training_ready",
        examples=len(examples),
        targets=sum(len(v["targets"]) for v in labels.values()),
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.iterdir()
            if p.is_file()
        },
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
