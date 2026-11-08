"""Reviewed category constructions; never fresh acceptance data."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .joint_support_data import audit_groups, request_digest
from .joint_support_expansion import build_expansion
from .joint_support_partitions import build_partitions
from .joint_support_seed import build_seed


def build_coverage():
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

    def add(family, split, suffix, query, candidates, decisions, reference="2026-11-08"):
        identity = family + "-" + suffix
        request = dict(query=query, reference_date=reference, candidates=deepcopy(candidates))
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

    family = "coverage-overlapping-validity"
    source_a = "Schedule A: Station N's sole morning departure is at 08:00 on each date from November 1 through November 10, 2026, inclusive."
    source_b = "Schedule B: Station N's sole morning departure is at 09:00 on each date from November 8 through November 15, 2026, inclusive."
    a = fact(
        "station-eight",
        "Station N's scheduled sole morning departure is at 08:00 on November 1 through 10, 2026.",
        source_a,
    )
    b = fact(
        "station-nine",
        "Station N's scheduled sole morning departure is at 09:00 on November 8 through 15, 2026.",
        source_b,
    )
    for suffix, date, decisions in [
        (
            "before",
            "2026-11-07",
            [
                ("supported", "Only schedule A covers the requested date."),
                ("temporal_mismatch", "Schedule B starts the next day."),
            ],
        ),
        (
            "overlap-start",
            "2026-11-08",
            [
                (
                    "unresolved_conflict",
                    "Both schedules apply and specify different sole departures; no precedence is supplied.",
                ),
                (
                    "unresolved_conflict",
                    "Both schedules apply and specify different sole departures; no precedence is supplied.",
                ),
            ],
        ),
        (
            "overlap-end",
            "2026-11-10",
            [
                (
                    "unresolved_conflict",
                    "The inclusive last day of A still overlaps B with incompatible sole times.",
                ),
                (
                    "unresolved_conflict",
                    "The inclusive last day of A still overlaps B with incompatible sole times.",
                ),
            ],
        ),
        (
            "after",
            "2026-11-11",
            [
                ("temporal_mismatch", "Schedule A has expired for the requested date."),
                ("supported", "Only schedule B covers the requested date."),
            ],
        ),
    ]:
        add(
            family,
            "development",
            suffix,
            "What time is Station N's scheduled sole morning departure today? Neither schedule has authority over the other.",
            [a, b],
            decisions,
            reference=date,
        )
    reports = [
        fact(
            "schedule-a-report",
            "Schedule A lists 08:00 for Station N on November 8, 2026.",
            source_a,
        ),
        fact(
            "schedule-b-report",
            "Schedule B lists 09:00 for Station N on November 8, 2026.",
            source_b,
        ),
    ]
    add(
        family,
        "development",
        "reports",
        "What time does each schedule list for Station N on November 8, 2026?",
        reports,
        [
            (
                "supported",
                "The query asks for A's stated time, not resolution of the actual departure.",
            ),
            (
                "supported",
                "The query asks for B's stated time; reports of distinct schedules are compatible.",
            ),
        ],
    )

    family = "coverage-expiring-dual-signature"
    rule = "Pass R authorizes Room 7 entry if and only if Ana and Bo have both signed it and the visit is no later than November 8, 2026. A signature cannot extend this deadline."
    a = fact(
        "pass-rule",
        "Pass R permits Room 7 entry exactly when both Ana and Bo have signed and the visit is on or before November 8, 2026.",
        rule,
    )
    for suffix, query, reference, reason, why in [
        (
            "at-deadline",
            "Ana and Bo have signed Pass R. May I enter Room 7 today using it?",
            "2026-11-08",
            "supported",
            "Both signature conditions and the inclusive deadline are satisfied.",
        ),
        (
            "expired",
            "Ana and Bo have signed Pass R. May I enter Room 7 today using it?",
            "2026-11-09",
            "supported",
            "The passed deadline independently prohibits entry even though both signatures exist.",
        ),
        (
            "unknown-signature",
            "Ana has signed Pass R; whether Bo signed is unknown. May I enter Room 7 today?",
            "2026-11-08",
            "unresolved_condition",
            "The live deadline cannot resolve the missing second signature state.",
        ),
        (
            "unknown-but-expired",
            "Ana has signed Pass R; whether Bo signed is unknown. May I enter Room 7 today?",
            "2026-11-09",
            "supported",
            "The expired deadline is sufficient for a negative answer regardless of the unknown signature.",
        ),
        (
            "hypothetical",
            "For this question, assume both Ana and Bo signed Pass R and the visit is on November 8, 2026. Would Room 7 entry be permitted?",
            "2026-11-09",
            "supported",
            "The explicitly requested historical hypothetical supplies the visit date and both signatures.",
        ),
    ]:
        add(family, "development", suffix, query, [a], [(reason, why)], reference=reference)

    family = "coverage-ratification-records"
    registry = "Quota registry for Team U: the November quota is the value in a notice signed by both chair and clerk. A notice signed by only one is a proposal. If two notices have both signatures, neither has priority without an explicit replacement record."
    proposed = "Notice P: November 2026 quota for Team U is 60 units; signed by chair only."
    ratified = "Notice Q: November 2026 quota for Team U is 40 units; signed by chair and clerk."
    a = fact(
        "quota-sixty", "Team U's ratified November 2026 quota is 60 units.", registry, proposed
    )
    b = fact(
        "quota-forty", "Team U's ratified November 2026 quota is 40 units.", registry, ratified
    )
    add(
        family,
        "tuning",
        "proposal",
        "What is Team U's ratified November 2026 quota?",
        [a, b],
        [
            (
                "unmet_condition",
                "Notice P lacks the clerk signature required to establish this ratified claim.",
            ),
            (
                "supported",
                "Notice Q meets the registry's two-signature requirement and the proposal is not a competing ratified value.",
            ),
        ],
    )
    both_signed = deepcopy(a)
    both_signed["sources"][1][
        "excerpt"
    ] = "Notice P: November 2026 quota for Team U is 60 units; signed by chair and clerk."
    add(
        family,
        "tuning",
        "two-ratified",
        "What is Team U's ratified November 2026 quota? No replacement record exists.",
        [both_signed, b],
        [
            (
                "unresolved_conflict",
                "Both values have qualifying signatures and neither has precedence.",
            ),
            (
                "unresolved_conflict",
                "Both values have qualifying signatures and neither has precedence.",
            ),
        ],
    )
    uncertain = deepcopy(a)
    uncertain["sources"][1][
        "excerpt"
    ] = "Notice P: November 2026 quota for Team U is 60 units; chair signature verified, clerk signature status unknown."
    add(
        family,
        "tuning",
        "unknown",
        "Is Team U's ratified November 2026 quota 60 units?",
        [uncertain],
        [
            (
                "unresolved_condition",
                "Unknown clerk signature status prevents establishing ratification.",
            )
        ],
    )
    report = fact(
        "quota-proposal-report",
        "Notice P proposes a November 2026 quota of 60 units for Team U.",
        registry,
        proposed,
    )
    add(
        family,
        "tuning",
        "reported",
        "What quota does the single-signature Notice P propose for Team U?",
        [report, a],
        [
            (
                "supported",
                "The attributed proposal is directly established even though it is not ratified.",
            ),
            ("attribution_mismatch", "The target promotes the proposal into a ratified decision."),
        ],
    )
    return examples, labels


def write_coverage(destination: Path):
    destination.mkdir(exist_ok=False)
    examples, labels = build_seed()
    for build in (build_expansion, build_partitions, build_coverage):
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
        status="reviewed_coverage_draft_not_training_ready",
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
