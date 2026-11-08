"""Reviewed development-only quorum/veto and negated-report contrasts."""

from copy import deepcopy

from .joint_support_data import request_digest


def build_contrasts():
    examples, labels = [], {}

    def fact(identity, content, excerpt):
        return dict(
            id=identity,
            version=1,
            content=content,
            assertion_status="asserted",
            sources=[
                dict(id=identity + "-source", source_event_id=identity + "-event", excerpt=excerpt)
            ],
        )

    def add(family, suffix, query, candidates, decisions):
        identity = family + "-" + suffix
        request = dict(query=query, reference_date="2026-11-08", candidates=deepcopy(candidates))
        examples.append(
            dict(
                id=identity,
                split="development",
                known_material=False,
                request=request,
                lineage={
                    k: [family]
                    for k in ("scenario", "source", "template", "paraphrase", "counterfactual")
                },
            )
        )
        labels[identity] = dict(
            request_sha256=request_digest(request),
            review_status="reviewed",
            targets=[
                dict(
                    id=c["id"],
                    supported=reason == "supported",
                    reason=reason,
                    critical=False,
                    rationale=why,
                )
                for c, (reason, why) in zip(candidates, decisions, strict=True)
            ],
        )

    rule_text = (
        "Vault K authorization rule: opening is permitted if and only if at least two "
        "of the three named custodians Ada, Ben and Cy approve and the safety veto is "
        "inactive. An active safety veto forbids opening regardless of approvals. "
        "There are no other authorization conditions."
    )
    rule = fact("vault-quorum-veto-rule", rule_text, rule_text)
    family = "contrast-quorum-veto"
    cases = [
        (
            "unknown-decisive",
            "Ada approves, Ben does not approve, Cy's approval is unknown, "
            "and the veto is inactive.",
            "unresolved_condition",
            "Cy's unknown approval decides whether the quorum of two exists.",
        ),
        (
            "veto-overrides-unknown",
            "Ada approves, Ben does not approve, Cy's approval is unknown, "
            "and the veto is active.",
            "supported",
            "The active veto forbids opening regardless of the unknown quorum.",
        ),
        (
            "quorum-fails-veto-unknown",
            "Ada and Ben do not approve, Cy approves, " "and the veto state is unknown.",
            "supported",
            "Only one custodian approves, so opening is forbidden regardless of veto state.",
        ),
        (
            "quorum-passes-veto-unknown",
            "Ada and Ben approve, Cy does not approve, " "and the veto state is unknown.",
            "unresolved_condition",
            "The quorum exists but the unknown veto state determines authorization.",
        ),
        (
            "authorized",
            "Ada and Ben approve, Cy does not approve, and the veto is inactive.",
            "supported",
            "The quorum and inactive veto establish permission.",
        ),
        (
            "unneeded-third-vote",
            "Ada and Ben approve, Cy's approval is unknown, " "and the veto is inactive.",
            "supported",
            "Two known approvals meet quorum without needing Cy's unknown vote.",
        ),
    ]
    for suffix, circumstances, reason, why in cases:
        add(
            family,
            suffix,
            "Take these circumstances as established: "
            + circumstances
            + " May Vault K be opened under its authorization rule?",
            [rule],
            [(reason, why)],
        )
    add(
        family,
        "hypothetical-veto",
        "In a hypothetical where all three custodians approve but the safety veto is active, "
        "may Vault K be opened?",
        [rule],
        [
            (
                "supported",
                "The explicitly requested hypothetical establishes the veto, which forbids opening.",
            )
        ],
    )
    add(
        family,
        "reported-veto-only",
        "Ada approves and Ben does not approve. Cy's approval and the actual veto state are "
        "unknown. An unverified memo claims 'the veto is active'; do not adopt that claim as "
        "an established circumstance. May Vault K be opened?",
        [rule],
        [
            (
                "unresolved_condition",
                "Neither the quorum nor veto state is established; the report is not an adopted condition.",
            )
        ],
    )
    add(
        family,
        "rule-request",
        "What is Vault K's opening authorization rule?",
        [rule],
        [
            (
                "supported",
                "The source directly states the requested rule; no current authorization decision is requested.",
            )
        ],
    )

    family = "contrast-negated-report"
    denial = (
        'Mira said: "I did not report that tank R was empty." No observation of tank R is provided.'
    )
    report = (
        'Mira said: "I reported that tank R was not empty." No observation of tank R is provided.'
    )
    negative_denial = 'Mira said: "I did not report that tank R was not empty." No observation of tank R is provided.'
    meanings = [
        ("denied-report-empty", "Mira denies having reported that tank R was empty."),
        ("reported-not-empty", "Mira says she reported that tank R was not empty."),
        ("denied-report-not-empty", "Mira denies having reported that tank R was not empty."),
        ("actual-not-empty", "Tank R was not empty."),
    ]
    query = "Which claims about Mira's statement or tank R's state are established by the supplied evidence?"
    for suffix, excerpt, supported in [
        ("denied-positive-report", denial, 0),
        ("affirmed-negative-report", report, 1),
        ("denied-negative-report", negative_denial, 2),
    ]:
        candidates = [fact(f"{suffix}-{name}", content, excerpt) for name, content in meanings]
        decisions = [
            (
                ("supported", "This precisely preserves the scope of Mira's denial or affirmation.")
                if i == supported
                else (
                    (
                        "incomplete_source_support",
                        "A statement about reporting supplies no independent observation of tank R.",
                    )
                    if i == 3
                    else (
                        "attribution_mismatch",
                        "Moving negation across the report changes what Mira said; this claim is not established.",
                    )
                )
            )
            for i in range(4)
        ]
        if suffix == "affirmed-negative-report":
            decisions[3] = (
                "attribution_mismatch",
                "Mira's report of non-emptiness is not an independent observation establishing the actual tank state.",
            )
        add(family, suffix, query, candidates, decisions)
        if suffix == "denied-positive-report":
            add(family, "denial-permutation", query, candidates[::-1], decisions[::-1])
            add(
                family,
                "actual-state-unavailable",
                "Was tank R empty or not empty?",
                [candidates[3]],
                [decisions[3]],
            )
    observed = fact(
        "tank-observed-not-empty",
        "Tank R was not empty at 12:00 on November 8, 2026.",
        "Instrument record: at 12:00 on November 8, 2026, tank R contained 40 liters of water and was not empty.",
    )
    add(
        family,
        "observed-counterpart",
        "Was tank R empty or not empty at 12:00 on November 8, 2026?",
        [observed],
        [
            (
                "supported",
                "A direct instrument record establishes the requested state at the specified time.",
            )
        ],
    )
    return examples, labels
