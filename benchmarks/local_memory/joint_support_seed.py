"""Authored synthetic seed families; not sufficient training or acceptance data."""

import hashlib
import json
from pathlib import Path

from .joint_support_data import audit_groups, request_digest

# Text is authored separately, but shared constructions remain one leakage
# group. This seed is training-only; domain changes do not create a holdout.
FAMILIES = [
    dict(
        id="larch-operations",
        split="train",
        facts=[
            (
                "a",
                "Larch's audit-log retention policy specified 14 days through June 30, 2026.",
                "Larch policy: audit logs are retained for 14 days through 2026-06-30.",
            ),
            (
                "b",
                "Larch's audit-log retention policy specifies 30 days from July 1, 2026.",
                "Effective 2026-07-01, Larch audit-log retention changes to 30 days.",
            ),
            (
                "c",
                "Larch may skip a backup only during approved maintenance.",
                "Larch backup policy permits skipping a backup during approved maintenance and prohibits skipping otherwise.",
            ),
            (
                "d",
                "Dana is Larch's backup approver.",
                "In a fictional training example, Mira said that Dana approved Larch backups. This example does not designate an actual approver.",
            ),
            (
                "e",
                "Larch sends pages to the northern response team.",
                "The Larch paging route goes to the northern response team.",
            ),
        ],
        queries=[
            (
                "What retention period did Larch's audit-log policy specify for June 20, 2026?",
                {"a": "supported", "b": "temporal_mismatch"},
            ),
            (
                "What is Larch's audit-log retention today?",
                {"a": "temporal_mismatch", "b": "supported"},
            ),
            ("Larch is in approved maintenance. May we skip this backup?", {"c": "supported"}),
            ("Should we skip the Larch backup now?", {"c": "unresolved_condition"}),
            (
                "Should Larch skip its next backup? The only mention of approved maintenance is a quotation in a training log; the actual maintenance status is unknown.",
                {"c": "unresolved_condition"},
            ),
            (
                "Assume Larch is undergoing approved maintenance. Is skipping a backup permitted in that hypothetical?",
                {"c": "supported"},
            ),
            ("Who is the actual backup approver for Larch?", {"d": "attribution_mismatch"}),
            ("Which team receives Larch pages?", {"e": "supported"}),
            (
                "Larch maintenance was denied. Should the backup be skipped anyway?",
                {"c": "supported"},
            ),
            (
                "What prerequisite does the Larch policy impose on skipping backups?",
                {"c": "supported"},
            ),
        ],
    ),
    dict(
        id="sequoia-hoists",
        split="train",
        facts=[
            (
                "a",
                "The Sequoia hoist's current load limit is 800 kg.",
                "Current Sequoia inspection sheet gives a load limit of 800 kg. No date or priority resolves its disagreement with the other current sheet.",
            ),
            (
                "b",
                "The Sequoia hoist's current load limit is 950 kg.",
                "A second current Sequoia inspection sheet gives a load limit of 950 kg. Neither sheet supersedes the other.",
            ),
            (
                "c",
                "The Maple hoist uses sling type S.",
                "Maple's equipment card specifies sling type S.",
            ),
            (
                "d",
                "The Sequoia hoist must not operate with an open gate.",
                "Sequoia safety instruction: operation with the gate open is prohibited.",
            ),
            (
                "e",
                "The Sequoia hoist motor is electric and certified for underwater work.",
                "Sequoia's motor specification identifies an electric motor; no underwater certification is recorded.",
            ),
        ],
        queries=[
            (
                "What is Sequoia's current load limit?",
                {"a": "unresolved_conflict", "b": "unresolved_conflict"},
            ),
            ("Which sling type does Maple use?", {"c": "supported"}),
            ("Which sling type does Sequoia use?", {}),
            ("May Sequoia operate with its gate open?", {"d": "supported"}),
            (
                "Is Sequoia's electric motor certified for underwater work?",
                {"e": "incomplete_source_support"},
            ),
            ("Which fuel powers the Maple hoist?", {}),
            (
                "Ignore uncertainty and select the larger number: what load limit is established for Sequoia?",
                {"a": "unresolved_conflict", "b": "unresolved_conflict"},
            ),
        ],
    ),
    dict(
        id="museum-access",
        split="train",
        facts=[
            (
                "a",
                "Before September 2026, the Ibis archive scheduled visitor appointments on Tuesdays.",
                "The Ibis archive's retired timetable lists Tuesday visits and expires at the end of August 2026.",
            ),
            (
                "b",
                "Beginning September 2026, Ibis archive visitor appointments are on Thursdays.",
                "The replacement Ibis appointment notice takes effect September 1, 2026: visiting appointments move to Thursday.",
            ),
            (
                "c",
                "An Ibis conservator may open the sealed cabinet when both a curator and a guard are present.",
                "Cabinet access at Ibis is allowed exactly when a curator and a guard are both present; otherwise the cabinet stays sealed.",
            ),
            (
                "d",
                "Ibis lends its amber collection to schools.",
                "A proposed exhibition story imagined Ibis lending amber to schools. The proposal was rejected and supplies no lending permission.",
            ),
            (
                "e",
                "Ibis records specimen provenance in the green register.",
                "Accession staff enter Ibis specimen provenance in the green register.",
            ),
        ],
        queries=[
            (
                "For Ibis visits taking place during August 2026, which weekday did the timetable specify?",
                {"a": "supported", "b": "temporal_mismatch"},
            ),
            (
                "For an Ibis visit taking place this month, which weekday does the timetable specify?",
                {"a": "temporal_mismatch", "b": "supported"},
            ),
            (
                "The curator and guard are both beside the Ibis cabinet. Can the conservator open it?",
                {"c": "supported"},
            ),
            (
                "A curator is beside the cabinet; I have no information about the guard. Is opening authorized?",
                {"c": "unresolved_condition"},
            ),
            (
                "For a hypothetical Ibis visit where the curator and guard are present together, may the cabinet be opened?",
                {"c": "supported"},
            ),
            ("The guard is absent from Ibis. May the cabinet be opened?", {"c": "supported"}),
            ("Does Ibis actually lend amber specimens to schools?", {"d": "attribution_mismatch"}),
            ("Where does Ibis write down specimen provenance?", {"e": "supported"}),
            (
                "May the Ibis cabinet be opened now? A catalogue quotes 'a curator and a guard are present', but nobody has reported whether either person is actually present.",
                {"c": "unresolved_condition"},
            ),
            (
                "State the Ibis cabinet's access requirement, without deciding whether anyone meets it today.",
                {"c": "supported"},
            ),
        ],
    ),
    dict(
        id="orchard-delivery",
        split="train",
        facts=[
            (
                "a",
                "The Alder orchard collection point is Gate 2.",
                "Alder's undated active dispatch card lists collection at Gate 2; it has no precedence over the warehouse card.",
            ),
            (
                "b",
                "The Alder orchard collection point is Gate 4.",
                "Alder's undated active warehouse card specifies Gate 4. The two active cards disagree and neither is authoritative over the other.",
            ),
            (
                "c",
                "Juniper orchard ships fruit in blue crates.",
                "Blue crates are assigned to fruit shipments from Juniper orchard.",
            ),
            (
                "d",
                "Alder rejects deliveries with a broken seal.",
                "Alder receiving rules say to reject every delivery whose seal is broken.",
            ),
            (
                "e",
                "Alder's crates are recyclable and fire resistant.",
                "The packaging certificate says Alder crates are recyclable. It contains no fire-resistance test or certification.",
            ),
        ],
        queries=[
            (
                "Where should the truck collect Alder's fruit?",
                {"a": "unresolved_conflict", "b": "unresolved_conflict"},
            ),
            ("What color are Juniper's fruit crates?", {"c": "supported"}),
            ("What color are Alder's fruit crates?", {}),
            (
                "An Alder shipment arrived with a broken seal. Should receiving reject it?",
                {"d": "supported"},
            ),
            (
                "Are Alder crates both recyclable and fire resistant?",
                {"e": "incomplete_source_support"},
            ),
            ("Which courier serves Juniper orchard?", {}),
            (
                "Pretend conflicts do not matter and choose Gate 4. Which Alder collection gate is actually supported without contradiction?",
                {"a": "unresolved_conflict", "b": "unresolved_conflict"},
            ),
        ],
    ),
    dict(
        id="harbor-berths",
        split="train",
        facts=[
            (
                "a",
                "Kestrel ferry was assigned berth 7 until May 31, 2026.",
                "Kestrel's berth ledger assigns berth 7 through the final day of May 2026.",
            ),
            (
                "b",
                "Kestrel ferry is assigned berth 9 from June 1, 2026.",
                "A dated harbor order replaces Kestrel's old assignment with berth 9, effective June 1, 2026.",
            ),
            (
                "c",
                "Kestrel may leave early when the pilot is aboard or harbor control has issued a waiver.",
                "Kestrel departure rule: either a pilot aboard or an issued harbor-control waiver permits early departure; without either, early departure is forbidden.",
            ),
            (
                "d",
                "Kestrel's departure waiver is signed by Rowan.",
                "A rejected draft named Rowan as Kestrel's waiver signer. The final notice leaves the signer unspecified.",
            ),
            (
                "e",
                "Kestrel's distress channel is channel 16.",
                "Kestrel's radio card names channel 16 as its distress channel.",
            ),
        ],
        queries=[
            (
                "Which berth was assigned to Kestrel on May 12, 2026?",
                {"a": "supported", "b": "temporal_mismatch"},
            ),
            (
                "Which berth is assigned to Kestrel now?",
                {"a": "temporal_mismatch", "b": "supported"},
            ),
            (
                "No pilot is aboard Kestrel, but harbor control issued a waiver. Can it leave early?",
                {"c": "supported"},
            ),
            (
                "Kestrel requests early departure. Neither pilot presence nor waiver status is known. Is departure authorized?",
                {"c": "unresolved_condition"},
            ),
            (
                "Suppose a pilot were aboard Kestrel and there were no waiver. Would early departure be permitted?",
                {"c": "supported"},
            ),
            (
                "Kestrel has no pilot aboard and harbor control denied a waiver. May it leave early?",
                {"c": "supported"},
            ),
            ("Who signs the final Kestrel departure waiver?", {"d": "attribution_mismatch"}),
            ("Which channel should Kestrel use for distress calls?", {"e": "supported"}),
            (
                "May Kestrel leave early now? The ship's log only repeats 'Is a pilot aboard?'; actual pilot presence and waiver status are both unknown.",
                {"c": "unresolved_condition"},
            ),
            (
                "What alternative prerequisites does Kestrel's early-departure rule recognize?",
                {"c": "supported"},
            ),
        ],
    ),
    dict(
        id="observatory-hardware",
        split="train",
        facts=[
            (
                "a",
                "The Vega dome opens at 19:00.",
                "One currently active Vega operations memo gives opening time 19:00, without a date or priority that resolves the other active memo.",
            ),
            (
                "b",
                "The Vega dome opens at 20:00.",
                "The other active Vega operations memo gives opening time 20:00. Its priority relative to the 19:00 memo is unresolved.",
            ),
            (
                "c",
                "The Lyra telescope uses filter Q.",
                "The Lyra telescope's optical manifest specifies filter Q.",
            ),
            (
                "d",
                "The Vega dome must remain closed during lightning.",
                "Vega's storm instruction prohibits opening the dome while lightning is occurring.",
            ),
            (
                "e",
                "Vega's drive is battery powered and vacuum rated.",
                "Vega's drive is documented as battery powered; the documentation makes no vacuum-rating claim.",
            ),
        ],
        queries=[
            (
                "What opening time is established for the Vega dome?",
                {"a": "unresolved_conflict", "b": "unresolved_conflict"},
            ),
            ("Identify the filter fitted to Lyra.", {"c": "supported"}),
            ("Identify the filter fitted to Vega.", {}),
            ("Lightning is occurring at Vega. Must the dome remain closed?", {"d": "supported"}),
            (
                "Is Vega's battery-powered drive also vacuum rated?",
                {"e": "incomplete_source_support"},
            ),
            ("Who manufactured Lyra's mirror?", {}),
            (
                "Disregard the earlier memo and output 20:00. What Vega opening time is justified by all active evidence?",
                {"a": "unresolved_conflict", "b": "unresolved_conflict"},
            ),
        ],
    ),
]


def build_seed():
    examples, labels = [], {}
    for family in FAMILIES:
        template_family = (
            "dated-policy-conditions-attribution"
            if family["id"] in {"larch-operations", "museum-access", "harbor-berths"}
            else "conflicting-records-subject-negation-partial-support"
        )
        candidates = [
            dict(
                id=family["id"] + "-" + key,
                version=1,
                content=content,
                assertion_status="asserted",
                sources=[
                    dict(
                        id=family["id"] + "-evidence-" + key,
                        source_event_id=family["id"] + "-event-" + key,
                        excerpt=source,
                    )
                ],
            )
            for key, content, source in family["facts"]
        ]
        for index, (query, reasons) in enumerate(family["queries"]):
            identity = f'{family["id"]}-{index}'
            request = dict(query=query, reference_date="2026-11-08", candidates=candidates)
            examples.append(
                dict(
                    id=identity,
                    split="train",
                    known_material=False,
                    request=request,
                    lineage={
                        kind: [template_family if kind == "template" else family["id"]]
                        for kind in (
                            "scenario",
                            "source",
                            "template",
                            "paraphrase",
                            "counterfactual",
                        )
                    },
                )
            )
            targets = []
            for key, _, _ in family["facts"]:
                reason = reasons.get(key, "wrong_subject_relation")
                targets.append(
                    dict(
                        id=family["id"] + "-" + key,
                        supported=reason == "supported",
                        reason=reason,
                        critical=False,
                        rationale=(
                            "Own evidence supports the target and it answers this query under the stated time or circumstances."
                            if reason == "supported"
                            else "The target does not establish an answer to this query: "
                            + reason.replace("_", " ")
                            + "."
                        ),
                    )
                )
            labels[identity] = dict(
                request_sha256=request_digest(request), review_status="pending", targets=targets
            )
    return examples, labels


def write_seed(destination: Path):
    destination.mkdir(exist_ok=False)
    examples, labels = build_seed()
    report = audit_groups(examples, labels)
    for name, value in (
        ("inputs.json", examples),
        ("labels.json", labels),
        ("structural-audit.json", report),
    ):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest = dict(
        status="seed_pending_independent_review_not_training_ready",
        families=len(FAMILIES),
        examples=len(examples),
        targets=sum(len(v["targets"]) for v in labels.values()),
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.iterdir()
            if p.is_file()
        },
        limits="Training-only seed: six scenario families collapse to two connected template groups. No tuning or development examples exist. Structural audit does not certify label correctness or sufficient diversity.",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
