"""Generate the fixed synthetic mechanism cases before observing embeddings.

These authored labels describe fixture intent. They are not human judgments of
model output. The ten held-out templates limit generalization; repetitions and
renamed variants are never additional independent real-world workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# situation, seed dependency, bridge asset, exact source-backed target action
SCENARIOS = (
    (
        "checkout authorizations pile up",
        "payment intake worker",
        "lease registry",
        "clear the expired claim before resuming the worker",
    ),
    (
        "search results stop updating",
        "catalog indexing worker",
        "generation marker",
        "restore the last validated marker before resuming indexing",
    ),
    (
        "uploaded photographs remain blank",
        "thumbnail rendering worker",
        "color profile cache",
        "reload the approved profile bundle before rerendering thumbnails",
    ),
    (
        "outbound email delivery stalls",
        "notification dispatch worker",
        "sender permit ledger",
        "renew the sender permit before restarting dispatch",
    ),
    (
        "nightly backups fail verification",
        "archive verification worker",
        "manifest signature store",
        "restore the trusted public key before verifying the archive again",
    ),
    (
        "calendar invitations appear twice",
        "schedule synchronization worker",
        "delivery cursor register",
        "reset the cursor to the last acknowledged invitation",
    ),
    (
        "package downloads return stale files",
        "artifact distribution worker",
        "release alias table",
        "point the release alias at the verified immutable digest",
    ),
    (
        "sensor readings arrive out of order",
        "telemetry ingestion worker",
        "clock offset journal",
        "apply the recorded clock correction before replaying the readings",
    ),
    (
        "support tickets lose attachments",
        "case attachment worker",
        "object retention catalog",
        "restore the object retention rule before retrying attachment links",
    ),
    (
        "warehouse reservations stop clearing",
        "inventory allocation worker",
        "reservation epoch register",
        "advance the epoch after reconciling outstanding reservations",
    ),
)

COMPETITOR_NOTES = (
    "dashboards refresh every minute",
    "the operations team owns the alert",
    "logs use UTC timestamps",
    "the staging environment uses separate credentials",
    "the weekly report excludes test traffic",
    "the status page groups warnings by region",
    "the audit log records operator identity",
    "the troubleshooting guide includes a dependency map",
    "the health probe runs once per minute",
    "daily volume is summarized after midnight",
    "traces retain the original request identifier",
    "load tests run in a disposable environment",
    "the reporting view omits archived accounts",
    "the review checklist includes rollback ownership",
    "failed requests keep their correlation identifier",
    "the documentation records expected response shapes",
    "the capacity chart includes queued work",
    "the team reviews incident trends each month",
    "the sandbox uses synthetic customer records",
    "the dashboard separates traffic by deployment",
    "the alert message includes a runbook link",
    "the audit report groups records by event date",
    "the client sends an idempotency identifier",
    "the service account has a named owner",
    "the test suite includes malformed input examples",
    "the tracing view links parent and child operations",
    "the service catalog records escalation contacts",
    "the release checklist includes a dry run",
    "the reporting job records completion time",
    "the daily chart distinguishes successes and failures",
    "the test environment has a separate endpoint",
)

BACKGROUND_FACTS = (
    "The design team stores logo source files as SVG.",
    "Documentation examples use four-space indentation.",
    "The onboarding checklist names an equipment owner.",
    "The office inventory lists spare keyboards.",
    "Meeting notes use an ISO date in their filenames.",
    "The training library keeps an index of recorded sessions.",
    "The accessibility review checks keyboard focus visibility.",
    "The glossary expands abbreviations on first use.",
    "The expense template groups receipts by month.",
    "The research notebook labels exploratory results.",
    "The internal handbook lists room booking rules.",
    "The travel checklist includes an emergency contact.",
    "The illustration library includes dark and light backgrounds.",
    "The welcome guide links the team directory.",
    "The spelling dictionary includes product names.",
    "The diagram template includes a revision date.",
    "The procurement form records the delivery address.",
    "The workshop agenda leaves time for questions.",
    "The slide template uses a readable body font.",
    "The feedback form accepts anonymous suggestions.",
    "The office map labels accessible entrances.",
    "The editorial calendar records the planned author.",
    "The learning plan includes a practice exercise.",
    "The reading list separates books from articles.",
    "The interview guide records the session duration.",
    "The event checklist includes captioning arrangements.",
    "The design review records the chosen color palette.",
    "The handover template names an incoming owner.",
    "The stationery cupboard contains blank index cards.",
    "The workshop materials include a printable worksheet.",
)

CALM = {"valence": 0.4, "arousal": 0.1, "complexity": 0.8}
URGENT = {"valence": -0.6, "arousal": 0.9, "complexity": 0.3}


def _context(identifier: str) -> dict[str, Any]:
    return {
        "user_profile_id": "mechanism-profile",
        "project_id": identifier + "-project",
        "workspace_id": identifier + "-workspace",
        "session_id": identifier,
        "host": "manual",
    }


def _fact(
    identifier: str,
    content: str,
    context: dict[str, Any],
    *,
    scope: str = "project",
    state: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "content": content,
        "evidence_id": identifier + "-source",
        "scope": scope,
        "user_profile_id": context["user_profile_id"],
        "project_id": context["project_id"],
        "workspace_id": context["workspace_id"],
        "state": state,
    }


def make_graph_case(index: int, *, split: str) -> dict[str, Any]:
    identifier = f"{split}-graph-{index:03d}"
    context = _context(identifier)
    scenario = (
        SCENARIOS[index % len(SCENARIOS)]
        if split == "heldout"
        else (
            "preview documents fail to render",
            "document conversion worker",
            "font mapping register",
            "restore the approved font mapping before rendering the preview again",
        )
    )
    situation, worker, asset, action = scenario
    variant = (
        f"deployment {index // len(SCENARIOS) + 1}"
        if split == "heldout"
        else f"preview lab {index + 1}"
    )
    prefix = f"For {variant}, "
    query = f"When {situation} in {variant}, what single recovery action should the operator take?"
    seed = _fact(
        identifier + "-seed",
        prefix + f"when {situation}, investigate the {worker} and its dependent control record.",
        context,
    )
    bridge = _fact(
        identifier + "-bridge",
        prefix
        + f"the {worker} relies on the {asset}; restoring this worker requires the procedure attached to that record.",
        context,
    )
    target = _fact(
        identifier + "-target", prefix + f"the {asset} procedure is to {action}.", context
    )
    facts = [seed, bridge, target]
    for number, note in enumerate(COMPETITOR_NOTES):
        facts.append(
            _fact(
                f"{identifier}-competitor-{number:02d}",
                prefix + f"the guide for cases where {situation} records that {note}.",
                context,
            )
        )
    for number, content in enumerate(BACKGROUND_FACTS):
        facts.append(_fact(f"{identifier}-background-{number:02d}", content, context))
    controls = [
        _fact(
            identifier + "-shared-global",
            "The operator handbook requires an audit entry after a recovery action.",
            context,
            scope="global",
        ),
        _fact(
            identifier + "-shared-workspace",
            prefix + f"the workspace service catalog names the {worker} as a shared dependency.",
            context,
            scope="workspace",
        ),
        _fact(
            identifier + "-foreign-project",
            query + " The recorded action belongs to a different project.",
            {**context, "project_id": identifier + "-foreign-project", "workspace_id": None},
        ),
        _fact(
            identifier + "-foreign-profile",
            query + " The recorded action belongs to a different user profile.",
            {
                **context,
                "user_profile_id": "mechanism-foreign-profile",
                "project_id": identifier + "-foreign-profile-project",
                "workspace_id": None,
            },
        ),
    ]
    return {
        "id": identifier,
        "split": split,
        "template_id": (
            f"graph-template-{index % 10}" if split == "heldout" else "graph-development-template"
        ),
        "query": query,
        "context": context,
        "target_id": target["id"],
        "seed_id": seed["id"],
        "bridge_id": bridge["id"],
        "answer": action,
        "facts": facts,
        "scope_controls": controls,
    }


def make_affective_pair(index: int, *, split: str) -> dict[str, Any]:
    identifier = f"{split}-affective-{index:03d}"
    context = _context(identifier)
    situation = (
        SCENARIOS[index % len(SCENARIOS)][0]
        if split == "heldout"
        else "preview documents fail to render"
    )
    situation += (
        f" in deployment {index // len(SCENARIOS) + 1}"
        if split == "heldout"
        else f" in preview lab {index + 1}"
    )
    query = f"What should I do when {situation}?"
    facts = [
        _fact(
            identifier + "-calm",
            f"When {situation}, use the planned investigation: compare traces, reproduce the fault in staging, and review the root cause with the team.",
            context,
            state=CALM,
        ),
        _fact(
            identifier + "-urgent",
            f"When {situation}, use the incident response: contain the affected requests, apply the approved recovery action, and confirm service health.",
            context,
            state=URGENT,
        ),
        _fact(
            identifier + "-missing", f"The runbook explains what to do when {situation}.", context
        ),
    ]
    for number, note in enumerate(COMPETITOR_NOTES[:5]):
        facts.append(
            _fact(
                f"{identifier}-other-{number}",
                f"The guide for cases where {situation} records that {note}.",
                context,
                state={"valence": 0.0, "arousal": 0.5, "complexity": 0.5},
            )
        )
    return {
        "id": identifier,
        "split": split,
        "template_id": (
            f"affective-template-{index % 10}"
            if split == "heldout"
            else "affective-development-template"
        ),
        "semantic_query": query,
        "context": context,
        "facts": facts,
        "conditions": [
            {
                "query": "I feel calm and optimistic, with time for a careful and technically detailed investigation. "
                + query,
                "state": CALM,
                "target_id": facts[0]["id"],
            },
            {
                "query": "I feel worried and urgently need a short, simple incident response right now. "
                + query,
                "state": URGENT,
                "target_id": facts[1]["id"],
            },
        ],
    }


def build_dataset() -> dict[str, Any]:
    return {
        "version": "module-mechanism-v1",
        "label_provenance": "authored synthetic fixture states and source facts; not human outcome judgments",
        "generalization_limit": "50 heldout cases per mechanism use ten scenario templates and five named variants; repeated templates are not independent real-world workflows",
        "graph_cases": [
            make_graph_case(i, split=split)
            for split, count in (("development", 5), ("heldout", 50))
            for i in range(count)
        ],
        "affective_pairs": [
            make_affective_pair(i, split=split)
            for split, count in (("development", 5), ("heldout", 50))
            for i in range(count)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to replace an existing dataset; use a new versioned path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_dataset(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
