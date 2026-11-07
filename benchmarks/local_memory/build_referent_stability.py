"""Author a development corpus with fixed semantic labels and matched views."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def build_corpus() -> dict[str, Any]:
    same, different, insufficient = "same_variable", "different_variable", "insufficient_identity"
    plain = "The Juniper snapshot window is Monday at 21:00."
    production = "The Juniper production snapshot window is Monday at 21:00."
    staging = "The Juniper staging snapshot window is Monday at 21:00."
    workers = "The Harbor worker pool has twelve workers."
    fixtures = "The Harbor test set contains twelve fixtures."
    rule = "Include device serial numbers in every Finch incident report."
    # Each row supplies its semantic labels and operation explicitly. Variant
    # construction changes presentation, never the annotation for a literal pair.
    seeds = [
        (
            "schedule-basic",
            "schedule",
            False,
            "Correction: the Juniper snapshot window is Tuesday at 21:00, not Monday.",
            [("primary", plain, same)],
            "correct",
            "primary",
            "The named schedule is unchanged; the day is the corrected value.",
        ),
        (
            "schedule-qualified",
            "schedule",
            False,
            "Correction: the Juniper production snapshot window is Tuesday at 21:00, not Monday.",
            [("primary", staging, different), ("secondary", production, same)],
            "correct",
            "secondary",
            "Production identifies the second target; staging is an incompatible environment.",
        ),
        (
            "schedule-wrong-environment",
            "schedule",
            False,
            "Correction: the Juniper production snapshot window is Tuesday at 21:00, not Monday.",
            [("primary", staging, different)],
            "unresolved",
            None,
            "The correction names production, while the only schedule target names staging.",
        ),
        (
            "schedule-missing-environment",
            "schedule",
            False,
            "Correction: the Juniper snapshot window is Tuesday at 21:00, not Monday.",
            [("primary", production, insufficient), ("secondary", staging, insufficient)],
            "unresolved",
            None,
            "Neither environment is specified by the candidate, and both remain plausible.",
        ),
        (
            "count-explicit",
            "numeric",
            False,
            "Correction: the Harbor worker pool needs eighteen workers, not twelve.",
            [("primary", workers, same)],
            "correct",
            "primary",
            "The worker pool and its size are identified; only the number changes.",
        ),
        (
            "count-vague",
            "numeric",
            False,
            "Correction: the count should be eighteen, not twelve.",
            [("primary", workers, insufficient), ("secondary", fixtures, insufficient)],
            "unresolved",
            None,
            "The candidate does not identify what is counted; a matching old value is insufficient.",
        ),
        (
            "requirement-polarity",
            "report",
            True,
            "From now on, omit device serial numbers from every Finch incident report.",
            [("primary", rule, same)],
            "correct",
            "primary",
            "Omit reverses include for the same data and report rule.",
        ),
        (
            "requirement-missing-environment",
            "schedule",
            True,
            "From now on, the Juniper snapshot window must be Tuesday at 21:00 instead of Monday.",
            [("primary", production, insufficient), ("secondary", staging, insufficient)],
            "unresolved",
            None,
            "The prospective requirement still lacks the environment needed to replace a rule.",
        ),
        (
            "requirement-vague-count",
            "numeric",
            True,
            "From now on, the count must be eighteen instead of twelve.",
            [("primary", workers, insufficient), ("secondary", fixtures, insufficient)],
            "unresolved",
            None,
            "An unidentified count does not become a new independent requirement.",
        ),
        (
            "requirement-unrelated",
            "report",
            True,
            "From now on, omit device serial numbers from every Finch incident report.",
            [("primary", "Harbor releases require two reviewers.", different)],
            "add",
            None,
            "The complete report rule is independent of the observed release-reviewer requirement.",
        ),
        (
            "retention-paraphrase",
            "retention",
            False,
            "Correction: Morrow build artifacts should be kept for thirty days, not seven.",
            [("primary", "The retention period for Morrow build artifacts is seven days.", same)],
            "correct",
            "primary",
            "Kept for a duration and retention period identify the same property of the artifacts.",
        ),
        (
            "requirement-different-audience",
            "repository",
            True,
            "From now on, require review for changes in the Atlas public repository.",
            [("primary", "Changes in the Atlas private repository require no review.", different)],
            "add",
            None,
            "The complete public-repository rule does not replace a private-repository rule.",
        ),
    ]
    distractors = [
        ("distractor-format", "The Granite artifact export format is TIFF.", different),
        ("distractor-theme", "The Quartz public status page uses the violet theme.", different),
    ]
    canonical = {
        "primary": "fact-10",
        "secondary": "fact-20",
        "distractor-format": "fact-70",
        "distractor-theme": "fact-90",
    }
    renamed = {
        "primary": "item-831",
        "secondary": "item-204",
        "distractor-format": "item-557",
        "distractor-theme": "item-069",
    }
    cases = []
    for base, cluster, requirement, content, targets, operation, chosen, rationale in seeds:
        augmented = [distractors[0], *targets, distractors[1]]
        variants = [
            ("base", targets, canonical),
            ("renamed", targets, renamed),
            ("augmented", augmented, canonical),
            ("reversed", list(reversed(augmented)), canonical),
            ("augmented-renamed", augmented, renamed),
            ("reversed-renamed", list(reversed(augmented)), renamed),
        ]
        for variant, visible, aliases in variants:
            source_id = base + "-source"
            scope = {
                "scope": "project",
                "scope_id": "referent-development",
                "profile_id": "referent-profile",
            }
            cases.append(
                {
                    "id": base + "/" + variant,
                    "base_id": base,
                    "variant": variant,
                    "cluster_id": cluster,
                    "events": [
                        {
                            "id": source_id,
                            "role": "user",
                            "content": content,
                            **scope,
                            "source": "codex_hook",
                        }
                    ],
                    "targets": [
                        {"id": aliases[logical], "content": text, "version": 4, **scope}
                        for logical, text, _ in visible
                    ],
                    "expected": [],
                    "selections": [
                        {
                            "source_id": source_id,
                            "first_unit": 0,
                            "last_unit": 0,
                            "kind": (
                                "continuing_requirement"
                                if requirement
                                else "correction_or_rejection"
                            ),
                            "retention_request_unit": None,
                            "revises_existing_requirement": requirement,
                        }
                    ],
                    "expected_alternatives": [
                        {
                            "candidate_index": 0,
                            "operation": operation,
                            "target_fact_id": aliases[chosen] if chosen is not None else None,
                            "expected_version": 4 if chosen is not None else None,
                            "reason": (
                                "The referent requires review."
                                if operation == "unresolved"
                                else None
                            ),
                        }
                    ],
                    "expected_relations": {
                        aliases[logical]: relation for logical, _, relation in visible
                    },
                    "logical_target_ids": {aliases[logical]: logical for logical, _, _ in visible},
                    "annotation_rationale": rationale,
                }
            )
    return deepcopy(
        {
            "schema_version": 1,
            "version": "referent-stability-development-v1",
            "stage": "development",
            "evaluation": "preclassified_reconciliation",
            "relation_annotations": True,
            "scope": "Twelve authored seed scenarios in five clusters, each with six matched presentations. Seventy-two views are not independent scenarios. Relations and final operations are labeled before model output. Development diagnostics, not fresh acceptance.",
            "cases": cases,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(build_corpus(), stream, indent=2, ensure_ascii=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
