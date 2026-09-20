"""Derive a disclosed contract repair from the exact, previously observed v3 input."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import load_dataset

SOURCE_SHA256 = "036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc"
VERSION = "local-memory-heldout-v4"


def derive_v4(source_path: Path, destination_path: Path) -> None:
    """Preserve source events and old artifacts; write a new LF-only dataset."""
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination_path}")
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("source hash must match the audited frozen v3 input")
    original = load_dataset(source_path)
    if original.version != "local-memory-heldout-v3":
        raise ValueError("source must be the audited v3 dataset")
    rows: list[dict[str, Any]] = [json.loads(line) for line in source.splitlines()]
    manifest = rows[0]
    manifest.update(
        dataset_version=VERSION,
        amendment_v3_sha256=SOURCE_SHA256,
        no_answer_query_count=120,
        amendment_disclosure=(
            "The v3 fixtures and labels were previously observed. This v4 contract repair "
            "is a repaired rerun, not new unseen held-out evidence. Original v3 results "
            "remain unchanged. No model output is used to select replacement gold text."
        ),
        amendment_scope=[
            "40_supported_explicit_fact_bodies_equal_existing_api_bodies",
            "60_unscored_explicit_body_annotations",
            "20_state_matched_content_values_follow_supported_fact_bodies",
            "20_foreign_queries_become_no_answer",
            "20_new_in_scope_answerable_queries",
            "all_historical_stale_patterns_preserved",
            "all_source_events_and_lifecycle_targets_preserved",
            "all_existing_thresholds_and_scoring_functions_preserved",
        ],
    )
    supported_changes = annotations = state_changes = canary_changes = 0
    for session in rows[1:]:
        facts = {fact["fact_id"]: fact for fact in session["facts"]}
        changed_ids = set()
        for event in session["events"]:
            if event["lifecycle_operation"] != "remember":
                continue
            for fact_id in event["evidence_fact_ids"]:
                fact = facts[fact_id]
                fact["contract_annotation"] = {
                    "explicit_api_body_unscored": event["content"],
                    "v3_content_unscored": fact["content"],
                    "historical_stale_pattern_preserved": fact["gold_status"] == "forgotten",
                }
                annotations += 1
                if fact["gold_status"] == "supported":
                    fact["content"] = event["content"]
                    changed_ids.add(fact_id)
                    supported_changes += 1
        for pair in session["state_pairs"]:
            changed = set(pair["relevant_fact_ids"]) & changed_ids
            if changed:
                if len(changed) != 1:
                    raise ValueError("state target must identify exactly one amended fact")
                pair["matched_content"] = facts[next(iter(changed))]["content"]
                state_changes += 1
        added_queries = []
        for query in session["queries"]:
            if (
                query["kind"] == "answerable"
                and query.get("target_project_id") is not None
                and query["target_project_id"] != session["project_id"]
            ):
                local = copy.deepcopy(query)
                local["query_id"] += "-in-scope-v4"
                for field in ("target_user_profile_id", "target_project_id", "target_workspace_id"):
                    local.pop(field, None)
                added_queries.append(local)
                query.update(kind="no_answer", relevant_fact_ids=[], expected_answer=None)
                canary_changes += 1
        session["queries"].extend(added_queries)
    if (supported_changes, annotations, state_changes, canary_changes) != (40, 60, 20, 20):
        raise ValueError("amendment coverage differs from the independently audited contract")
    output = (
        "\n".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) for row in rows) + "\n"
    ).encode()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("xb") as stream:
        stream.write(output)
    load_dataset(destination_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    derive_v4(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
