"""Derive the lifecycle v3 fixture from the frozen v2 JSONL dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

V2_VERSION = "local-memory-heldout-v2"
V3_VERSION = "local-memory-heldout-v3"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _amend_session(session: dict[str, Any]) -> dict[str, Any]:
    amended = copy.deepcopy(session)
    facts = amended["facts"]
    events = amended["events"]
    superseded = {fact["fact_id"]: fact for fact in facts if fact["gold_status"] == "superseded"}
    forgotten = {fact["fact_id"]: fact for fact in facts if fact["gold_status"] == "forgotten"}
    for index, event in enumerate(events):
        operation = event["lifecycle_operation"]
        if operation == "correct":
            replacement_ids = set(event["evidence_fact_ids"])
            replacement = next(fact for fact in facts if fact["fact_id"] in replacement_ids)
            target = next(iter(superseded.values()))
            target_event = next(
                candidate
                for candidate in events[:index]
                if target["fact_id"] in candidate["evidence_fact_ids"]
            )
            event.update(
                {
                    "target_source_event_id": target_event["event_id"],
                    "target_fact_id": target["fact_id"],
                    "replacement_content": replacement["content"],
                }
            )
        elif operation == "forget":
            target_id = event["evidence_fact_ids"][0]
            target = forgotten[target_id]
            remember = next(
                candidate
                for candidate in events[:index]
                if candidate["lifecycle_operation"] == "remember"
                and target_id in candidate["evidence_fact_ids"]
            )
            event.update(
                {
                    "target_source_event_id": remember["event_id"],
                    "target_fact_id": target["fact_id"],
                }
            )
    return amended


def derive_v3(source_path: Path, destination_path: Path) -> None:
    source_bytes = source_path.read_bytes()
    rows = _read_rows(source_path)
    if not rows or rows[0].get("record_type") != "manifest":
        raise ValueError("v2 dataset must begin with a manifest")
    if rows[0].get("dataset_version") != V2_VERSION:
        raise ValueError("source dataset must be local-memory-heldout-v2")
    manifest = copy.deepcopy(rows[0])
    manifest["dataset_version"] = V3_VERSION
    manifest["amendment_v2_sha256"] = _digest_bytes(source_bytes)
    manifest["amendment_disclosure"] = (
        "The v2 lifecycle labels were previously observed during harness review. "
        "This v3 file is a repaired rerun and is not new unseen held-out evidence."
    )
    manifest["amendment_scope"] = [
        "manifest_version",
        "lifecycle_target_metadata",
        "no_new_events_or_base_source_event_field_edits",
    ]
    manifest["protocol"]["lifecycle_target_contract"] = {
        "version": "v3",
        "correct_target": "earlier user capture or explicit user remember for superseded fact",
        "forget_target": "earlier explicit user remember for forgotten fact",
        "replacement_content": "new supported fact content evidenced by correct action",
        "source_event_preservation": (
            "v2 event order, content, roles, timestamps, operations, and evidence are preserved; "
            "only target fields are added to existing correct or forget actions"
        ),
    }
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination_path}")
    output = [manifest]
    for row in rows[1:]:
        output.append(_amend_session(row) if row.get("record_type") == "session" else row)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    derive_v3(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
