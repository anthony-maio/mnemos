"""Supplemental synthetic fixture for module-utility comparisons.

The lifecycle dataset intentionally remains the frozen T18 v2 contract. Its
singleton sessions cannot establish graph traversal or affective rank shifts.
This loader validates a separate challenge fixture with competing candidates;
it is not human-accepted conversation data and is never advertised as such.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import HeldoutDataset, _parse_session

MODULE_CHALLENGE_VERSION = "local-memory-module-challenge-v1"
MODULE_CHALLENGE_MIN_PAIRS = 50
MODULE_CHALLENGE_MIN_CANDIDATES = 3


def load_module_challenge_dataset(path: Path) -> HeldoutDataset:
    """Load the supplemental challenge rows without applying lifecycle minima."""
    manifest: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number}: row must be an object")
        if item.get("record_type") == "manifest":
            if manifest is not None:
                raise ValueError("challenge dataset may contain only one manifest")
            manifest = item
        else:
            rows.append(item)
    if manifest is None or manifest.get("dataset_version") != MODULE_CHALLENGE_VERSION:
        raise ValueError("module challenge manifest version is unsupported")
    if (
        manifest.get("source_label") != "synthetic_module_challenge"
        or manifest.get("human_acceptance") != "not_run"
        or manifest.get("advertised_as_real") is not False
    ):
        raise ValueError("module challenge must retain synthetic labeling")
    sessions = tuple(_parse_session(item, index) for index, item in enumerate(rows))
    heldout = tuple(session for session in sessions if session.split == "heldout")
    development_ids = tuple(str(value) for value in manifest.get("development_session_ids", []))
    if len(heldout) < MODULE_CHALLENGE_MIN_PAIRS:
        raise ValueError("module challenge needs at least 50 held-out sessions")
    if set(development_ids) != {
        session.session_id for session in sessions if session.split == "development"
    }:
        raise ValueError("module challenge development IDs must be disjoint and exact")
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError("module challenge session IDs must be unique")
    pair_count = 0
    for session in heldout:
        fact_ids = {fact.fact_id for fact in session.facts}
        for pair in session.state_pairs:
            pair_count += 1
            candidates = set(pair.candidate_fact_ids)
            if len(candidates) < MODULE_CHALLENGE_MIN_CANDIDATES:
                raise ValueError(f"state pair {pair.pair_id} needs three candidates")
            direct_seeds = set(pair.direct_vector_seed_fact_ids)
            graph_targets = set(pair.graph_target_fact_ids)
            if (
                not candidates <= fact_ids
                or not graph_targets <= candidates
                or not direct_seeds <= candidates
                or graph_targets & direct_seeds
            ):
                raise ValueError(f"state pair {pair.pair_id} has invalid candidate target IDs")
            if not set(pair.relevant_fact_ids) <= candidates:
                raise ValueError(f"state pair {pair.pair_id} target is outside candidates")
    if pair_count < MODULE_CHALLENGE_MIN_PAIRS:
        raise ValueError("module challenge needs at least 50 state pairs")
    declared = {
        "heldout_session_count": len(heldout),
        "development_session_count": len(sessions) - len(heldout),
        "state_pair_count": pair_count,
    }
    for field, actual in declared.items():
        if manifest.get(field) != actual:
            raise ValueError(f"module challenge manifest {field} does not match parsed count")
    return HeldoutDataset(
        version=MODULE_CHALLENGE_VERSION,
        sessions=sessions,
        development_session_ids=development_ids,
        digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest=manifest,
        source_path=path,
    )
