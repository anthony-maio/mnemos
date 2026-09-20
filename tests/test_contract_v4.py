from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.local_memory.contracts import MINIMUMS, load_dataset
from benchmarks.local_memory.generate_contract_v4 import derive_v4
from benchmarks.local_memory.ollama_adapter import OllamaEvaluatorAdapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/validation/2026-09-20/retrieval-fixture-audit-v1/dataset-exact.jsonl"
SOURCE_SHA = "036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(
        ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode()
    )


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    path = tmp_path / "v4.jsonl"
    derive_v4(SOURCE, path)
    return path


def test_v4_is_reproducible_and_preserves_counts_and_criteria(
    candidate: Path, tmp_path: Path
) -> None:
    second = tmp_path / "second.jsonl"
    derive_v4(SOURCE, second)
    assert candidate.read_bytes() == second.read_bytes()
    assert b"\r\n" not in candidate.read_bytes()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    dataset = load_dataset(candidate)
    old = load_dataset(SOURCE)
    assert dataset.version == "local-memory-heldout-v4"
    assert len(dataset.sessions) == 220
    assert len(dataset.heldout_sessions) == 200
    assert len(dataset.facts) == 240
    assert len(dataset.state_pairs) == 50
    manifest = dataset.manifest
    assert manifest["answerable_query_count"] == 100
    assert manifest["no_answer_query_count"] == 120
    assert manifest["critical_fact_count"] == 100
    assert manifest["development_session_count"] == 20
    assert manifest["amendment_v3_sha256"] == SOURCE_SHA
    assert "previously observed" in manifest["amendment_disclosure"]
    assert "not new unseen" in manifest["amendment_disclosure"]
    assert manifest["human_acceptance"] == "not_run"
    assert manifest["advertised_as_real"] is False
    assert manifest["protocol"] == old.manifest["protocol"]
    assert MINIMUMS == {
        "heldout_sessions": 200,
        "critical_facts": 100,
        "answerable_queries": 100,
        "no_answer_queries": 100,
        "state_pairs": 50,
    }


def test_v4_changes_only_declared_fixture_fields(candidate: Path) -> None:
    old = _rows(SOURCE)[1:]
    new = _rows(candidate)[1:]
    supported_changes = 0
    annotations = 0
    state_changes = 0
    canary_changes = 0
    for before, after in zip(old, new, strict=True):
        restored = copy.deepcopy(after)
        assert after["events"] == before["events"]
        for original, amended in zip(before["facts"], restored["facts"], strict=True):
            annotation = amended.pop("contract_annotation", None)
            remembers = [
                event
                for event in before["events"]
                if event["lifecycle_operation"] == "remember"
                and original["fact_id"] in event["evidence_fact_ids"]
            ]
            if remembers:
                annotations += 1
                assert annotation == {
                    "explicit_api_body_unscored": remembers[0]["content"],
                    "v3_content_unscored": original["content"],
                    "historical_stale_pattern_preserved": original["gold_status"] == "forgotten",
                }
            else:
                assert annotation is None
            if amended["content"] != original["content"]:
                supported_changes += 1
                assert original["gold_status"] == "supported"
                assert amended["content"] == remembers[0]["content"]
                amended["content"] = original["content"]
        for original, amended in zip(before["state_pairs"], restored["state_pairs"], strict=True):
            if amended["matched_content"] != original["matched_content"]:
                state_changes += 1
                target = next(
                    fact
                    for fact in after["facts"]
                    if fact["fact_id"] in amended["relevant_fact_ids"]
                )
                assert amended["matched_content"] == target["content"]
                amended["matched_content"] = original["matched_content"]
        if len(after["queries"]) != len(before["queries"]):
            canary_changes += 1
            assert len(before["queries"]) == 1
            assert len(after["queries"]) == 2
            foreign, local = after["queries"]
            assert foreign["query_id"] == before["queries"][0]["query_id"]
            assert foreign["kind"] == "no_answer"
            assert foreign["relevant_fact_ids"] == []
            assert foreign["expected_answer"] is None
            assert foreign["target_project_id"] != before["project_id"]
            assert foreign["target_user_profile_id"] == "t18-canary-profile"
            assert local["query_id"] == foreign["query_id"] + "-in-scope-v4"
            assert local["kind"] == "answerable"
            assert not any(key.startswith("target_") for key in local)
            local_old = {key: value for key, value in local.items() if key != "query_id"}
            old_local = {
                key: value
                for key, value in before["queries"][0].items()
                if key != "query_id" and not key.startswith("target_")
            }
            assert local_old == old_local
            restored["queries"] = before["queries"]
        assert restored == before
    assert (supported_changes, annotations, state_changes, canary_changes) == (40, 60, 20, 20)


def test_v4_keeps_forgotten_and_superseded_stale_detection(candidate: Path) -> None:
    old = {session.session_id: session for session in load_dataset(SOURCE).sessions}
    checked = 0
    for session in load_dataset(candidate).sessions:
        for fact in session.facts:
            if fact.gold_status not in {"forgotten", "superseded"}:
                continue
            result = {
                "q": {
                    "facts": [
                        {"content": fact.content},
                        {"content": "Echo: " + fact.content},
                        {"content": "unrelated control"},
                    ]
                }
            }
            assert OllamaEvaluatorAdapter._stale_counts(session, result) == (2, 3)
            assert OllamaEvaluatorAdapter._stale_counts(old[session.session_id], result) == (2, 3)
            checked += 1
    assert checked == 40


def test_v4_state_targets_match_supported_stored_bodies(candidate: Path) -> None:
    for session in load_dataset(candidate).sessions:
        for pair in session.state_pairs:
            fact = next(fact for fact in session.facts if fact.fact_id in pair.relevant_fact_ids)
            assert pair.matched_content == fact.content
            assert (
                OllamaEvaluatorAdapter._state_hit(
                    pair, {"facts": [{"content": fact.content}]}, semantic=True
                )
                is True
            )


def test_v4_generator_refuses_overwrite_and_changed_parent(tmp_path: Path) -> None:
    destination = tmp_path / "preserved.jsonl"
    destination.write_bytes(b"preserve\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        derive_v4(SOURCE, destination)
    assert destination.read_bytes() == b"preserve\n"
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source hash"):
        derive_v4(changed, tmp_path / "must-not-exist.jsonl")
    assert not (tmp_path / "must-not-exist.jsonl").exists()


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("missing_target", "target_source_event_id does not exist"),
        ("late_target", "target event must precede action"),
        ("wrong_context", "target event context mismatch"),
        ("missing_replacement", "correct needs replacement_content"),
        ("forget_replacement", "forget cannot have replacement_content"),
        ("wrong_explicit_body", "explicit remember body"),
        ("foreign_answerable", "outside source scope"),
        ("foreign_profile", "outside source scope"),
        ("no_answer_with_answer", "no_answer cannot have expected_answer"),
        ("wrong_state_target", "state target must match"),
        ("canary_missing_project", "foreign scope canary"),
        ("canary_missing_profile", "foreign scope canary"),
        ("canary_local_project", "foreign scope canary"),
        ("canary_local_profile", "foreign scope canary"),
        ("canary_missing_local_pair", "paired scope canary"),
    ],
)
def test_v4_rejects_contract_regressions(candidate: Path, mutation: str, expected: str) -> None:
    rows = _rows(candidate)
    sessions = {row["session_id"]: row for row in rows[1:]}
    correction = sessions["heldout-041"]["events"][1]
    if mutation == "missing_target":
        correction["target_source_event_id"] = "does-not-exist"
    elif mutation == "late_target":
        correction["target_source_event_id"] = "heldout-041-event-03"
    elif mutation == "wrong_context":
        correction["target_project_id"] = "foreign-project"
    elif mutation == "missing_replacement":
        correction["replacement_content"] = None
    elif mutation == "forget_replacement":
        sessions["heldout-161"]["events"][1]["replacement_content"] = "invalid"
    elif mutation == "wrong_explicit_body":
        sessions["heldout-001"]["facts"][0]["content"] = "normalized but not the stored body"
    elif mutation == "foreign_answerable":
        sessions["heldout-061"]["queries"][1]["target_project_id"] = "canary-project-01"
    elif mutation == "foreign_profile":
        sessions["heldout-061"]["queries"][1]["target_user_profile_id"] = "t18-canary-profile"
    elif mutation == "no_answer_with_answer":
        sessions["heldout-061"]["queries"][0]["expected_answer"] = "Render"
    elif mutation == "wrong_state_target":
        sessions["heldout-001"]["state_pairs"][0]["matched_content"] = "old normalized body"
    elif mutation == "canary_missing_project":
        sessions["heldout-061"]["queries"][0].pop("target_project_id")
    elif mutation == "canary_missing_profile":
        sessions["heldout-061"]["queries"][0].pop("target_user_profile_id")
    elif mutation == "canary_local_project":
        sessions["heldout-061"]["queries"][0]["target_project_id"] = "project-01"
    elif mutation == "canary_local_profile":
        sessions["heldout-061"]["queries"][0]["target_user_profile_id"] = "t18-synthetic-profile"
    elif mutation == "canary_missing_local_pair":
        sessions["heldout-061"]["queries"][1]["query_id"] = "no-longer-the-paired-query"
    _write(candidate, rows)
    with pytest.raises(ValueError, match=expected):
        load_dataset(candidate)
