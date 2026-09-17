from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.local_memory.module_mechanism import (
    GRAPH_FACT_COUNT,
    _FrozenEmbedder,
    evaluate_affective_pair,
    evaluate_dataset,
    evaluate_graph_case,
    load_mechanism_dataset,
    prepare_frozen_embeddings,
)
from mnemos.types import CognitiveState
from mnemos.utils.embeddings import EmbeddingProvider


def _unit(*values: float) -> list[float]:
    length = sum(value * value for value in values) ** 0.5
    return [value / length for value in values]


def _graph_case() -> dict:
    facts = []
    for index in range(GRAPH_FACT_COUNT):
        # Thirty high-similarity distractors keep the target out of the direct
        # top-30 pool while the seed -> bridge -> target chain remains strong.
        if index == 0:
            vector = _unit(1, 0)
            content = "incident seed investigate the worker"
        elif index == 1:
            vector = _unit(0.8, 0.6)
            content = "the bridge record connects worker and recovery"
        elif index == 2:
            vector = _unit(0.6, 0.8)
            content = "apply the approved recovery action"
        else:
            vector = _unit(1, (index % 7) / 100)
            content = f"unrelated operational note {index}"
        facts.append(
            {
                "id": f"fact-{index}",
                "content": content,
                "embedding": vector,
                "evidence_id": f"event-{index}",
                "scope": "project",
                "user_profile_id": "profile-a",
                "project_id": "project-a",
                "workspace_id": "workspace-a",
            }
        )
    controls = [
        {
            "id": "foreign-fact",
            "content": "incident seed apply the approved recovery action",
            "embedding": _unit(1, 0),
            "evidence_id": "foreign-event",
            "scope": "project",
            "user_profile_id": "profile-b",
            "project_id": "project-b",
        }
    ]
    return {
        "id": "graph-1",
        "split": "development",
        "template_id": "chain",
        "query": {"text": "incident seed recovery", "embedding": _unit(1, 0)},
        "context": {
            "user_profile_id": "profile-a",
            "project_id": "project-a",
            "workspace_id": "workspace-a",
            "session_id": "s",
            "host": "manual",
        },
        "target_id": "fact-2",
        "seed_id": "fact-0",
        "bridge_id": "fact-1",
        "facts": facts,
        "scope_controls": controls,
    }


def _affective_pair() -> dict:
    context = {"user_profile_id": "profile-a", "project_id": "project-a"}
    state_calm = {"valence": 0.5, "arousal": 0.1, "complexity": 0.8}
    state_urgent = {"valence": -0.6, "arousal": 0.9, "complexity": 0.3}
    return {
        "id": "pair-1",
        "split": "development",
        "template_id": "state",
        "semantic_query": {"text": "service recovery", "embedding": _unit(1, 0)},
        "context": context,
        "facts": [
            {
                "id": "calm",
                "content": "service recovery calm",
                "embedding": _unit(1, 0),
                "evidence_id": "e1",
                "scope": "project",
                **context,
                "state": state_calm,
            },
            {
                "id": "urgent",
                "content": "service recovery urgent",
                "embedding": _unit(1, 0),
                "evidence_id": "e2",
                "scope": "project",
                **context,
                "state": state_urgent,
            },
            {
                "id": "missing",
                "content": "service recovery untagged",
                "embedding": _unit(1, 0),
                "evidence_id": "e3",
                "scope": "project",
                **context,
                "state": None,
            },
        ],
        "no_state_fact_ids": ["missing"],
        "conditions": [
            {
                "query": "calm careful recovery",
                "embedding": _unit(1, 0),
                "state": state_calm,
                "target_id": "calm",
            },
            {
                "query": "urgent simple recovery",
                "embedding": _unit(1, 0),
                "state": state_urgent,
                "target_id": "urgent",
            },
        ],
    }


def test_loader_requires_exactly_64_same_context_graph_facts(tmp_path: Path) -> None:
    payload = {
        "version": "module-mechanism-v1",
        "graph_cases": [_graph_case()],
        "affective_pairs": [],
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = load_mechanism_dataset(path)
    assert len(dataset.graph_cases[0].facts) == 64
    assert len(dataset.graph_cases[0].scope_controls) == 1

    payload["graph_cases"][0]["facts"] = payload["graph_cases"][0]["facts"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 64"):
        load_mechanism_dataset(path)


def test_graph_observation_is_scope_filtered_and_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = load_mechanism_dataset_from_object(
        {"version": "module-mechanism-v1", "graph_cases": [_graph_case()], "affective_pairs": []}
    ).graph_cases[0]
    import sys
    import types

    def measure_graph_repetition(case, *, work_dir, repetition):
        return {
            "case_id": case.id,
            "repetition": repetition,
            "direct_ids": ["fact-0"],
            "scope_control_observations": [],
            "target_in_graph_results": True,
            "target_energy": 0.8,
            "paths": [{"nodes": ["fact-0", "fact-1", "fact-2"], "energy": 0.5}],
            "status": "observed",
        }

    helper = types.ModuleType("benchmarks.local_memory.graph_measurement")
    helper.measure_graph_repetition = measure_graph_repetition
    monkeypatch.setitem(sys.modules, "benchmarks.local_memory.graph_measurement", helper)
    first = evaluate_graph_case(case, work_dir=tmp_path)
    second = evaluate_graph_case(case, work_dir=tmp_path)
    assert first == second
    assert all("foreign-fact" not in row["direct_ids"] for row in first)
    assert all("foreign-fact" not in row["scope_control_observations"] for row in first)
    assert all(
        row["target_incoming_edge_cutset"] == []
        or isinstance(row["target_incoming_edge_cutset"], list)
        for row in first
    )
    assert all(row["repetition"] in {0, 1, 2} for row in first)


def test_affective_ranking_keeps_missing_state_missing() -> None:
    dataset = load_mechanism_dataset_from_object(
        {
            "version": "module-mechanism-v1",
            "graph_cases": [],
            "affective_pairs": [_affective_pair()],
        }
    )
    rows = evaluate_affective_pair(dataset.affective_pairs[0])
    assert len(rows) == 6
    assert {row["affective_top_id"] for row in rows} == {"calm", "missing"}
    assert all(row["no_state_ids_present"] == ["missing"] for row in rows)
    assert all(row["classifier_status"] == "not_run" for row in rows)
    assert all(row["semantic_score_components"] for row in rows)
    assert all(row["affective_score_components"] for row in rows)
    assert all(row["classifier_state"] is None for row in rows)


def test_loader_rejects_null_state_and_context_contract_violations(tmp_path: Path) -> None:
    pair = _affective_pair()
    pair["no_state_fact_ids"] = []
    path = tmp_path / "invalid-null.json"
    path.write_text(
        json.dumps(
            {"version": "module-mechanism-v1", "graph_cases": [], "affective_pairs": [pair]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no_state_fact_ids"):
        load_mechanism_dataset(path)

    pair = _affective_pair()
    pair["facts"][0]["project_id"] = "foreign-project"
    path.write_text(
        json.dumps(
            {"version": "module-mechanism-v1", "graph_cases": [], "affective_pairs": [pair]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="primary facts"):
        load_mechanism_dataset(path)

    case = _graph_case()
    case["answer"] = "answer absent from all facts"
    path.write_text(
        json.dumps(
            {"version": "module-mechanism-v1", "graph_cases": [case], "affective_pairs": []}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target fact"):
        load_mechanism_dataset(path)


def test_embedding_preparation_is_explicit_and_refuses_replacement(tmp_path: Path) -> None:
    raw = {"version": "module-mechanism-v1", "graph_cases": [_graph_case()], "affective_pairs": []}
    source = tmp_path / "raw.json"
    destination = tmp_path / "frozen.json"
    source.write_text(json.dumps(raw), encoding="utf-8")

    class Provider(EmbeddingProvider):
        @property
        def dim(self) -> int:
            return 2

        def embed(self, text: str) -> list[float]:
            return _unit(1, 0)

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed(text) for text in texts]

    prepare_frozen_embeddings(source, destination, Provider())
    loaded = load_mechanism_dataset(destination)
    assert loaded.graph_cases[0].query_embedding is not None
    with pytest.raises(FileExistsError):
        prepare_frozen_embeddings(source, destination, Provider())


def test_checkpoint_keeps_observations_and_rejects_stale_binding(tmp_path: Path) -> None:
    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps(
            {
                "version": "module-mechanism-v1",
                "manifest": {"binding": {"protocol": "p1"}, "affective_pairs": 1},
                "graph_cases": [],
                "affective_pairs": [_affective_pair()],
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)
    checkpoint = Path(report["checkpoint"])
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(payload["affective_rows"]) == 6
    assert payload["repetitions"] == 3
    payload["binding_sha256"] = "stale"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol binding"):
        evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)


def test_affective_report_labels_observations_and_coverage_separately(tmp_path: Path) -> None:
    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps(
            {
                "version": "module-mechanism-v1",
                "graph_cases": [],
                "affective_pairs": [_affective_pair()],
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)
    assert report["status"] == "complete"
    assert report["interpretation"].startswith("observations only")
    assert report["acceptance"]["status"] == "NOT RUN"
    assert report["coverage"]["failed"] is True
    assert report["affective"]["summary"]["incomplete_pair_ids"] == []


def test_graph_checkpoint_resumes_missing_repetitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps(
            {
                "version": "module-mechanism-v1",
                "manifest": {"graph_cases": 1},
                "graph_cases": [_graph_case()],
                "affective_pairs": [],
            }
        ),
        encoding="utf-8",
    )
    calls: list[int] = []
    fail_once = {"value": True}

    def measure_graph_repetition(case, *, work_dir, repetition):
        calls.append(repetition)
        if repetition == 1 and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("interrupted")
        return {
            "target_in_graph_results": True,
            "target_energy": 0.8,
            "paths": [{"nodes": ["a", "b"]}],
            "status": "observed",
        }

    helper = types.ModuleType("benchmarks.local_memory.graph_measurement")
    helper.measure_graph_repetition = measure_graph_repetition
    monkeypatch.setitem(sys.modules, "benchmarks.local_memory.graph_measurement", helper)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)
    checkpoint = json.loads(
        (tmp_path / "work" / "module-mechanism.checkpoint.json").read_text(encoding="utf-8")
    )
    assert [row["repetition"] for row in checkpoint["graph_rows"]] == [0]
    evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)
    assert calls == [0, 1, 1, 2]
    checkpoint = json.loads(
        (tmp_path / "work" / "module-mechanism.checkpoint.json").read_text(encoding="utf-8")
    )
    assert [row["repetition"] for row in checkpoint["graph_rows"]] == [0, 1, 2]


def test_malformed_checkpoint_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps(
            {
                "version": "module-mechanism-v1",
                "graph_cases": [],
                "affective_pairs": [_affective_pair()],
            }
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "module-mechanism.checkpoint.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        evaluate_dataset(source, work_dir=work, repetitions=3)


def test_affective_checkpoint_resumes_missing_condition_repetitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.local_memory.module_mechanism as mechanism

    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps(
            {
                "version": "module-mechanism-v1",
                "manifest": {"affective_pairs": 1},
                "graph_cases": [],
                "affective_pairs": [_affective_pair()],
            }
        ),
        encoding="utf-8",
    )
    calls: list[set[tuple[str, int]]] = []
    interrupted = {"value": True}

    def fake_pair(pair, *, repetitions, llm=None, on_row=None, skip_repetitions=None):
        skipped = set(skip_repetitions or ())
        calls.append(skipped)
        for condition in pair.conditions:
            for repetition in range(repetitions):
                if (condition.target_id, repetition) in skipped:
                    continue
                row = {
                    "pair_id": pair.id,
                    "target_id": condition.target_id,
                    "repetition": repetition,
                    "semantic_target_rank": 1,
                    "affective_target_rank": 1,
                    "classifier_status": "not_run",
                }
                if on_row is not None:
                    on_row(row)
                if interrupted["value"]:
                    interrupted["value"] = False
                    raise RuntimeError("interrupted pair")
        return []

    monkeypatch.setattr(mechanism, "evaluate_affective_pair", fake_pair)
    with pytest.raises(RuntimeError, match="interrupted pair"):
        mechanism.evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)
    mechanism.evaluate_dataset(source, work_dir=tmp_path / "work", repetitions=3)
    assert calls[0] == set()
    assert calls[1] == {("calm", 0)}


def load_mechanism_dataset_from_object(payload: dict) -> object:
    # Keep the test fixture in the same serialized path used by the CLI.
    import tempfile

    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle)
        path = Path(handle.name)
    try:
        return load_mechanism_dataset(path)
    finally:
        path.unlink(missing_ok=True)


def test_graph_correctness_failure_is_retained_for_ineligible_case() -> None:
    from benchmarks.local_memory.module_mechanism import _graph_correctness

    row = {
        "case_id": "negative",
        "repetition": 0,
        "eligible": False,
        "production": {"scope_control_ids_forbidden_observed": ["foreign"]},
        "coverage": {"cycle_observed": True},
        "correctness_controls": {
            "snapshot": {"byte_identical_before_open": True},
            "activation_vs_independent_path": {"matches": False},
            "insertion_order": {"edges_equal": False, "results_equal": True},
        },
    }
    report = _graph_correctness([row])
    assert report["status"] == "FAIL"
    assert {item["check"] for item in report["failures"]} == {
        "scope_isolation",
        "activation_energy",
        "insertion_edges",
    }
    assert report["cycle_observations"] == 1


def test_graph_summary_clusters_repetitions_by_case_and_templates() -> None:
    from benchmarks.local_memory.module_mechanism import _summary_graph

    rows = [
        {
            "case_id": case,
            "template_id": "shared-template",
            "split": "heldout",
            "repetition": repetition,
            "status": "observed",
            "target_id": "target",
            "target_in_graph_results": repetition != 2,
            "target_energy": 0.6,
            "valid_path": True,
            "baseline_result_ids": [],
        }
        for case in ("first", "second")
        for repetition in range(3)
    ]
    report = _summary_graph(rows)
    interval = report["paired_graph_top_k_difference_95"]
    assert interval["cluster_count"] == 1
    assert interval["sample_count"] == 2
    assert interval["estimate"] == pytest.approx(2 / 3)
    assert report["positive_cases"] == 0
    assert report["positive_repetitions"] == 4
