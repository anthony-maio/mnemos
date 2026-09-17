from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.local_memory.graph_measurement import measure_graph_repetition
from benchmarks.local_memory.module_mechanism import FactSpec, GraphCase


def _unit(x: float, y: float) -> list[float]:
    length = (x * x + y * y) ** 0.5
    return [x / length, y / length]


def _case() -> GraphCase:
    context: dict[str, str | None] = {
        "user_profile_id": "profile-a",
        "project_id": "project-a",
        "workspace_id": "workspace-a",
    }
    facts = tuple(
        FactSpec.from_json(
            {
                "id": fact_id,
                "content": content,
                "embedding": embedding,
                "evidence_id": f"event-{fact_id}",
                "scope": "project",
                **context,
            },
            fallback_context=context,
        )
        for fact_id, content, embedding in (
            ("seed", "incident seed investigate worker", _unit(1.0, 0.0)),
            ("bridge", "worker bridge recovery procedure", _unit(0.9, 0.1)),
            ("target", "approved recovery action", _unit(0.8, 0.2)),
            ("noise", "unrelated office schedule", _unit(0.0, 1.0)),
        )
    )
    foreign = FactSpec.from_json(
        {
            "id": "foreign",
            "content": "incident seed approved recovery action",
            "embedding": _unit(1.0, 0.0),
            "evidence_id": "event-foreign",
            "scope": "project",
            "user_profile_id": "profile-b",
            "project_id": "project-b",
        },
        fallback_context=context,
    )
    return GraphCase(
        id="measurement-case",
        split="development",
        template_id="test-template",
        query="incident seed recovery",
        query_embedding=_unit(1.0, 0.0),
        top_k=3,
        target_id="target",
        context=context,
        facts=facts,
        scope_controls=(foreign,),
        seed_id="seed",
        bridge_id="bridge",
    )


def test_measurement_persists_source_evidence_and_byte_identical_copies(tmp_path: Path) -> None:
    result = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=0)
    snapshots = result["artifacts"]["initial_snapshots"]
    assert len({item["sha256"] for item in snapshots.values()}) == 1
    assert all(Path(item["path"]).exists() for item in snapshots.values())
    assert result["source_fixture"]["episodic_events"] == 5
    assert result["source_fixture"]["fact_evidence"] == 5
    assert result["query"]["actual_seed_ids"]
    assert result["correctness_controls"]["edge_cut"]["query_executed"] is True


def test_measurement_repetitions_have_distinct_artifacts_and_scope_controls_are_filtered(
    tmp_path: Path,
) -> None:
    first = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=0)
    second = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=1)
    assert first["artifacts"]["directory"] != second["artifacts"]["directory"]
    assert first["binding"] != second["binding"]
    for result in (first, second):
        assert "foreign" not in result["query"]["direct_union_ids"]
        assert "foreign" not in result["production"]["scope_control_ids_in_graph_results"]
        assert result["correctness_controls"]["insertion_order"]["status"] == "measured"
        report = json.loads(
            Path(result["artifacts"]["measurement_json"]).read_text(encoding="utf-8")
        )
        assert report["binding"] == result["binding"]


def test_measurement_records_cut_graph_and_reverse_insertion_results(tmp_path: Path) -> None:
    result = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=2)
    controls = result["correctness_controls"]
    assert controls["edge_cut"]["query_executed"] is True
    assert controls["insertion_order"]["reverse_query_executed"] is True
    assert Path(result["artifacts"]["edge_cut_sqlite"]).exists()
    assert Path(result["artifacts"]["reverse_insertion_sqlite"]).exists()


def test_completed_measurement_resumes_only_with_valid_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this resume contract test independent of concurrent source edits in
    # the parent runtime files; source binding itself is covered in the report.
    monkeypatch.setattr(
        "benchmarks.local_memory.graph_measurement._source_hashes",
        lambda: {"test": "stable"},
    )
    first = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=0)
    resumed = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=0)
    assert resumed["binding"] == first["binding"]
    assert resumed["artifacts"]["directory"] == first["artifacts"]["directory"]

    graph_path = Path(first["artifacts"]["graph_sqlite"])
    graph_path.write_bytes(graph_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        measure_graph_repetition(_case(), work_dir=tmp_path, repetition=0)


def test_partial_attempt_is_preserved_and_retry_uses_numbered_attempt(tmp_path: Path) -> None:
    partial = tmp_path / "case-measurement-case" / "repetition-000" / "attempt-000"
    partial.mkdir(parents=True)
    marker = partial / "partial.marker"
    marker.write_text("preserve", encoding="utf-8")
    (partial / "measurement.json").write_text(
        json.dumps({"measurement_status": "started"}), encoding="utf-8"
    )

    result = measure_graph_repetition(_case(), work_dir=tmp_path, repetition=0)
    assert result["artifacts"]["directory"].endswith("attempt-001")
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_path_oracle_uses_strongest_allowed_path_without_cycle_amplification() -> None:
    from benchmarks.local_memory.graph_measurement import _independent_activation
    from mnemos.config import SpreadingConfig

    edges = {
        "seed": {"bridge": 0.9, "target": 0.6, "foreign": 1.0},
        "bridge": {"seed": 0.9, "target": 0.95},
        "target": {"seed": 0.6, "bridge": 0.95},
        "foreign": {"target": 1.0},
    }
    measured = _independent_activation(
        edges, "seed", {"seed", "bridge", "target"}, SpreadingConfig(max_visited=1000)
    )
    assert measured["seed"] == 1.0
    assert measured["bridge"] == pytest.approx(0.8 * 0.9)
    assert measured["target"] == pytest.approx(0.8 * 0.9 * 0.8 * 0.95)
    assert "foreign" not in measured
