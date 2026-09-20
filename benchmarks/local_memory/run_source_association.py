"""Run the fixed five-case association diagnostic with frozen local vectors."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .prepare_source_association_vectors import _committed_inputs
from .source_association_fixture import populate_source_association_fixture
from .source_association_lifecycle import run_lifecycle_control
from .source_association_measurement import observe_retrieval, open_measurement_engine
from .source_association_view import RelationArm

ARMS: tuple[RelationArm, ...] = ("semantic", "source_association", "combined")
MANIFEST_SHA256 = "809a3a27ed0052e0fdf872542168af0a38337c1dc42f3d0b2af66d769cee5741"
VECTORS_SHA256 = "bb66da961d731c80a9bb965f72db082371f5faac2f13b32ff268bb408a7420cb"


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "candidate_pool",
            "edges",
            "typed_provenance",
            "candidate_scores",
            "returned_ids",
            "activation",
            "target",
        )
    }


async def _observe_database(
    database: Path,
    case: Mapping[str, Any],
    vectors: Mapping[str, Sequence[float]],
    options: Mapping[str, Any],
    arm: RelationArm | None,
    *,
    cut_fact_id: str | None = None,
) -> dict[str, Any]:
    engine, store = open_measurement_engine(database, case, vectors, options)
    try:
        return await observe_retrieval(
            engine, store, case, vectors, options, arm=arm, cut_fact_id=cut_fact_id
        )
    finally:
        await engine.aclose()


async def measure_repetition(
    case: Mapping[str, Any],
    vectors: Mapping[str, Sequence[float]],
    options: Mapping[str, Any],
    destination: Path,
    repetition: int,
) -> list[dict[str, Any]]:
    """Keep one fixed case's three arms and all control observations."""
    destination.mkdir(parents=True, exist_ok=False)
    canonical = destination / "canonical.sqlite"
    reverse = destination / "reverse.sqlite"
    forward_fixture = populate_source_association_fixture(case, vectors, canonical)
    reverse_fixture = populate_source_association_fixture(
        case, vectors, reverse, insertion_order="reverse"
    )
    # Derive each graph through the normal rebuild API before copying arms.
    for database in (canonical, reverse):
        engine, _ = open_measurement_engine(database, case, vectors, options)
        await engine.aclose()
    _write(destination / "fixtures.json", {"forward": forward_fixture, "reverse": reverse_fixture})
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        database = destination / (arm + ".sqlite")
        reverse_database = destination / (arm + "-reverse.sqlite")
        shutil.copyfile(canonical, database)
        shutil.copyfile(reverse, reverse_database)
        untouched = (
            await _observe_database(database, case, vectors, options, None)
            if arm == "semantic"
            else None
        )
        row = await _observe_database(database, case, vectors, options, arm)
        restarted = await _observe_database(database, case, vectors, options, arm)
        reversed_row = await _observe_database(reverse_database, case, vectors, options, arm)
        cut = await _observe_database(
            database, case, vectors, options, arm, cut_fact_id=str(case["target_id"])
        )
        target_is_seed = row["candidate_pool"]["actual_selected_seed_id"] == case["target_id"]
        row["repetition"] = repetition
        row["controls"].update(
            {
                "restart_pass": _stable_observation(row) == _stable_observation(restarted)
                and all(restarted["controls"].values()),
                "insertion_order_pass": _stable_observation(row)
                == _stable_observation(reversed_row)
                and all(reversed_row["controls"].values()),
                "edge_cut_pass": (
                    cut["candidate_pool"] == row["candidate_pool"]
                    and (target_is_seed or cut["target"]["energy"] == 0)
                    and all(cut["controls"].values())
                ),
                "semantic_parity_pass": (
                    (
                        untouched["candidate_scores"] == row["candidate_scores"]
                        and untouched["returned_ids"] == row["returned_ids"]
                        and all(untouched["controls"].values())
                    )
                    if untouched is not None
                    else None
                ),
            }
        )
        row["control_observations"] = {
            "restart": restarted,
            "reverse_insertion": reversed_row,
            "edge_cut": cut,
            "untouched_semantic": untouched,
            "edge_cut_target_is_seed": target_is_seed,
        }
        rows.append(row)
        _write(destination / (arm + ".json"), row)
    if case.get("lifecycle_actions"):
        lifecycle_database = destination / "lifecycle.sqlite"
        shutil.copyfile(canonical, lifecycle_database)
        lifecycle = run_lifecycle_control(case, vectors, lifecycle_database)
        _write(destination / "lifecycle.json", lifecycle)
        for row, arm in zip(rows, ARMS):
            post_database = destination / (arm + "-after-lifecycle.sqlite")
            shutil.copyfile(lifecycle_database, post_database)
            post = await _observe_database(post_database, case, vectors, options, arm)
            row["control_observations"]["lifecycle"] = lifecycle
            row["control_observations"]["after_lifecycle"] = post
            row["controls"]["lifecycle_pass"] = (
                bool(lifecycle["controls"])
                and all(value is True for value in lifecycle["controls"].values())
                and all(post["controls"].values())
            )
            _write(destination / (arm + ".json"), row)
    return rows


async def run_measurement(manifest_path: Path, vector_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    manifest_path, vector_path, output = (
        manifest_path.resolve(),
        vector_path.resolve(),
        output.resolve(),
    )
    if output.exists() or output.is_relative_to(root):
        raise ValueError("measurement output must be fresh and outside the repository")
    manifest_bytes, vector_bytes = manifest_path.read_bytes(), vector_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("authored case manifest changed after vector preparation")
    if hashlib.sha256(vector_bytes).hexdigest() != VECTORS_SHA256:
        raise ValueError("frozen vector artifact digest mismatch")
    manifest, artifact = json.loads(manifest_bytes), json.loads(vector_bytes)
    if artifact["status"] != "complete" or artifact["manifest"]["sha256"] != MANIFEST_SHA256:
        raise ValueError("vector preparation is incomplete or belongs to different cases")
    # Exact input hashes above bind all text/vector validation performed by the
    # archived independent verifier. No re-embedding or data selection occurs.
    vectors = {value["text"]: value["vector"] for value in artifact["vectors"].values()}
    sources = [manifest_path, vector_path, Path(__file__).resolve()]
    protocol = root / "docs/plans/2026-09-20-graph-association-development-v1.md"
    sources.append(protocol)
    sources.extend(
        root / "benchmarks/local_memory" / name
        for name in (
            "source_association_measurement.py",
            "source_association_fixture.py",
            "source_association_view.py",
            "source_association_lifecycle.py",
            "source_association_diagnostic.py",
            "graph_measurement.py",
            "module_mechanism.py",
            "prepare_source_association_vectors.py",
        )
    )
    sources.extend(sorted((root / "mnemos").rglob("*.py")))
    _, head, hashes = _committed_inputs(root, sources)
    options = manifest["options"]
    if len(manifest["graph_cases"]) != 5 or options["repetitions"] != 3:
        raise ValueError("fixed experiment requires five cases and three repetitions")
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "source-association-measurement-v1",
        "measurement_status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "input_hashes": hashes,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "repository_status_before": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
        ),
        "measurement_inputs_match_head_before": True,
        "manifest_sha256": MANIFEST_SHA256,
        "vector_artifact_sha256": VECTORS_SHA256,
        "model": artifact["model"],
        "model_digest": artifact["expected_digest"],
        "embedding_text_contract": artifact["embedding_text_contract"],
        "options": options,
        "expected_rows": 45,
        "independent_cases": 5,
        "extraction": "synthetic_fixture",
        "quality_acceptance": "not_measured",
        "rows": [],
    }
    report_path = output / "measurement.json"
    _write(report_path, report)
    try:
        for case in manifest["graph_cases"]:
            for repetition in range(3):
                rows = await measure_repetition(
                    case,
                    vectors,
                    options,
                    output / str(case["id"]) / f"repetition-{repetition:02d}",
                    repetition,
                )
                report["rows"].extend(rows)
                _write(report_path, report)
        _, final_head, final_hashes = _committed_inputs(root, sources)
        if final_head != head or final_hashes != hashes:
            raise RuntimeError("measurement source changed while running")
        report["measurement_inputs_match_head_after"] = True
        report["measurement_status"] = "complete"
        report["execution_complete"] = len(report["rows"]) == 45
        report["controls_pass"] = all(
            value is True
            for row in report["rows"]
            for value in row["controls"].values()
            if value is not None
        )
        eligible_ids = {
            case["id"]: {fact["id"] for fact in case["facts"]} for case in manifest["graph_cases"]
        }
        shared_cases = {
            row["case_id"]
            for row in report["rows"]
            if any(
                set(relation["fact_ids"]) <= eligible_ids[row["case_id"]]
                for relation in row["source_diagnostic"]["relations"]
            )
        }
        semantic_controls = [
            row for row in report["rows"] if row["case_id"] == "semantic_only_chain"
        ]
        report["coverage_pass"] = (
            len(shared_cases) >= 3
            and len(semantic_controls) == 9
            and all(not row["source_diagnostic"]["relations"] for row in semantic_controls)
            and report["controls_pass"]
        )
        report["cases_with_shared_source_relations"] = sorted(shared_cases)
        report["summary_by_arm"] = {
            arm: {
                "rows": len(rows := [row for row in report["rows"] if row["arm"] == arm]),
                "direct_target_rows": sum(row["target"]["direct_candidate"] for row in rows),
                "structural_path_rows": sum(row["target"]["structural_path"] for row in rows),
                "target_top5_rows": sum(row["target"]["in_top_k"] for row in rows),
            }
            for arm in ARMS
        }
    except BaseException as exc:
        report.update(
            measurement_status="failed",
            execution_complete=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        report["repository_status_after"] = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
        )
        report["artifacts"] = {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != report_path
        }
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run_measurement(args.manifest, args.vectors, args.output))
    return 0 if report["execution_complete"] and report["coverage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
