"""Run both frozen embedding contracts through the unchanged association controls."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .embedding_prefix_development_v1 import MANIFEST_PATH, load_cases
from .prepare_source_association_vectors import DIGEST, _committed_inputs, _vectors
from .run_source_association import ARMS, measure_repetition
from .source_association_role_preparation_v1 import PREFIXES, corrected_contents

VECTORS_SHA256 = "c5e0e073859e8343885f0d4a3659ca70de3afd44d83590160a9a9b279396c444"


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_role_vectors(manifest: dict[str, Any], roles: dict[str, Any]) -> None:
    if set(roles) != set(PREFIXES):
        raise ValueError("embedding contracts changed")
    queries = {case["query"] for case in manifest["graph_cases"]}
    documents = {
        fact["content"]
        for case in manifest["graph_cases"]
        for fact in [*case["facts"], *case.get("scope_controls", [])]
    }
    documents.update(corrected_contents(manifest))
    for label, data in roles.items():
        if data.get("prefixes") != PREFIXES[label]:
            raise ValueError("embedding role prefixes changed")
        if set(data.get("queries", {})) != queries or set(data.get("documents", {})) != documents:
            raise ValueError("role vector coverage is incomplete or unexpected")
        for role in ("queries", "documents"):
            _vectors({"embeddings": list(data[role].values())}, len(data[role]))


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        (role, case["id"], arm, rep)
        for role in PREFIXES
        for case in manifest["graph_cases"]
        for arm in ARMS
        for rep in range(3)
    }
    actual = [
        (row["embedding_contract"], row["case_id"], row["arm"], row["repetition"]) for row in rows
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("measurement rows do not match the scheduled grid")
    failures = [
        {
            **{k: row[k] for k in ("embedding_contract", "case_id", "arm", "repetition")},
            "control": key,
            "value": value,
        }
        for row in rows
        for key, value in row["controls"].items()
        if value is not None and value is not True
    ]
    eligible = {
        case["id"]: {fact["id"] for fact in case["facts"]} for case in manifest["graph_cases"]
    }
    coverage = {}
    summaries: dict[str, Any] = {}
    for role in PREFIXES:
        selected = [row for row in rows if row["embedding_contract"] == role]
        shared_cases = sorted(
            {
                row["case_id"]
                for row in selected
                if any(
                    set(relation["fact_ids"]) <= eligible[row["case_id"]]
                    for relation in row["source_diagnostic"]["relations"]
                )
            }
        )
        negative = [row for row in selected if row["case_id"] == "semantic_only_chain"]
        coverage[role] = {
            "cases_with_shared_source_relations": shared_cases,
            "semantic_only_negative_control": len(negative) == 9
            and all(not row["source_diagnostic"]["relations"] for row in negative),
            "shared_case_coverage": len(shared_cases) >= 3,
        }
        summaries[role] = {}
        for arm in ARMS:
            arm_rows = [row for row in selected if row["arm"] == arm]
            summaries[role][arm] = {
                "rows": len(arm_rows),
                "direct_target_rows": sum(row["target"]["direct_candidate"] for row in arm_rows),
                "structural_path_rows": sum(row["target"]["structural_path"] for row in arm_rows),
                "activation_path_rows": sum(row["target"]["activation_path"] for row in arm_rows),
                "target_top5_rows": sum(row["target"]["in_top_k"] for row in arm_rows),
            }
    return {
        "execution_complete": True,
        "controls_pass": not failures,
        "failed_controls": failures,
        "coverage_by_embedding": coverage,
        "coverage_pass": not failures
        and all(
            value["semantic_only_negative_control"] and value["shared_case_coverage"]
            for value in coverage.values()
        ),
        "summary_by_embedding": summaries,
    }


async def run_measurement(output: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    vector_path = (
        repo / "docs/validation/2026-09-20/source-association-role-vectors-v1/artifact.json"
    )
    protocol = repo / "docs/plans/2026-09-20-source-association-role-protocol-v1.md"
    raw = vector_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != VECTORS_SHA256:
        raise ValueError("frozen role vector artifact changed")
    artifact = json.loads(raw)
    manifest, manifest_sha, _ = load_cases()
    if (
        artifact["status"] != "complete"
        or artifact["manifest_sha256"] != manifest_sha
        or artifact["model_digest"] != DIGEST
    ):
        raise ValueError("role preparation is incomplete or misbound")
    roles = artifact["role_vectors"]
    validate_role_vectors(manifest, roles)
    sources = [
        MANIFEST_PATH,
        vector_path,
        protocol,
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in (
                "run_source_association.py",
                "source_association_measurement.py",
                "source_association_fixture.py",
                "source_association_view.py",
                "source_association_lifecycle.py",
                "source_association_diagnostic.py",
                "graph_measurement.py",
                "module_mechanism.py",
                "prepare_source_association_vectors.py",
                "embedding_prefix_development_v1.py",
                "source_association_role_preparation_v1.py",
            )
        ],
        *sorted((repo / "mnemos").rglob("*.py")),
    ]
    _, head, hashes = _committed_inputs(repo, sources)
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("measurement output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "source-association-role-measurement-v1",
        "measurement_status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "input_hashes": hashes,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "manifest_sha256": manifest_sha,
        "vector_artifact_sha256": VECTORS_SHA256,
        "model": artifact["model"],
        "model_digest": artifact["model_digest"],
        "embedding_role_contracts": PREFIXES,
        "options": manifest["options"],
        "expected_rows": 90,
        "independent_cases": 5,
        "extraction": "synthetic_fixture",
        "quality_acceptance": "not_measured",
        "lifecycle": "required_for_both_embedding_contracts",
        "measurement_inputs_match_head_before": True,
        "repository_status_before": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True
        ),
        "rows": [],
    }
    path = output / "measurement.json"
    write(path, report)
    try:
        for label, vectors in roles.items():
            for case in manifest["graph_cases"]:
                for rep in range(3):
                    destination = output / label / case["id"] / f"repetition-{rep:02d}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    rows = await measure_repetition(
                        case,
                        vectors["documents"],
                        manifest["options"],
                        destination,
                        rep,
                        query_vectors=vectors["queries"],
                    )
                    for row in rows:
                        row["embedding_contract"] = label
                        write(destination / (row["arm"] + ".json"), row)
                    report["rows"].extend(rows)
                    write(path, report)
                    print(
                        json.dumps(
                            {
                                "embedding": label,
                                "case": case["id"],
                                "repetition": rep,
                                "completed_rows": len(report["rows"]),
                            }
                        ),
                        flush=True,
                    )
        _, final_head, final_hashes = _committed_inputs(repo, sources)
        if (final_head, final_hashes) != (head, hashes):
            raise RuntimeError("measurement inputs changed during execution")
        report["measurement_inputs_match_head_after"] = True
        report.update(summarize(report["rows"], manifest))
        report["measurement_status"] = "complete"
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
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True
        )
        report["artifacts"] = {
            p.relative_to(output).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != path
        }
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run_measurement(args.output))
    return 0 if report["execution_complete"] and report["coverage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
