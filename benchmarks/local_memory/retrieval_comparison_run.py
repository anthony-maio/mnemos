"""Prepare and run the separate retrieval-strategy comparison.

Preparation is offline and writes a bound NOT RUN manifest.  Execution delegates
session/checkpoint handling to the ordinary held-out runner after injecting the
retrieval comparator adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactRecorder, environment_metadata, file_digest, json_digest
from .contracts import HeldoutDataset, load_dataset
from .dev_smoke import _code_provenance
from .heldout_run import run_heldout
from .ollama_adapter import OllamaThink, add_ollama_think, ollama_options_sha256, parse_ollama_think
from .retrieval_comparison import RetrievalComparatorAdapter, STRATEGIES
from .scoring import paired_bootstrap_interval, wilson_interval

PROTOCOL_VERSION = "retrieval-comparison-protocol-v1"
RUN_NAMESPACE = "t18-retrieval-comparison-v1"
MIN_SESSIONS = 200
MIN_REPETITIONS = 3
STRATEGY_CONFIG: dict[str, Any] = {
    "vector": {"semantic_floor": 0.2, "candidate_pool": "min(max(k*6,20),max_candidates)"},
    "lexical": {"ranking": "bm25", "semantic_floor": None, "embeddings": False},
    "hybrid": {"ranking": "current_union_cosine"},
    "full": {"profile": "production"},
}
QUALITY_KINDS = ("answerable", "no_answer", "state_pair_semantic", "state_pair_state")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _options(
    *,
    endpoint: str,
    model: str,
    embedding_model: str,
    repetitions: int,
    think: OllamaThink,
    protocol_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "endpoint": endpoint.rstrip("/"),
        "model": model,
        "embedding_model": embedding_model,
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout_seconds": 120.0,
        "top_k": 5,
        "max_context_tokens": 1200,
        "repetitions": repetitions,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "strategies": STRATEGY_CONFIG,
        "environment": environment_metadata(),
    }
    add_ollama_think(value, think)
    return value


def build_binding(
    *,
    dataset: HeldoutDataset,
    dataset_path: Path,
    protocol_path: Path,
    code_sha256: str,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    strategy_sha256 = json_digest(STRATEGY_CONFIG)
    return {
        "namespace": RUN_NAMESPACE,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "version": dataset.version,
            "sha256": dataset.digest,
            "bytes": dataset_path.stat().st_size,
        },
        "protocol": {
            "version": PROTOCOL_VERSION,
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path),
        },
        "strategies": STRATEGY_CONFIG,
        "strategy_config_sha256": strategy_sha256,
        "options": dict(options),
        "options_sha256": ollama_options_sha256(options),
        "code_sha256": code_sha256,
    }


def validate_existing_binding(
    path: Path, binding: Mapping[str, Any], output_dir: Path, work_dir: Path
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing comparator manifest: {path}")
    manifest = _object(path)
    if manifest.get("binding") != dict(binding):
        raise ValueError("retrieval comparator binding drift; refusing resume")
    if manifest.get("output_dir") != str(output_dir.resolve()) or manifest.get("work_dir") != str(
        work_dir.resolve()
    ):
        raise ValueError("retrieval comparator directory binding drift; refusing resume")
    return manifest


def _require_fresh(output_dir: Path, work_dir: Path) -> None:
    _require_disjoint(output_dir, work_dir)
    if output_dir.exists() or work_dir.exists():
        raise FileExistsError("retrieval comparator output/work directories must be new")


def _require_disjoint(output_dir: Path, work_dir: Path) -> None:
    output = output_dir.resolve()
    work = work_dir.resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("output and work directories must be disjoint")


def _validate_dataset(dataset: HeldoutDataset, repetitions: int) -> None:
    if len(dataset.heldout_sessions) < MIN_SESSIONS:
        raise ValueError(f"comparator requires at least {MIN_SESSIONS} held-out sessions")
    if repetitions < MIN_REPETITIONS:
        raise ValueError(f"comparator requires at least {MIN_REPETITIONS} repetitions")


def _validate_completed_artifacts(
    manifest: Mapping[str, Any], binding: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    if manifest.get("status") != "completed":
        raise ValueError(
            "retrieval comparator is incomplete or failed; refusing overwrite on resume"
        )
    summary_path = output_dir / "retrieval-comparison-summary.json"
    full_path = output_dir / "full.json"
    raw_path = output_dir / "full.raw-outputs.jsonl"
    if not summary_path.is_file() or not full_path.is_file() or not raw_path.is_file():
        raise ValueError("retrieval comparator completed artifacts are incomplete")
    summary = _object(summary_path)
    if summary.get("binding") != dict(binding):
        raise ValueError("retrieval comparator summary binding drift; refusing resume")
    if summary.get("summary", {}).get("coverage_status") != "complete":
        raise ValueError("retrieval comparator coverage is incomplete; refusing completed resume")
    artifact_binding = summary.get("artifact_binding")
    if not isinstance(artifact_binding, Mapping):
        raise ValueError("retrieval comparator summary has no artifact binding")
    current_full = file_digest(full_path)
    current_raw = file_digest(raw_path)
    if (
        artifact_binding.get("full_report") != current_full
        or artifact_binding.get("full_raw_outputs") != current_raw
    ):
        raise ValueError("retrieval comparator report/raw artifact drift; refusing resume")
    if manifest.get("summary_sha256") != _sha256(summary_path):
        raise ValueError("retrieval comparator summary drift; refusing resume")
    if (
        manifest.get("full_report_sha256") != current_full["sha256"]
        or manifest.get("full_raw_outputs_sha256") != current_raw["sha256"]
    ):
        raise ValueError("retrieval comparator manifest artifact drift; refusing resume")
    full = _object(full_path)
    if (
        full.get("status") != "completed"
        or full.get("run_id") != f"{RUN_NAMESPACE}-{binding['dataset']['version']}-full"
    ):
        raise ValueError("retrieval comparator full report is not a completed bound run")
    raw_rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if full.get("raw_outputs") != raw_rows:
        raise ValueError("retrieval comparator embedded/raw output drift; refusing resume")
    return summary


def _not_run(
    dataset: HeldoutDataset,
    dataset_path: Path,
    options: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "local-memory-retrieval-comparator-v1",
        "status": "not_run",
        "reason": "Prepare-only artifact; no model, provider, or worker call was made.",
        "run_namespace": RUN_NAMESPACE,
        "strategies": list(STRATEGIES),
        "dataset": {
            "path": str(dataset_path.resolve()),
            "version": dataset.version,
            "sha256": dataset.digest,
            "bytes": dataset_path.stat().st_size,
        },
        "protocol": binding["protocol"],
        "binding": dict(binding),
        "config": dict(options),
        "gates": {
            "real_model": "NOT RUN",
            "retrieval_comparison": "NOT RUN",
            "human_acceptance": "NOT RUN",
        },
        "raw_outputs": [],
    }


def expected_comparison_keys(
    dataset: HeldoutDataset, repetitions: int
) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    for session in dataset.heldout_sessions:
        for repetition in range(1, repetitions + 1):
            keys.update(
                (session.session_id, repetition, query.query_id, query.kind)
                for query in session.queries
            )
            keys.update(
                (session.session_id, repetition, pair.pair_id + ":semantic", "state_pair_semantic")
                for pair in session.state_pairs
            )
            keys.update(
                (session.session_id, repetition, pair.pair_id + ":state", "state_pair_state")
                for pair in session.state_pairs
            )
    return keys


def _rows(reports: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in reports.get("raw_outputs", []) if isinstance(reports, Mapping) else []:
        if isinstance(row, Mapping):
            raw = row.get("raw_output")
            if isinstance(raw, Mapping) and isinstance(raw.get("retrieval_comparisons"), list):
                for comparison in raw["retrieval_comparisons"]:
                    if isinstance(comparison, Mapping):
                        raw_repetition = row.get("repetition")
                        if isinstance(raw_repetition, bool) or not isinstance(
                            raw_repetition, (int, str)
                        ):
                            raise ValueError("retrieval comparison has an invalid repetition")
                        try:
                            repetition = int(raw_repetition)
                        except ValueError as exc:
                            raise ValueError(
                                "retrieval comparison has an invalid repetition"
                            ) from exc
                        result.append(
                            {
                                **comparison,
                                "session_id": str(row.get("session_id", "")),
                                "repetition": repetition,
                                "execution_status": row.get("execution_status"),
                                "verified_model_provenance": row.get(
                                    "verified_model_provenance", False
                                ),
                                "verified_embedding_provenance": row.get(
                                    "verified_embedding_provenance", False
                                ),
                            }
                        )
    return result


def summarize_strategy_reports(
    report: Mapping[str, Any], *, dataset: HeldoutDataset | None = None, repetitions: int = 3
) -> dict[str, Any]:
    """Summarize each quality kind, rejecting duplicate or unknown query rows."""
    comparisons = _rows(report)
    observed_keys = {
        (
            str(row.get("session_id", "")),
            int(row["repetition"]),
            str(row.get("query_id", "")),
            str(row.get("quality_kind", "")),
        )
        for row in comparisons
    }
    expected = (
        expected_comparison_keys(dataset, repetitions) if dataset is not None else observed_keys
    )
    seen: set[tuple[str, int, str, str]] = set()
    for row in comparisons:
        kind = str(row.get("quality_kind", ""))
        key = (
            str(row.get("session_id", "")),
            int(row["repetition"]),
            str(row.get("query_id", "")),
            kind,
        )
        if kind not in QUALITY_KINDS or key not in expected:
            raise ValueError(f"unknown retrieval comparison identity: {key}")
        if key in seen:
            raise ValueError(f"duplicate retrieval comparison identity: {key}")
        seen.add(key)
    summary: dict[str, Any] = {
        "quality_kinds": {},
        "comparison_rows": len(comparisons),
        "expected_rows": len(expected),
        "missing_rows": len(expected - seen),
        "provenance": {
            "source_label": "synthetic_heldout_fixture",
            "human_acceptance": "not_run",
            "quality_gate": "NOT RUN",
            "eligible_observations": 0,
            "unverified_observations": 0,
        },
    }
    for kind in QUALITY_KINDS:
        kind_expected = {key for key in expected if key[3] == kind}
        kind_rows = [row for row in comparisons if row.get("quality_kind") == kind]
        maps: dict[str, dict[tuple[str, int, str, str], bool]] = {name: {} for name in STRATEGIES}
        for row in kind_rows:
            key = (str(row["session_id"]), int(row["repetition"]), str(row["query_id"]), kind)
            strategies = row.get("strategies")
            provenance_ok = (
                row.get("execution_status") == "complete"
                and row.get("verified_model_provenance") is True
                and row.get("verified_embedding_provenance") is True
            )
            if not provenance_ok:
                continue
            summary["provenance"]["eligible_observations"] += 1
            if not isinstance(strategies, Mapping):
                continue
            for name in STRATEGIES:
                value = strategies.get(name)
                if (
                    isinstance(value, Mapping)
                    and value.get("status") == "complete"
                    and type(value.get("quality_hit")) is bool
                ):
                    maps[name][key] = bool(value["quality_hit"])
        kind_summary: dict[str, Any] = {
            "expected_count": len(kind_expected),
            "observed_rows": len(kind_rows),
            "missing_row_count": len(kind_expected - {key for key in seen if key[3] == kind}),
            "strategies": {},
            "paired_to_full": {},
        }
        for name in STRATEGIES:
            measured = list(maps[name].values())
            metric: dict[str, Any] = {
                "measured_count": len(measured),
                "missing_count": len(kind_expected) - len(measured),
                "status": "MEASURED" if measured else "NOT MEASURED",
            }
            if measured:
                metric["hit_interval"] = wilson_interval(sum(measured), len(measured)).as_dict()
            else:
                metric["hit_interval"] = None
            kind_summary["strategies"][name] = metric
        full = maps["full"]
        for name in STRATEGIES:
            if name == "full":
                continue
            paired_keys = sorted(set(full) & set(maps[name]))
            if not paired_keys:
                kind_summary["paired_to_full"][name] = {
                    "status": "NOT MEASURED",
                    "eligible_count": 0,
                    "missing_count": len(kind_expected),
                }
                continue
            interval = paired_bootstrap_interval(
                [float(full[key]) for key in paired_keys],
                [float(maps[name][key]) for key in paired_keys],
                cluster_ids=[key[0] for key in paired_keys],
            ).as_dict()
            interval.update(
                {
                    "status": "MEASURED",
                    "eligible_count": len(paired_keys),
                    "missing_count": len(kind_expected) - len(paired_keys),
                }
            )
            kind_summary["paired_to_full"][name] = interval
        summary["quality_kinds"][kind] = kind_summary
    summary["provenance"]["unverified_observations"] = len(comparisons) - int(
        summary["provenance"]["eligible_observations"]
    )
    summary["coverage_status"] = (
        "complete"
        if expected
        and summary["missing_rows"] == 0
        and summary["provenance"]["unverified_observations"] == 0
        and all(
            metric["missing_count"] == 0
            for kind_summary in summary["quality_kinds"].values()
            for metric in kind_summary["strategies"].values()
        )
        else "incomplete"
    )
    return summary


def run_comparator(
    *,
    dataset_path: Path,
    protocol_path: Path,
    output_dir: Path,
    work_dir: Path,
    endpoint: str,
    model: str,
    embedding_model: str = "nomic-embed-text:latest",
    repetitions: int = 3,
    think: OllamaThink = False,
    prepare_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    _validate_dataset(dataset, repetitions)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_sha256 = _sha256(protocol_path)
    code_sha256, code_files = _code_provenance(Path(__file__).resolve().parents[2])
    options = _options(
        endpoint=endpoint,
        model=model,
        embedding_model=embedding_model,
        repetitions=repetitions,
        think=think,
        protocol_sha256=protocol_sha256,
    )
    binding = build_binding(
        dataset=dataset,
        dataset_path=dataset_path,
        protocol_path=protocol_path,
        code_sha256=code_sha256,
        options=options,
    )
    manifest_path = work_dir / "manifest.json"
    if resume:
        if prepare_only:
            raise ValueError("prepare-only cannot resume")
        _require_disjoint(output_dir, work_dir)
        manifest = validate_existing_binding(manifest_path, binding, output_dir, work_dir)
    else:
        _require_fresh(output_dir, work_dir)
        output_dir.mkdir(parents=True)
        work_dir.mkdir(parents=True)
        manifest = {
            "schema_version": "local-memory-retrieval-comparator-run-v1",
            "status": "not_run" if prepare_only else "running",
            "run_namespace": RUN_NAMESPACE,
            "binding": binding,
            "code_files": code_files,
            "output_dir": str(output_dir.resolve()),
            "work_dir": str(work_dir.resolve()),
            "strategies": list(STRATEGIES),
        }
        ArtifactRecorder(manifest_path).write(manifest)
    if prepare_only:
        report = _not_run(dataset, dataset_path, options, binding)
        ArtifactRecorder(output_dir / "retrieval-comparison.json").write(report)
        manifest["status"] = "not_run"
        manifest["report"] = str(output_dir / "retrieval-comparison.json")
        ArtifactRecorder(manifest_path).write(manifest)
        return {"manifest": manifest, "report": report}
    if resume:
        return {
            "manifest": manifest,
            "summary": _validate_completed_artifacts(manifest, binding, output_dir),
        }
    try:
        reports = run_heldout(
            dataset_path=dataset_path,
            output_dir=output_dir,
            work_dir=work_dir,
            endpoint=endpoint,
            model=model,
            embedding_model=embedding_model,
            repetitions=repetitions,
            profile_names={"full"},
            root=Path(__file__).resolve().parents[2],
            think=think,
            adapter_type=RetrievalComparatorAdapter,
            run_namespace=RUN_NAMESPACE,
            experiment_binding={
                "protocol_sha256": protocol_sha256,
                "strategy_config_sha256": binding["strategy_config_sha256"],
            },
        )
        full = reports.get("full")
        if not isinstance(full, Mapping):
            raise ValueError("held-out runner did not return the full retrieval report")
        summary = summarize_strategy_reports(full, dataset=dataset, repetitions=repetitions)
        summary_payload = {
            "schema_version": "local-memory-retrieval-comparison-summary-v1",
            "binding": binding,
            "summary": summary,
            "raw_timing_note": "Latency is descriptive only; no latency confidence interval or gate is claimed.",
        }
        full_report = output_dir / "full.json"
        full_raw = output_dir / "full.raw-outputs.jsonl"
        summary_payload["artifact_binding"] = {
            "full_report": file_digest(full_report),
            "full_raw_outputs": file_digest(full_raw),
        }
        ArtifactRecorder(output_dir / "retrieval-comparison-summary.json").write(summary_payload)
        manifest["summary_sha256"] = _sha256(output_dir / "retrieval-comparison-summary.json")
        manifest["full_report_sha256"] = _sha256(full_report)
        manifest["full_raw_outputs_sha256"] = _sha256(full_raw)
        manifest["status"] = (
            "completed"
            if full.get("status") == "completed" and summary["coverage_status"] == "complete"
            else "incomplete"
        )
        manifest["summary"] = str(output_dir / "retrieval-comparison-summary.json")
        ArtifactRecorder(manifest_path).write(manifest)
        return {"manifest": manifest, "reports": reports, "summary": summary_payload}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        ArtifactRecorder(manifest_path).write(manifest)
        raise


run_retrieval_comparison = run_comparator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("datasets") / "local-memory-heldout-v3.jsonl",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/plans/2026-09-18-retrieval-comparison-protocol-v1.md"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.86.35:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--think", type=parse_ollama_think, default=False)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_comparator(
        dataset_path=args.dataset,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        repetitions=args.repetitions,
        think=args.think,
        prepare_only=args.prepare_only,
        resume=args.resume,
    )
    print(json.dumps({"status": result["manifest"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
