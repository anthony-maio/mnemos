"""Prepare and run the separate applied encoding T18 comparator.

The comparator is deliberately separate from the normal seven-profile
lifecycle run.  Its three arms are full, ablated surprisal, and the applied
length/noise replacement.  Preparation never contacts Ollama.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactRecorder, environment_metadata, json_digest
from .contracts import HeldoutDataset, load_dataset
from .dev_smoke import _code_provenance
from .encoding_comparator import (
    COMPARATOR_PROFILES,
    POLICY_CONFIG,
    make_comparator_service_factory,
)
from .harness import (
    ExperimentConfig,
    ExperimentHarness,
    compare_observation_metrics,
)
from .heldout_run import _live_model_metadata, _registry_for_dataset
from .ollama_adapter import (
    OllamaThink,
    add_ollama_think,
    ollama_options_sha256,
    parse_ollama_think,
)

PROTOCOL_VERSION = "encoding-comparison-protocol-v1"
RUN_NAMESPACE = "t18-encoding-comparator-v1"
MIN_SESSIONS = 200
MIN_REPETITIONS = 3


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _policy_config() -> dict[str, Any]:
    return dict(POLICY_CONFIG)


def build_binding(
    *,
    dataset: HeldoutDataset,
    dataset_path: Path,
    protocol_path: Path,
    code_sha256: str,
    options: Mapping[str, Any],
    model_metadata: Mapping[str, Any] | None = None,
    embedding_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable identity used by manifests and checkpoints."""
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
            "sha256": _path_sha256(protocol_path),
        },
        "policy": _policy_config(),
        "policy_sha256": json_digest(_policy_config()),
        "options": dict(options),
        "options_sha256": ollama_options_sha256(options),
        "code_sha256": code_sha256,
        "model": dict(model_metadata) if model_metadata is not None else None,
        "embedding_model": (dict(embedding_metadata) if embedding_metadata is not None else None),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_existing_binding(path: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an existing manifest before reading or writing checkpoints."""
    if not path.exists():
        raise ValueError(f"missing comparator manifest: {path}")
    manifest = _read_json_object(path)
    if manifest.get("binding") != dict(binding):
        raise ValueError("comparator binding drift; refusing resume")
    return manifest


def _require_fresh_paths(output_dir: Path, work_dir: Path) -> None:
    output = output_dir.resolve()
    work = work_dir.resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("output and work directories must be disjoint")
    if output_dir.exists() or work_dir.exists():
        raise FileExistsError("comparator output/work directories must be new")


def _validated_report(
    output: Path, binding: Mapping[str, Any], config: ExperimentConfig, dataset: HeldoutDataset
) -> dict[str, Any]:
    report = _read_json_object(output)
    recorded_manifest = report.get("run_manifest", {})
    if (
        report.get("run_id") != f"{RUN_NAMESPACE}-{config.profile.name}"
        or report.get("status") != "completed"
        or not isinstance(recorded_manifest, dict)
        or recorded_manifest.get("binding") != binding
        or report.get("config") != config.as_dict()
        or report.get("profile") != config.profile.as_dict()
        or report.get("dataset", {}).get("sha256") != dataset.digest
    ):
        raise ValueError("comparator report binding drift or incomplete report; refusing resume")
    raw_file = report.get("raw_outputs_file", {})
    expected_raw = output.with_name(output.stem + ".raw-outputs.jsonl")
    if (
        not isinstance(raw_file, dict)
        or not expected_raw.is_file()
        or raw_file.get("sha256") != _path_sha256(expected_raw)
        or raw_file.get("bytes") != expected_raw.stat().st_size
    ):
        raise ValueError("comparator report raw-output binding drift; refusing resume")
    rows = [json.loads(line) for line in expected_raw.read_text(encoding="utf-8").splitlines()]
    if report.get("raw_outputs") != rows:
        raise ValueError("comparator report embedded raw-output binding drift; refusing resume")
    return report


def _next_attempt_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in parent.glob("attempt-*"):
        try:
            numbers.append(int(child.name.removeprefix("attempt-")))
        except ValueError:
            continue
    return parent / f"attempt-{max(numbers, default=-1) + 1:03d}"


def _validate_dataset(dataset: HeldoutDataset, repetitions: int) -> None:
    if len(dataset.heldout_sessions) < MIN_SESSIONS:
        raise ValueError(f"comparator requires at least {MIN_SESSIONS} held-out sessions")
    if repetitions < MIN_REPETITIONS:
        raise ValueError(f"comparator requires at least {MIN_REPETITIONS} repetitions")


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
        "policy": _policy_config(),
        "environment": environment_metadata(),
    }
    add_ollama_think(value, think)
    return value


def _not_run_artifact(
    dataset: HeldoutDataset,
    dataset_path: Path,
    profile: Any,
    options: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "local-memory-encoding-comparator-v1",
        "status": "not_run",
        "reason": "Prepare-only artifact; no model, provider, or worker call was made.",
        "run_namespace": RUN_NAMESPACE,
        "profile": profile.as_dict(),
        "provider": f"ollama:{options['endpoint']}",
        "repetitions_requested": int(options["repetitions"]),
        "dataset": {
            "path": str(dataset_path.resolve()),
            "version": dataset.version,
            "sha256": dataset.digest,
            "bytes": dataset_path.stat().st_size,
            "source_label": "synthetic_seed",
            "advertised_as_real": False,
        },
        "protocol": binding["protocol"],
        "policy": binding["policy"],
        "binding": dict(binding),
        "config": dict(options),
        "config_sha256": json_digest(options),
        "gates": {
            "real_model": "NOT RUN",
            "length_filter_decision": "NOT RUN",
            "human_acceptance": "NOT RUN",
        },
        "measurements": {},
        "raw_outputs": [],
    }


def _comparison_config(
    *,
    profile: Any,
    options: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    embedding_metadata: Mapping[str, Any],
    code_sha256: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        profile=profile,
        provider=f"ollama:{options['endpoint']}",
        repetitions=int(options["repetitions"]),
        top_k=int(options["top_k"]),
        max_context_tokens=int(options["max_context_tokens"]),
        model_sha256=str(model_metadata["digest"]),
        embedding_sha256=str(embedding_metadata["digest"]),
        options_sha256=ollama_options_sha256(options),
        code_sha256=code_sha256,
    )


def _attach_comparisons(reports: dict[str, dict[str, Any]], *, config: ExperimentConfig) -> None:
    full = reports.get("full")
    if full is None:
        return
    paired = {
        name: compare_observation_metrics(full, report, config=config)
        for name, report in reports.items()
        if name != "full"
    }
    filter_pairs = {
        name: compare_filter_decisions(full, report)
        for name, report in reports.items()
        if name != "full"
    }
    full["comparator"] = {
        "primary_question": "full surprisal gate versus applied length/noise replacement",
        "ungated_control": "ablate-surprisal_gating",
        "policy": _policy_config(),
        "paired_comparisons": paired,
        "filter_decision_comparisons": filter_pairs,
    }


def _filter_rows(artifact: Mapping[str, Any]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    rows: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for observation in artifact.get("raw_outputs", []):
        if not isinstance(observation, Mapping):
            continue
        raw = observation.get("raw_output")
        decisions = raw.get("encoding_policy_decisions") if isinstance(raw, Mapping) else None
        if not isinstance(decisions, list):
            continue
        session_id = str(observation.get("session_id", ""))
        try:
            repetition = int(observation.get("repetition", 0))
        except (TypeError, ValueError):
            continue
        for decision in decisions:
            if isinstance(decision, Mapping) and decision.get("event_id"):
                rows[(session_id, repetition, str(decision["event_id"]))] = decision
    return rows


def compare_filter_decisions(
    baseline_artifact: Mapping[str, Any], treatment_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare durable gate decisions without treating missing work as false."""
    baseline = _filter_rows(baseline_artifact)
    treatment = _filter_rows(treatment_artifact)
    keys = sorted(set(baseline) & set(treatment))
    counts: Counter[str] = Counter()
    comparable = 0
    for key in keys:
        before = baseline[key]
        after = treatment[key]
        before_policy = str(before.get("policy", "unobserved"))
        after_policy = str(after.get("policy", "unobserved"))
        before_kept = before.get("kept") if type(before.get("kept")) is bool else None
        after_kept = after.get("kept") if type(after.get("kept")) is bool else None
        if before_kept is None or after_kept is None:
            counts["unobserved"] += 1
            continue
        comparable += 1
        counts[
            f"{before_policy}_kept_{str(before_kept).lower()}__"
            f"{after_policy}_kept_{str(after_kept).lower()}"
        ] += 1
    return {
        "paired_event_count": len(keys),
        "comparable_event_count": comparable,
        "unobserved_event_count": len(keys) - comparable,
        "decision_pairs": dict(sorted(counts.items())),
        "direction": "baseline_to_treatment; no quality inference",
    }


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
    protocol_sha256 = _path_sha256(protocol_path)
    code_sha256, code_files = _code_provenance(Path(__file__).resolve().parents[2])
    options = _options(
        endpoint=endpoint,
        model=model,
        embedding_model=embedding_model,
        repetitions=repetitions,
        think=think,
        protocol_sha256=protocol_sha256,
    )
    model_metadata: dict[str, Any] | None = None
    embedding_metadata: dict[str, Any] | None = None
    if not prepare_only:
        records = _live_model_metadata(endpoint, (model, embedding_model))
        model_metadata = records[model]
        embedding_metadata = records[embedding_model]
    binding = build_binding(
        dataset=dataset,
        dataset_path=dataset_path,
        protocol_path=protocol_path,
        code_sha256=code_sha256,
        options=options,
        model_metadata=model_metadata,
        embedding_metadata=embedding_metadata,
    )
    manifest_path = work_dir / "manifest.json"
    if resume:
        if prepare_only:
            raise ValueError("prepare-only cannot resume an execution")
        validate_existing_binding(manifest_path, binding)
    else:
        _require_fresh_paths(output_dir, work_dir)
        output_dir.mkdir(parents=True)
        work_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "local-memory-encoding-comparator-run-v1",
        "status": "not_run" if prepare_only else "running",
        "run_namespace": RUN_NAMESPACE,
        "binding": binding,
        "code_files": code_files,
        "output_dir": str(output_dir.resolve()),
        "work_dir": str(work_dir.resolve()),
        "arms": [profile.as_dict() for profile in COMPARATOR_PROFILES],
    }
    if resume:
        existing = validate_existing_binding(manifest_path, binding)
        if existing.get("output_dir") != str(output_dir.resolve()) or existing.get(
            "work_dir"
        ) != str(work_dir.resolve()):
            raise ValueError("comparator manifest directory binding does not match")
        manifest = {**existing, "status": "running"}
    else:
        ArtifactRecorder(manifest_path).write(manifest)
    if prepare_only:
        reports: dict[str, Any] = {}
        for profile in COMPARATOR_PROFILES:
            report_path = output_dir / f"{profile.name}.json"
            artifact = _not_run_artifact(dataset, dataset_path, profile, options, binding)
            ArtifactRecorder(report_path).write(artifact)
            reports[profile.name] = artifact
        manifest["status"] = "not_run"
        manifest["reports"] = {name: str(output_dir / f"{name}.json") for name in reports}
        ArtifactRecorder(manifest_path).write(manifest)
        return {"manifest": manifest, "reports": reports}
    assert model_metadata is not None and embedding_metadata is not None
    registry = _registry_for_dataset(dataset, Path.cwd().resolve())
    reports = {}
    experiment_config = _comparison_config(
        profile=COMPARATOR_PROFILES[0],
        options=options,
        model_metadata=model_metadata,
        embedding_metadata=embedding_metadata,
        code_sha256=code_sha256,
    )
    configs = {
        profile.name: _comparison_config(
            profile=profile,
            options=options,
            model_metadata=model_metadata,
            embedding_metadata=embedding_metadata,
            code_sha256=code_sha256,
        )
        for profile in COMPARATOR_PROFILES
    }
    # Validate every existing report before an unfinished arm can write work.
    cached = {
        profile.name: _validated_report(
            output_dir / f"{profile.name}.json", binding, configs[profile.name], dataset
        )
        for profile in COMPARATOR_PROFILES
        if resume and (output_dir / f"{profile.name}.json").exists()
    }
    for profile in COMPARATOR_PROFILES:
        run_id = f"{RUN_NAMESPACE}-{profile.name}"
        checkpoint = work_dir / f"{profile.name}.checkpoint.json"
        output = output_dir / f"{profile.name}.json"
        config = configs[profile.name]
        if profile.name in cached:
            reports[profile.name] = cached[profile.name]
            continue
        if output.exists():
            raise FileExistsError(f"refusing to overwrite comparator report {output}")
        attempt_root = _next_attempt_dir(work_dir / "attempts" / profile.name)
        attempt_root.mkdir(parents=True, exist_ok=False)
        try:
            factory = make_comparator_service_factory(
                attempt_root / f"{profile.name}.db",
                registry,
                base_url=endpoint,
                model=model,
                embedding_model=embedding_model,
                timeout=float(options["timeout_seconds"]),
                temperature=float(options["temperature"]),
                max_tokens=int(options["max_tokens"]),
                think=think,
            )
            from .encoding_comparator import EncodingComparatorAdapter

            adapter = EncodingComparatorAdapter(
                factory,
                base_url=endpoint,
                model=model,
                embedding_model=embedding_model,
                real_model=True,
                model_revision=str(model_metadata["digest"]),
                embedding_revision=str(embedding_metadata["digest"]),
                model_sha256=str(model_metadata["digest"]),
                embedding_sha256=str(embedding_metadata["digest"]),
                options_sha256=ollama_options_sha256(options),
                code_sha256=code_sha256,
                model_provenance=model_metadata,
                embedding_provenance=embedding_metadata,
            )
            artifact = ExperimentHarness(dataset, config).run(
                adapter, checkpoint_path=checkpoint, run_id=run_id
            )
            if profile.name == "length-noise":
                expected_keys = {
                    (session.session_id, repetition, event.event_id)
                    for session in dataset.heldout_sessions
                    for repetition in range(1, repetitions + 1)
                    for event in session.events
                    if event.lifecycle_operation in {"capture", "noise", "response"}
                }
                decision_rows = _filter_rows(artifact)
                observed_keys = {
                    key
                    for key, row in decision_rows.items()
                    if row.get("job_status") == "completed"
                    and row.get("policy") == "length_noise"
                    and type(row.get("kept")) is bool
                }
                artifact["applied_length_coverage"] = {
                    "expected": len(expected_keys),
                    "observed": len(observed_keys & expected_keys),
                    "missing": len(expected_keys - observed_keys),
                    "unexpected": len(observed_keys - expected_keys),
                }
                artifact.setdefault("gates", {})["length_filter_decision"] = (
                    "OBSERVED"
                    if expected_keys and observed_keys == expected_keys
                    else "NOT VERIFIED"
                )
            artifact["run_manifest"] = manifest
            reports[profile.name] = artifact
            ArtifactRecorder(output).write_run(
                artifact,
                [row for row in artifact.get("raw_outputs", []) if isinstance(row, dict)],
            )
            manifest.setdefault("attempts", []).append(
                {
                    "profile": profile.name,
                    "path": str(attempt_root),
                    "status": artifact.get("status", "incomplete"),
                }
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest.setdefault("attempts", []).append(
                {
                    "profile": profile.name,
                    "path": str(attempt_root),
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            ArtifactRecorder(manifest_path).write(manifest)
            raise
    _attach_comparisons(reports, config=experiment_config)
    for name, artifact in reports.items():
        if name == "full" and artifact.get("comparator"):
            ArtifactRecorder(output_dir / f"{name}.json").write_run(
                artifact,
                [row for row in artifact.get("raw_outputs", []) if isinstance(row, dict)],
            )
    manifest["status"] = (
        "completed"
        if all(report.get("status") == "completed" for report in reports.values())
        else "incomplete"
    )
    manifest["reports"] = {name: str(output_dir / f"{name}.json") for name in reports}
    ArtifactRecorder(manifest_path).write(manifest)
    return {"manifest": manifest, "reports": reports}


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
        default=Path("docs/plans/2026-09-18-encoding-comparison-protocol-v1.md"),
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
