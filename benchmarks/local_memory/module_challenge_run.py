"""Resumable runner for the supplemental competing-candidate fixture.

This run is separate from the 200-session lifecycle matrix. It exists to
measure controlled affective scoring and graph candidate observations against
the 50 synthetic challenge sessions; it does not turn declared fixture IDs
into observed retrieval evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from typing import Any, Literal, cast

from mnemos.scopes import ProjectIdentity, ScopeContext, ScopeRegistry

from .artifacts import ArtifactRecorder
from .contracts import HeldoutDataset
from .dev_smoke import _code_provenance
from .harness import (
    DEFAULT_PROFILES,
    ExperimentConfig,
    ExperimentHarness,
    compare_observation_metrics,
)
from .module_challenge import load_module_challenge_dataset
from .ollama_adapter import OllamaEvaluatorAdapter, make_ollama_service_factory
from .heldout_run import _live_model_metadata

CHALLENGE_PROFILES = (
    "baseline",
    "full",
    "ablate-affective_routing",
    "ablate-spreading_activation",
)


def _select_profiles(profile_names: set[str] | None) -> tuple[Any, ...]:
    requested = set(profile_names or ())
    unknown = requested - set(CHALLENGE_PROFILES)
    if unknown:
        raise ValueError("unknown challenge profiles: " + ", ".join(sorted(unknown)))
    selected = tuple(
        profile
        for profile in DEFAULT_PROFILES
        if profile.name in (set(CHALLENGE_PROFILES) if profile_names is None else requested)
    )
    if not selected:
        raise ValueError("no recognized challenge profiles were selected")
    return selected


def _challenge_registry(dataset: HeldoutDataset, root: Path) -> ScopeRegistry:
    default_profile = "t18-synthetic-profile"
    project_profiles: dict[str, str] = {}
    for session in dataset.heldout_sessions:
        session_profile = next(
            (
                event.target_user_profile_id
                for event in session.events
                if event.target_user_profile_id
            ),
            default_profile,
        )
        existing = project_profiles.setdefault(session.project_id, session_profile)
        if existing != session_profile:
            raise ValueError(f"challenge project {session.project_id} has mixed profiles")
    for session in dataset.heldout_sessions:
        for target in (*session.events, *session.queries, *session.state_pairs):
            target_project = getattr(target, "target_project_id", None) or session.project_id
            target_profile = getattr(target, "target_user_profile_id", None) or default_profile
            existing = project_profiles.setdefault(target_project, target_profile)
            if existing != target_profile:
                raise ValueError(f"challenge target {target_project} has mixed profiles")
    projects = {
        project_id: ProjectIdentity(
            id=project_id,
            root=str(root),
            user_profile_id=profile,
            enabled=True,
            workspace_ids=[],
        )
        for project_id, profile in sorted(project_profiles.items())
    }
    return ScopeRegistry(
        projects=projects,
        workspaces={},
        profiles=sorted(set(project_profiles.values())),
    )


def _challenge_graph_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for raw in artifact.get("raw_outputs", []):
        if not isinstance(raw, dict):
            continue
        output = raw.get("raw_output")
        if not isinstance(output, dict):
            continue
        controlled = output.get("controlled_state_pairs")
        if isinstance(controlled, list):
            rows.extend(item for item in controlled if isinstance(item, dict))
        operations = output.get("operations")
        if isinstance(operations, list):
            retrieval_rows.extend(
                item
                for item in operations
                if isinstance(item, dict)
                and item.get("operation") == "retrieve"
                and isinstance(item.get("facts"), list)
            )
    statuses = [
        str(row.get("graph_target_observation", {}).get("status", "unknown"))
        for row in rows
        if isinstance(row.get("graph_target_observation"), dict)
    ]
    return {
        "rows": rows,
        "row_count": len(rows),
        "full_pipeline_retrieval_rows": retrieval_rows,
        "full_pipeline_retrieval_row_count": len(retrieval_rows),
        "not_discriminating_count": statuses.count(
            "not_discriminating_target_already_in_direct_seed"
        ),
        "unobserved_graph_traversal_count": statuses.count("not_observed_graph_traversal"),
        "declared_ids_are_not_observations": True,
    }


def _validate_challenge_contexts(dataset: HeldoutDataset, registry: ScopeRegistry) -> None:
    """Validate every source and query target before model execution."""
    default_profile = "t18-synthetic-profile"
    for session in dataset.heldout_sessions:
        for target in (*session.events, *session.queries, *session.state_pairs):
            profile = getattr(target, "target_user_profile_id", None) or next(
                (
                    event.target_user_profile_id
                    for event in session.events
                    if event.target_user_profile_id
                ),
                default_profile,
            )
            context = ScopeContext(
                user_profile_id=profile,
                project_id=getattr(target, "target_project_id", None) or session.project_id,
                workspace_id=getattr(target, "target_workspace_id", None),
                session_id=session.session_id,
                host=cast(Literal["codex", "claude-code", "hermes", "manual"], session.host),
            )
            registry.validate_context(context)


def run_module_challenge(
    *,
    dataset_path: Path,
    output_dir: Path,
    work_dir: Path,
    endpoint: str,
    model: str,
    embedding_model: str,
    repetitions: int = 3,
    profile_names: set[str] | None = None,
    root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    dataset = load_module_challenge_dataset(dataset_path)
    if repetitions < 3:
        raise ValueError("module challenge requires at least three repetitions")
    selected = _select_profiles(profile_names)
    model_records = _live_model_metadata(endpoint, (model, embedding_model))
    model_metadata = model_records[model]
    embedding_metadata = model_records[embedding_model]
    options: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout_seconds": 120.0,
        "top_k": 5,
        "max_context_tokens": 1200,
    }
    options_sha256 = hashlib.sha256(
        json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    code_sha256, code_files = _code_provenance(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    registry = _challenge_registry(dataset, project_root)
    _validate_challenge_contexts(dataset, registry)
    reports: dict[str, dict[str, Any]] = {}
    for profile in selected:
        config = ExperimentConfig(
            profile=profile,
            provider=f"ollama:{endpoint.rstrip('/')}",
            repetitions=repetitions,
            top_k=int(options["top_k"]),
            max_context_tokens=int(options["max_context_tokens"]),
            model_sha256=str(model_metadata["digest"]),
            embedding_sha256=str(embedding_metadata["digest"]),
            options_sha256=options_sha256,
            code_sha256=code_sha256,
        )
        run_id = f"t18-module-challenge-{dataset.version}-{profile.name}"
        checkpoint = work_dir / f"{profile.name}.checkpoint.json"
        output = output_dir / f"{profile.name}.json"
        attempt_id = uuid4().hex
        attempt_root = work_dir / "attempts" / profile.name / attempt_id
        attempt_root.mkdir(parents=True, exist_ok=True)
        attempt_db = attempt_root / f"{profile.name}.db"
        manifest: dict[str, Any] = {
            "schema_version": "local-memory-module-challenge-run-v1",
            "status": "running",
            "run_id": run_id,
            "dataset_sha256": dataset.digest,
            "dataset_version": dataset.version,
            "profile": profile.as_dict(),
            "repetitions": repetitions,
            "provider": {
                "endpoint": endpoint,
                "model": model_metadata,
                "embedding_model": embedding_metadata,
                "options": options,
            },
            "options_sha256": options_sha256,
            "code_sha256": code_sha256,
            "code_files": code_files,
            "checkpoint": str(checkpoint),
            "attempt_id": attempt_id,
            "attempt_db_root": str(attempt_root),
            "declared_candidate_ids_are_not_observations": True,
        }
        manifest_path = work_dir / f"{profile.name}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        factory = make_ollama_service_factory(
            attempt_db,
            registry,
            base_url=endpoint,
            model=model,
            embedding_model=embedding_model,
            timeout=float(options["timeout_seconds"]),
            temperature=float(options["temperature"]),
            max_tokens=int(options["max_tokens"]),
        )
        adapter = OllamaEvaluatorAdapter(
            factory,
            base_url=endpoint,
            model=model,
            embedding_model=embedding_model,
            real_model=True,
            model_revision=str(model_metadata["digest"]),
            embedding_revision=str(embedding_metadata["digest"]),
            model_sha256=str(model_metadata["digest"]),
            embedding_sha256=str(embedding_metadata["digest"]),
            options_sha256=options_sha256,
            code_sha256=code_sha256,
            model_provenance=model_metadata,
            embedding_provenance=embedding_metadata,
        )
        try:
            artifact = ExperimentHarness(dataset, config).run(
                adapter, checkpoint_path=checkpoint, run_id=run_id
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise
        artifact["run_manifest"] = manifest
        artifact["module_challenge"] = _challenge_graph_summary(artifact)
        rows = [row for row in artifact.get("raw_outputs", []) if isinstance(row, dict)]
        ArtifactRecorder(output).write_run(artifact, rows)
        reports[profile.name] = artifact
        manifest["status"] = artifact.get("status", "incomplete")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    baseline = reports.get("baseline")
    if baseline is not None:
        comparison_config = ExperimentConfig(repetitions=repetitions)
        comparisons = {
            name: compare_observation_metrics(baseline, report, config=comparison_config)
            for name, report in reports.items()
            if name != "baseline"
        }
        baseline["paired_comparisons"] = comparisons
        baseline_path = output_dir / "baseline.json"
        if baseline_path.exists():
            stored = json.loads(baseline_path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                stored["paired_comparisons"] = comparisons
                ArtifactRecorder(baseline_path).write(stored)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("datasets") / "local-memory-module-challenge-v1.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.86.35:11434")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", action="append", dest="profiles")
    args = parser.parse_args(argv)
    reports = run_module_challenge(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        repetitions=args.repetitions,
        profile_names=set(args.profiles) if args.profiles else None,
    )
    print(json.dumps({name: report["status"] for name, report in reports.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
