"""Resumable real-model launcher for the frozen local-memory evaluation.

This command is intentionally separate from developmental smoke runs. It
verifies the live Ollama model digests before creating one checkpoint per
profile, and it never replaces incomplete observations with estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from uuid import uuid4
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.request import Request, urlopen

from mnemos.settings import OllamaProviderSettings
from mnemos.config import SleepConfig
from mnemos.scopes import ProjectIdentity, ScopeRegistry, WorkspaceIdentity
from mnemos.runtime import embedding_task_contract

from .artifacts import ArtifactRecorder, environment_metadata
from .contracts import HeldoutDataset, load_dataset
from .dev_smoke import _code_provenance
from .harness import (
    DEFAULT_PROFILES,
    ExperimentConfig,
    ExperimentHarness,
    compare_observation_metrics,
)
from .ollama_adapter import (
    OllamaEvaluatorAdapter,
    OllamaThink,
    add_ollama_think,
    make_ollama_service_factory,
    ollama_options_sha256,
    parse_ollama_think,
)
from .module_challenge import load_module_challenge_dataset

# Recorded in each manifest but kept out of the checkpoint identity. A cloud VM
# can come back on a new kernel release, and that must not orphan a checkpoint
# that is otherwise the same experiment. Anything that can change outputs,
# such as the GPU model, belongs in experiment_binding instead.
_HOST_DETAILS = ("platform", "processor")


def _live_typesafe_model(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    """Identity for a hosted judge, labeled for what it is.

    There is no weights digest for a hosted model. The record carries the
    version string the API reports and a digest of that string, so the
    provenance check has something exact to bind to and every manifest says
    which kind of identity it is.
    """
    body = {
        "model": model,
        "state": "ping",
        "questions": {
            "ready": {
                "type": "noul",
                "instructions": "Is the state exactly the word ping?",
                "criteria": {"true": "it is", "false": "it is not"},
            }
        },
    }
    request = Request(
        base_url.rstrip("/") + "/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30.0) as response:  # noqa: S310 - explicit CLI endpoint
        payload = json.loads(response.read().decode("utf-8"))
    observed = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(observed, str) or not observed:
        raise RuntimeError("TypeSafe did not report a model identity")
    return {
        "name": observed,
        "model": observed,
        "requested": model,
        "digest": hashlib.sha256(f"typesafe:{observed}".encode("utf-8")).hexdigest(),
        "identity_kind": "vendor_version_string",
        "note": "hosted model; digest is of the reported version string, not of weights",
    }


def _live_model_metadata(endpoint: str, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Read exact model records from the reachable Ollama server."""
    request = Request(endpoint.rstrip("/") + "/api/tags", headers={"Accept": "application/json"})
    with urlopen(request, timeout=30.0) as response:  # noqa: S310 - explicit CLI endpoint
        payload = json.loads(response.read().decode("utf-8"))
    records = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("Ollama /api/tags returned no model list")
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        match = next(
            (item for item in records if isinstance(item, dict) and item.get("name") == name),
            None,
        )
        digest = match.get("digest") if isinstance(match, dict) else None
        if not isinstance(match, dict) or not isinstance(digest, str):
            raise RuntimeError(f"Ollama model {name!r} is not present in live /api/tags")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise RuntimeError(f"Ollama model {name!r} has an invalid live digest")
        result[name] = dict(match)
    return result


def _registry_for_dataset(dataset: HeldoutDataset, root: Path) -> ScopeRegistry:
    profile_id = "t18-synthetic-profile"
    project_profiles: dict[str, str] = {
        session.project_id: profile_id for session in dataset.heldout_sessions
    }
    workspace_profiles: dict[str, str] = {}
    memberships: dict[str, set[str]] = {}

    def bind_profile(identities: dict[str, str], kind: str, identity: str, profile: str) -> None:
        previous = identities.get(identity)
        if previous is not None and previous != profile:
            raise ValueError(f"{kind} {identity!r} has conflicting profiles")
        identities[identity] = profile

    for session in dataset.heldout_sessions:
        for target in (*session.events, *session.queries, *session.state_pairs):
            target_project = getattr(target, "target_project_id", None) or session.project_id
            target_profile = getattr(target, "target_user_profile_id", None) or profile_id
            target_workspace = getattr(target, "target_workspace_id", None)
            bind_profile(project_profiles, "project", target_project, target_profile)
            if target_workspace:
                bind_profile(workspace_profiles, "workspace", target_workspace, target_profile)
                memberships.setdefault(target_project, set()).add(target_workspace)
    projects = {
        project_id: ProjectIdentity(
            id=project_id,
            root=str(root),
            user_profile_id=project_profiles[project_id],
            enabled=True,
            workspace_ids=sorted(memberships.get(project_id, set())),
        )
        for project_id in sorted(project_profiles)
    }
    workspaces = {
        workspace_id: WorkspaceIdentity(
            id=workspace_id,
            name=workspace_id,
            user_profile_id=workspace_profiles[workspace_id],
        )
        for workspace_id in sorted(workspace_profiles)
    }
    profiles = sorted(set(project_profiles.values()) | set(workspace_profiles.values()))
    return ScopeRegistry(projects=projects, workspaces=workspaces, profiles=profiles)


def run_heldout(
    *,
    dataset_path: Path,
    output_dir: Path,
    work_dir: Path,
    endpoint: str,
    model: str,
    embedding_model: str,
    repetitions: int = 3,
    profile_names: set[str] | None = None,
    module_challenge_path: Path | None = None,
    root: Path | None = None,
    think: OllamaThink = None,
    extraction_mode: Literal["generated", "source_bound", "judged"] = "generated",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_seconds: float = 120.0,
    query_prefix: str = "",
    document_prefix: str = "",
    symmetric_prefix: str = "",
    adapter_type: type[OllamaEvaluatorAdapter] = OllamaEvaluatorAdapter,
    run_namespace: str = "t18-heldout",
    experiment_binding: Mapping[str, Any] | None = None,
    judge_provider: Literal["none", "typesafe", "ollama"] = "none",
    typesafe_api_key: str | None = None,
    typesafe_model: str = "jev-latest",
    typesafe_base_url: str = "https://api.typesafe.ai/v1",
    surprisal_mode: Literal["prediction", "novelty"] = "prediction",
) -> dict[str, dict[str, Any]]:
    selected_extraction = SleepConfig(extraction_mode=extraction_mode).extraction_mode
    if judge_provider == "typesafe" and not typesafe_api_key:
        raise ValueError("typesafe_api_key is required for judge_provider=typesafe")
    if judge_provider == "none" and (
        selected_extraction == "judged" or surprisal_mode == "novelty"
    ):
        raise ValueError("judged extraction and novelty surprisal need a judge_provider")
    generation = OllamaProviderSettings(
        temperature=temperature, max_tokens=max_tokens, timeout=timeout_seconds
    )
    dataset = load_dataset(dataset_path)
    if len(dataset.heldout_sessions) < 200:
        raise ValueError("held-out dataset does not meet the 200-session minimum")
    if repetitions < 3:
        raise ValueError("T18 requires at least three repetitions")
    registry = _registry_for_dataset(dataset, (root or Path.cwd()).resolve())
    known_profiles = {profile.name for profile in DEFAULT_PROFILES}
    requested_profiles = set(profile_names or ())
    unknown_profiles = requested_profiles - known_profiles
    if unknown_profiles:
        raise ValueError("unknown evaluation profiles: " + ", ".join(sorted(unknown_profiles)))
    selected = tuple(
        profile
        for profile in DEFAULT_PROFILES
        if profile_names is None or profile.name in requested_profiles
    )
    if not selected:
        raise ValueError("no recognized evaluation profiles were selected")
    challenge_path = module_challenge_path
    challenge_dataset = (
        load_module_challenge_dataset(challenge_path) if challenge_path is not None else None
    )
    model_records = _live_model_metadata(endpoint, (model, embedding_model))
    model_metadata = model_records[model]
    embedding_metadata = model_records[embedding_model]
    judge_metadata: dict[str, Any] | None = None
    if judge_provider == "typesafe":
        assert typesafe_api_key is not None
        judge_metadata = _live_typesafe_model(typesafe_base_url, typesafe_api_key, typesafe_model)
        # In a judged run the judge is the cognition being measured. Its
        # identity replaces the generative model's in the run binding.
        cognition_metadata = judge_metadata
        cognition_name = judge_metadata["name"]
    elif judge_provider == "ollama":
        judge_metadata = dict(model_metadata, identity_kind="ollama_digest", judge=True)
        cognition_metadata = model_metadata
        cognition_name = model
    else:
        cognition_metadata = model_metadata
        cognition_name = model
    runtime = environment_metadata()
    options: dict[str, Any] = {
        "extraction_mode": selected_extraction,
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
        "timeout_seconds": generation.timeout,
        "top_k": 5,
        "max_context_tokens": 1200,
        "runtime": {key: value for key, value in runtime.items() if key not in _HOST_DETAILS},
    }
    if judge_provider != "none":
        options["judge_provider"] = judge_provider
        options["surprisal_mode"] = surprisal_mode
        options["judge_model"] = typesafe_model if judge_provider == "typesafe" else model
        if judge_provider == "typesafe":
            options["judge_endpoint"] = typesafe_base_url.rstrip("/")
            options["typesafe_price_usd_per_million_input_tokens"] = 0.042
    add_ollama_think(options, think)
    task_contract = embedding_task_contract(
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        symmetric_prefix=symmetric_prefix,
    )
    if task_contract is not None:
        options["embedding_task_contract"] = task_contract
    if experiment_binding is not None:
        options["experiment"] = dict(experiment_binding)
    options_sha256 = ollama_options_sha256(options)
    code_sha256, code_files = _code_provenance(
        (root or Path(__file__).resolve().parents[2]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for profile in selected:
        config = ExperimentConfig(
            profile=profile,
            provider=f"ollama:{endpoint.rstrip('/')}",
            repetitions=repetitions,
            top_k=int(options["top_k"]),
            max_context_tokens=int(options["max_context_tokens"]),
            model_sha256=str(cognition_metadata["digest"]),
            embedding_sha256=str(embedding_metadata["digest"]),
            options_sha256=options_sha256,
            code_sha256=code_sha256,
        )
        run_id = f"{run_namespace}-{dataset.version}-{profile.name}"
        checkpoint = work_dir / f"{profile.name}.checkpoint.json"
        output = output_dir / f"{profile.name}.json"
        # A resumed checkpoint may skip completed observations. Remaining
        # observations must start on fresh files so an interrupted SQLite
        # transaction or queued job is never replayed against a dirty DB.
        attempt_id = uuid4().hex
        attempt_root = work_dir / "attempts" / profile.name / attempt_id
        attempt_root.mkdir(parents=True, exist_ok=True)
        attempt_db = attempt_root / f"{profile.name}.db"
        manifest = {
            "schema_version": "local-memory-heldout-run-v1",
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
                "judge": judge_metadata,
                "options": options,
            },
            "options_sha256": options_sha256,
            "run_namespace": run_namespace,
            "experiment_binding": dict(experiment_binding or {}),
            "code_sha256": code_sha256,
            "code_files": code_files,
            "checkpoint": str(checkpoint),
            "host": {key: runtime[key] for key in _HOST_DETAILS if key in runtime},
            "attempt_id": attempt_id,
            "attempt_db_root": str(attempt_root),
            "module_challenge": (
                {
                    "path": str(challenge_path),
                    "version": challenge_dataset.version,
                    "digest": challenge_dataset.digest,
                    "heldout_sessions": len(challenge_dataset.heldout_sessions),
                    "state_pairs": sum(
                        len(session.state_pairs) for session in challenge_dataset.heldout_sessions
                    ),
                    "status": "validated_preflight_not_run",
                }
                if challenge_dataset is not None
                else {"status": "not_supplied"}
            ),
        }
        (work_dir / f"{profile.name}.manifest.json").write_text(
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
            think=think,
            extraction_mode=selected_extraction,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            symmetric_prefix=symmetric_prefix,
            judge_provider=judge_provider,
            typesafe_api_key=typesafe_api_key,
            typesafe_model=typesafe_model,
            typesafe_base_url=typesafe_base_url,
            surprisal_mode=surprisal_mode,
        )
        adapter = adapter_type(
            factory,
            base_url=endpoint,
            model=cognition_name,
            embedding_model=embedding_model,
            real_model=True,
            model_revision=str(cognition_metadata["digest"]),
            embedding_revision=str(embedding_metadata["digest"]),
            model_sha256=str(cognition_metadata["digest"]),
            embedding_sha256=str(embedding_metadata["digest"]),
            options_sha256=options_sha256,
            code_sha256=code_sha256,
            model_provenance=cognition_metadata,
            embedding_provenance=embedding_metadata,
        )
        try:
            artifact = ExperimentHarness(dataset, config).run(
                adapter, checkpoint_path=checkpoint, run_id=run_id
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            (work_dir / f"{profile.name}.manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise
        artifact["run_manifest"] = manifest
        rows = [row for row in artifact.get("raw_outputs", []) if isinstance(row, dict)]
        ArtifactRecorder(output).write_run(artifact, rows)
        reports[profile.name] = artifact
        manifest["status"] = artifact.get("status", "incomplete")
        (work_dir / f"{profile.name}.manifest.json").write_text(
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
        default=Path(__file__).with_name("datasets") / "local-memory-heldout-v3.jsonl",
    )
    parser.add_argument("--module-challenge-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.86.35:11434")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument(
        "--extraction-mode", choices=("generated", "source_bound", "judged"), default="generated"
    )
    parser.add_argument("--judge", choices=("none", "typesafe", "ollama"), default="none")
    parser.add_argument("--typesafe-model", default="jev-latest")
    parser.add_argument("--typesafe-url", default="https://api.typesafe.ai/v1")
    parser.add_argument("--surprisal-mode", choices=("prediction", "novelty"), default="prediction")
    parser.add_argument("--think", type=parse_ollama_think, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--symmetric-prefix", default="")
    args = parser.parse_args(argv)
    reports = run_heldout(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        repetitions=args.repetitions,
        profile_names=set(args.profiles) if args.profiles else None,
        module_challenge_path=args.module_challenge_dataset,
        think=args.think,
        extraction_mode=args.extraction_mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        symmetric_prefix=args.symmetric_prefix,
        judge_provider=args.judge,
        typesafe_api_key=os.environ.get("MNEMOS_TYPESAFE_API_KEY"),
        typesafe_model=args.typesafe_model,
        typesafe_base_url=args.typesafe_url,
        surprisal_mode=args.surprisal_mode,
    )
    print(json.dumps({name: report["status"] for name, report in reports.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
