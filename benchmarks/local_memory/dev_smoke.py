"""Run one bounded Ollama smoke session for every evaluation profile.

This launcher is deliberately separate from the held-out harness. It writes
one development artifact per profile and never promotes those observations to
held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemos.scopes import ProjectIdentity, ScopeRegistry

from .contracts import load_dataset
from .harness import DEFAULT_PROFILES
from .ollama_adapter import OllamaEvaluatorAdapter, make_ollama_service_factory


def _code_provenance(repo_root: Path) -> tuple[str, list[str]]:
    """Hash the runnable Python/config surface used by a development run."""
    repo_root = repo_root.resolve()
    candidates = [
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".py", ".toml"}
        and ".git" not in path.relative_to(repo_root).parts
        and ".venv" not in path.relative_to(repo_root).parts
        and "__pycache__" not in path.relative_to(repo_root).parts
    ]
    candidates.sort()
    if not candidates:
        raise ValueError(f"No runnable Python/config files found under {repo_root}")
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in candidates:
        relative = path.relative_to(repo_root).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def run_development_smoke(
    *,
    dataset_path: Path,
    output_dir: Path,
    session_id: str,
    discovery_path: Path,
    endpoint: str,
    model: str,
    embedding_model: str,
    profile_names: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    session = next((item for item in dataset.sessions if item.session_id == session_id), None)
    if session is None or session.split != "development":
        raise ValueError(f"session {session_id!r} must identify a development row")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    models = {item["name"]: item for item in discovery.get("tags", {}).get("models", [])}
    model_metadata = models.get(model, {})
    embedding_metadata = models.get(embedding_model, {})
    profile_id = "t18-synthetic-profile"
    registry = ScopeRegistry(
        projects={
            session.project_id: ProjectIdentity(
                id=session.project_id,
                root=str(Path.cwd()),
                user_profile_id=profile_id,
                enabled=True,
            )
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"temperature": 0.0, "max_tokens": 512, "timeout_seconds": 120.0}
    options_sha256 = hashlib.sha256(
        json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    code_sha256, code_files = _code_provenance(Path(__file__).resolve().parents[2])
    reports: dict[str, dict[str, Any]] = {}
    selected_profiles = (
        tuple(profile for profile in DEFAULT_PROFILES if profile.name in profile_names)
        if profile_names is not None
        else DEFAULT_PROFILES
    )
    for profile in selected_profiles:
        with tempfile.TemporaryDirectory(prefix=f"mnemos-t18-{profile.name}-") as temporary:
            factory = make_ollama_service_factory(
                Path(temporary) / "memory.db",
                registry,
                base_url=endpoint,
                model=model,
                embedding_model=embedding_model,
                timeout=options["timeout_seconds"],
                temperature=options["temperature"],
                max_tokens=int(options["max_tokens"]),
            )
            adapter = OllamaEvaluatorAdapter(
                factory,
                base_url=endpoint,
                model=model,
                embedding_model=embedding_model,
                real_model=True,
                model_revision=model_metadata.get("digest"),
                embedding_revision=embedding_metadata.get("digest"),
                model_sha256=model_metadata.get("digest", ""),
                embedding_sha256=embedding_metadata.get("digest", ""),
                options_sha256=options_sha256,
                code_sha256=code_sha256,
                unsupported_modules=(),
                model_provenance=model_metadata,
                embedding_provenance=embedding_metadata,
            )
            observation = adapter.evaluate_session(session, profile, 1)
        artifact = {
            "schema_version": "local-memory-development-smoke-v1",
            "status": observation.execution_status,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "version": dataset.version,
                "sha256": dataset.digest,
                "source_label": "synthetic_seed",
                "advertised_as_real": False,
                "session_id": session.session_id,
            },
            "profile": profile.as_dict(),
            "provider": {
                "endpoint": endpoint,
                "model": model,
                "embedding_model": embedding_model,
                "model_metadata": model_metadata,
                "embedding_metadata": embedding_metadata,
                "options": options,
                "code_provenance": {
                    "sha256": code_sha256,
                    "files": code_files,
                },
            },
            "gates": {
                "real_embedding": (
                    "OBSERVED" if observation.verified_embedding_provenance else "NOT VERIFIED"
                ),
                "real_cognition": (
                    "OBSERVED" if observation.adapter_kind == "real_model" else "NOT OBSERVED"
                ),
                "controlled_state_ablation": "NOT MEASURED",
                "length_filter_decision": "NOT MEASURED",
                "human_acceptance": "NOT RUN",
            },
            "observation": observation.as_dict(),
        }
        path = output_dir / f"development-{session.session_id}-{profile.name}.json"
        # Some provider/job libraries retain exception objects in diagnostic
        # metadata. Stringifying only at the artifact boundary preserves those
        # failures without allowing one unrepresentable detail to discard the
        # complete run record.
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        reports[profile.name] = artifact
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("datasets") / "local-memory-heldout-v3.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", default="dev-001")
    parser.add_argument(
        "--discovery",
        type=Path,
        default=Path("docs/validation/2026-09-16/ollama-discovery.json"),
    )
    parser.add_argument("--endpoint", default="http://192.168.86.35:11434")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--profile", action="append", dest="profiles")
    args = parser.parse_args(argv)
    reports = run_development_smoke(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        session_id=args.session,
        discovery_path=args.discovery,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        profile_names=set(args.profiles) if args.profiles else None,
    )
    print(json.dumps({name: report["status"] for name, report in reports.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
