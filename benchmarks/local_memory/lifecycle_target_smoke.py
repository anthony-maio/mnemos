"""Development-only smoke for explicit correction and forget durability.

This diagnostic uses the normal Ollama evaluator adapter and a deliberately
small authored fixture. It reports whether the lifecycle commands were
accepted and processed across restart boundaries. It does not score answer
quality or establish a human acceptance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from mnemos.scopes import ProjectIdentity, ScopeRegistry

from .contracts import FactRecord, QueryRecord, SessionScenario, SourceEvent
from .dev_smoke import _code_provenance
from .harness import DEFAULT_PROFILES, ExperimentProfile, RunObservation
from .heldout_run import _live_model_metadata
from .ollama_adapter import OllamaEvaluatorAdapter, make_ollama_service_factory

DEFAULT_DATASET = Path(__file__).with_name("datasets") / "lifecycle-target-development-v1.json"
USER_PROFILE_ID = "lifecycle-target-smoke-profile"
SCHEMA_VERSION = "local-memory-lifecycle-target-smoke-v1"


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _tuple_text(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _load_scenarios(path: Path) -> tuple[SessionScenario, ...]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("lifecycle target fixture is malformed JSON") from exc
    if not isinstance(root, Mapping) or root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported lifecycle target fixture schema")
    rows = root.get("scenarios")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("lifecycle target fixture must contain exactly two scenarios")
    scenarios: list[SessionScenario] = []
    all_fact_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"scenario {index} must be an object")
        scenario_id = _text(raw.get("session_id"), f"scenario {index} session_id")
        if raw.get("split") != "development":
            raise ValueError(f"scenario {scenario_id} must be development")
        facts_raw = raw.get("facts")
        events_raw = raw.get("events")
        queries_raw = raw.get("queries", [])
        if (
            not isinstance(facts_raw, list)
            or not isinstance(events_raw, list)
            or not isinstance(queries_raw, list)
        ):
            raise ValueError(f"scenario {scenario_id} facts, events, and queries must be lists")
        facts: list[FactRecord] = []
        for fact_raw in facts_raw:
            if not isinstance(fact_raw, Mapping):
                raise ValueError(f"scenario {scenario_id} has a malformed fact")
            fact_id = _text(fact_raw.get("fact_id"), "fact_id")
            if fact_id in all_fact_ids:
                raise ValueError(f"fact ID {fact_id!r} is reused across scenarios")
            all_fact_ids.add(fact_id)
            facts.append(
                FactRecord(
                    fact_id=fact_id,
                    content=_text(fact_raw.get("content"), "fact content"),
                    critical=bool(fact_raw.get("critical", False)),
                    scope=_text(fact_raw.get("scope", "project"), "fact scope"),
                    source_label=_text(
                        fact_raw.get("source_label", "authored_development"), "fact source_label"
                    ),
                    gold_status=_text(fact_raw.get("gold_status"), "fact gold_status"),
                    evidence_event_ids=_tuple_text(
                        fact_raw.get("evidence_event_ids", []), "fact evidence_event_ids"
                    ),
                )
            )
        events: list[SourceEvent] = []
        fact_ids = {fact.fact_id for fact in facts}
        for event_raw in events_raw:
            if not isinstance(event_raw, Mapping):
                raise ValueError(f"scenario {scenario_id} has a malformed event")
            events.append(
                SourceEvent(
                    event_id=_text(event_raw.get("event_id"), "event_id"),
                    occurred_at=_text(event_raw.get("occurred_at"), "occurred_at"),
                    role=_text(event_raw.get("role"), "event role"),
                    content=_text(event_raw.get("content"), "event content"),
                    source_kind=_text(event_raw.get("source_kind"), "event source_kind"),
                    write_mode=_text(event_raw.get("write_mode"), "event write_mode"),
                    lifecycle_operation=_text(
                        event_raw.get("lifecycle_operation"), "lifecycle_operation"
                    ),
                    evidence_fact_ids=_tuple_text(
                        event_raw.get("evidence_fact_ids", []), "event evidence_fact_ids"
                    ),
                    target_user_profile_id=event_raw.get("target_user_profile_id"),
                    target_project_id=event_raw.get("target_project_id"),
                    target_workspace_id=event_raw.get("target_workspace_id"),
                    target_source_event_id=event_raw.get("target_source_event_id"),
                    target_fact_id=event_raw.get("target_fact_id"),
                    replacement_content=event_raw.get("replacement_content"),
                )
            )
        if any(set(event.evidence_fact_ids) - fact_ids for event in events):
            raise ValueError(f"scenario {scenario_id} contains an event with unknown evidence")
        event_ids = {event.event_id for event in events}
        if any(set(fact.evidence_event_ids) - event_ids for fact in facts):
            raise ValueError(f"scenario {scenario_id} contains a fact with unknown evidence")
        queries: list[QueryRecord] = []
        for query_raw in queries_raw:
            if not isinstance(query_raw, Mapping):
                raise ValueError(f"scenario {scenario_id} has a malformed query")
            queries.append(
                QueryRecord(
                    query_id=_text(query_raw.get("query_id"), "query_id"),
                    text=_text(query_raw.get("text"), "query text"),
                    kind=_text(query_raw.get("kind"), "query kind"),
                    relevant_fact_ids=_tuple_text(
                        query_raw.get("relevant_fact_ids", []), "query relevant_fact_ids"
                    ),
                    expected_answer=query_raw.get("expected_answer"),
                )
            )
        scenarios.append(
            SessionScenario(
                session_id=scenario_id,
                split="development",
                host=_text(raw.get("host"), "scenario host"),
                project_id=_text(raw.get("project_id"), "scenario project_id"),
                facts=tuple(facts),
                queries=tuple(queries),
                state_pairs=(),
                source_label="authored_development",
                human_acceptance="not_run",
                events=tuple(events),
                scenario_types=tuple(str(item) for item in raw.get("scenario_types", [])),
            )
        )
    if {scenario.session_id for scenario in scenarios} != {
        "development-correct-restart",
        "development-forget-replay",
    }:
        raise ValueError("fixture scenario IDs do not match the two lifecycle targets")
    return tuple(scenarios)


def load_target_scenarios(path: Path = DEFAULT_DATASET) -> tuple[SessionScenario, ...]:
    """Load the two authored development scenarios without held-out minimums."""

    return _load_scenarios(path)


def _operation(raw: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    operations = raw.get("operations", [])
    return [
        item for item in operations if isinstance(item, Mapping) and item.get("operation") == name
    ]


def _operation_index(raw: Mapping[str, Any], name: str) -> list[int]:
    operations = raw.get("operations", [])
    if not isinstance(operations, list):
        return []
    return [
        index
        for index, item in enumerate(operations)
        if isinstance(item, Mapping) and item.get("operation") == name
    ]


def _maintenance_completed(
    raw: Mapping[str, Any], *, before: str | None = None, kind: str | None = None
) -> bool:
    for record in _operation(raw, "maintenance"):
        if before is not None and record.get("before") != before:
            continue
        if record.get("status") != "completed":
            continue
        jobs = record.get("jobs", [])
        if not isinstance(jobs, list):
            continue
        if any(
            isinstance(job, Mapping)
            and job.get("status") == "completed"
            and (kind is None or job.get("kind") == kind)
            for job in jobs
        ):
            return True
    return False


def _diagnose_observation(session: SessionScenario, observation: RunObservation) -> dict[str, Any]:
    raw = dict(observation.raw_output)
    operation_rows = raw.get("operations", [])
    if not isinstance(operation_rows, list):
        operation_rows = []
    result: dict[str, Any] = {
        "session_id": session.session_id,
        "execution_status": observation.execution_status,
        "operations": operation_rows,
        "errors": raw.get("errors", []),
        "quality": "NOT MEASURED",
        "human_acceptance": "NOT RUN",
    }
    execution_ok = observation.execution_status == "complete" and not raw.get("errors")
    if session.session_id == "development-correct-restart":
        correct_rows = _operation(raw, "correct")
        correct_dispatched = any(row.get("status") == "committed" for row in correct_rows)
        target_ids = {
            row["fact_id"]
            for row in correct_rows
            if row.get("status") == "committed" and isinstance(row.get("fact_id"), str)
        }
        # The adapter drains jobs before dispatching a correction and again at
        # the end of the session. The correction job therefore appears in a
        # maintenance record without a ``before`` marker.
        correct_job_processed = _maintenance_completed(raw, kind="correct")
        restart_indexes = _operation_index(raw, "restart")
        correct_indexes = _operation_index(raw, "correct")
        restarts = [row for row in _operation(raw, "restart") if row.get("status") == "completed"]
        retrieval_rows = _operation(raw, "retrieve")
        retrieval_indexes = _operation_index(raw, "retrieve")
        restart_index = restart_indexes[0] if restart_indexes else -1
        correct_index = correct_indexes[0] if correct_indexes else -1
        retrieved_after_restart = [
            row for index, row in zip(retrieval_indexes, retrieval_rows) if index > restart_index
        ]
        current_versions = [
            fact.get("version")
            for row in retrieved_after_restart
            for fact in (row.get("facts", []) if isinstance(row.get("facts"), list) else [])
            if isinstance(fact, Mapping) and isinstance(fact.get("version"), int)
        ]
        replacement_content = next(
            (
                event.replacement_content
                for event in session.events
                if event.lifecycle_operation == "correct"
            ),
            None,
        )
        current_target_versions = [
            fact.get("version")
            for row in retrieved_after_restart
            for fact in (row.get("facts", []) if isinstance(row.get("facts"), list) else [])
            if isinstance(fact, Mapping)
            and fact.get("id") in target_ids
            and fact.get("content") == replacement_content
            and isinstance(fact.get("version"), int)
        ]
        result.update(
            {
                "target": "correct",
                "correct_dispatched": correct_dispatched,
                "correct_job_processed": correct_job_processed,
                "restart_observed": bool(restarts),
                "restart_after_correction": bool(restarts) and restart_index > correct_index,
                "retrieved_versions": current_versions,
                "retrieved_target_versions": current_target_versions,
                "current_version_observed": any(
                    isinstance(version, int) and version >= 2 for version in current_target_versions
                ),
                "status": (
                    "complete"
                    if correct_dispatched
                    and correct_job_processed
                    and execution_ok
                    and bool(restarts)
                    and restart_index > correct_index
                    and any(
                        isinstance(version, int) and version >= 2
                        for version in current_target_versions
                    )
                    else "blocked"
                ),
            }
        )
    else:
        forget_rows = _operation(raw, "forget")
        remember_dispatched = any(
            row.get("status") == "committed" for row in _operation(raw, "remember")
        )
        forget_dispatched = any(
            row.get("status") == "committed" and row.get("forgotten") is True for row in forget_rows
        )
        remember_job_processed = _maintenance_completed(raw, before="forget", kind="remember")
        replay_suppressed = any(
            row.get("processing_status") == "suppressed" for row in _operation(raw, "remember")
        )
        restart_indexes = _operation_index(raw, "restart")
        forget_indexes = _operation_index(raw, "forget")
        restarts = [row for row in _operation(raw, "restart") if row.get("status") == "completed"]
        retrieval_rows = _operation(raw, "retrieve")
        retrieval_indexes = _operation_index(raw, "retrieve")
        restart_index = restart_indexes[0] if restart_indexes else -1
        forget_index = forget_indexes[0] if forget_indexes else -1
        retrieved_after_restart = [
            row for index, row in zip(retrieval_indexes, retrieval_rows) if index > restart_index
        ]
        abstained = bool(retrieved_after_restart) and all(
            isinstance(row.get("facts"), list) and not row.get("facts")
            for row in retrieved_after_restart
        )
        result.update(
            {
                "target": "forget",
                "remember_dispatched": remember_dispatched,
                "remember_job_processed": remember_job_processed,
                "forget_dispatched": forget_dispatched,
                "replay_suppressed": replay_suppressed,
                "restart_observed": bool(restarts),
                "restart_after_forget": bool(restarts) and restart_index > forget_index,
                "abstain_observed": abstained,
                "status": (
                    "complete"
                    if forget_dispatched
                    and remember_job_processed
                    and replay_suppressed
                    and execution_ok
                    and bool(restarts)
                    and restart_index > forget_index
                    and abstained
                    else "blocked"
                ),
            }
        )
    return result


def run_lifecycle_target_smoke(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path,
    work_dir: Path,
    endpoint: str,
    model: str,
    embedding_model: str,
    profile_names: Sequence[str] = ("baseline", "full"),
    service_factory_builder: Callable[..., Any] = make_ollama_service_factory,
    adapter_type: type[OllamaEvaluatorAdapter] = OllamaEvaluatorAdapter,
    model_metadata_loader: Callable[
        [str, tuple[str, ...]], dict[str, dict[str, Any]]
    ] = _live_model_metadata,
) -> dict[str, Any]:
    scenarios = load_target_scenarios(dataset_path)
    requested = {str(name) for name in profile_names}
    profiles = tuple(
        profile
        for profile in DEFAULT_PROFILES
        if profile.name in requested and profile.name in {"baseline", "full"}
    )
    if not profiles or {profile.name for profile in profiles} != requested:
        raise ValueError("profile_names may select baseline and full only")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    source_sha256, source_files = _code_provenance(repo_root)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
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
    attempt_id = uuid4().hex
    attempt_root = work_dir / "attempts" / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=True)
    try:
        model_records = model_metadata_loader(endpoint, (model, embedding_model))
        model_metadata = model_records[model]
        embedding_metadata = model_records[embedding_model]
        model_digest = str(model_metadata["digest"])
        embedding_digest = str(embedding_metadata["digest"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", model_digest) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", embedding_digest
        ):
            raise ValueError("model metadata digests are unavailable or invalid")
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "interpretation": "development lifecycle diagnostic only; no quality or human-acceptance claim",
            "quality": "NOT MEASURED",
            "human_acceptance": "NOT RUN",
            "dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha256,
                "scenario_count": len(scenarios),
            },
            "source": {"sha256": source_sha256, "files": source_files},
            "model_pins": {
                "endpoint": endpoint.rstrip("/"),
                "model": model,
                "embedding_model": embedding_model,
            },
            "attempt_id": attempt_id,
            "runs": [],
            "blocked_runs": 1,
            "blocked_reason": {"error_type": type(exc).__name__, "error": str(exc)},
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = output_dir / f"lifecycle-target-smoke-{attempt_id}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(report_path)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    manifest_path = output_dir / f"lifecycle-target-smoke-{attempt_id}.manifest.json"
    raw_path = output_dir / f"lifecycle-target-smoke-{attempt_id}.raw-outputs.jsonl"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "dataset_sha256": dataset_sha256,
        "source_sha256": source_sha256,
        "source_files": source_files,
        "models": model_records,
        "endpoint": endpoint,
        "options": options,
        "options_sha256": options_sha256,
        "profiles": [profile.name for profile in profiles],
        "raw_outputs": str(raw_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        registry = ScopeRegistry(
            projects={
                scenario.project_id: ProjectIdentity(
                    id=scenario.project_id,
                    root=str(work_dir.resolve()),
                    user_profile_id=USER_PROFILE_ID,
                    enabled=True,
                )
            },
            profiles=[USER_PROFILE_ID],
        )
        for profile in profiles:
            run_key = f"{scenario.session_id}-{profile.name}-r1"
            run: dict[str, Any] = {
                "session_id": scenario.session_id,
                "profile": profile.as_dict(),
                "model_pins": {
                    "endpoint": endpoint.rstrip("/"),
                    "model": model,
                    "embedding_model": embedding_model,
                    "model_revision": model_digest,
                    "embedding_revision": embedding_digest,
                    "model_metadata": model_metadata,
                    "embedding_metadata": embedding_metadata,
                    "options": options,
                    "options_sha256": options_sha256,
                },
                "dataset_sha256": dataset_sha256,
                "source_sha256": source_sha256,
                "source_files": source_files,
                "attempt_id": attempt_id,
                "attempt_root": str(attempt_root),
            }
            try:
                profile_root = attempt_root / f"{scenario.session_id}-{profile.name}"
                profile_root.mkdir(parents=True, exist_ok=True)
                factory = service_factory_builder(
                    profile_root / f"{profile.name}.sqlite",
                    registry,
                    base_url=endpoint,
                    model=model,
                    embedding_model=embedding_model,
                    timeout=options["timeout_seconds"],
                    temperature=options["temperature"],
                    max_tokens=options["max_tokens"],
                )
                adapter = adapter_type(
                    factory,
                    base_url=endpoint,
                    model=model,
                    embedding_model=embedding_model,
                    user_profile_id=USER_PROFILE_ID,
                    real_model=True,
                    model_revision=model_digest,
                    embedding_revision=embedding_digest,
                    model_sha256=model_digest,
                    embedding_sha256=embedding_digest,
                    options_sha256=options_sha256,
                    code_sha256=source_sha256,
                    model_provenance=model_metadata,
                    embedding_provenance=embedding_metadata,
                )
                observation = adapter.evaluate_session(scenario, profile, 1)
                if not isinstance(observation, RunObservation):
                    raise TypeError("lifecycle adapter did not return RunObservation")
                run["observation"] = observation.as_dict()
                run["diagnostic"] = _diagnose_observation(scenario, observation)
                run["status"] = run["diagnostic"]["status"]
            except Exception as exc:
                run["status"] = "blocked"
                run["diagnostic"] = {
                    "session_id": scenario.session_id,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "quality": "NOT MEASURED",
                    "human_acceptance": "NOT RUN",
                }
            runs.append(run)
            # Preserve each completed attempt before starting the next model
            # call, including blocked diagnostics and their provider traces.
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(run, sort_keys=True, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    blocked = [run for run in runs if run.get("status") != "complete"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blocked else "complete",
        "interpretation": "development lifecycle diagnostic only; no quality or human-acceptance claim",
        "quality": "NOT MEASURED",
        "human_acceptance": "NOT RUN",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "scenario_count": len(scenarios),
        },
        "source": {"sha256": source_sha256, "files": source_files},
        "model_pins": {
            "endpoint": endpoint.rstrip("/"),
            "model": model_metadata,
            "embedding_model": embedding_metadata,
            "options": options,
            "options_sha256": options_sha256,
        },
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "profiles": [profile.name for profile in profiles],
        "runs": runs,
        "blocked_runs": len(blocked),
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / f"lifecycle-target-smoke-{attempt_id}.json"
    report["report_path"] = str(report_path)
    report["manifest_path"] = str(manifest_path)
    report["raw_outputs_path"] = str(raw_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    manifest.update(
        status=report["status"],
        report_path=str(report_path),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        completed_runs=len(runs),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", choices=("baseline", "full"))
    args = parser.parse_args(argv)
    report = run_lifecycle_target_smoke(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        model=args.model,
        embedding_model=args.embedding_model,
        profile_names=tuple(args.profile or ("baseline", "full")),
    )
    print(
        json.dumps(
            {"status": report["status"], "blocked_runs": report["blocked_runs"]}, sort_keys=True
        )
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
