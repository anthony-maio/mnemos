"""Adapter-based experiment harness for local-memory ablation runs.

The adapter is the only integration point with Mnemos, a local model, or an
embedding provider. This keeps the dataset and scoring preparation runnable in
an offline checkout while allowing the parent integration to supply a real
engine later.
"""

from __future__ import annotations

import argparse
import statistics
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .artifacts import (
    ArtifactRecorder,
    build_not_run_artifact,
    environment_metadata,
    file_digest,
    json_digest,
)
from .contracts import HeldoutDataset, SessionScenario, load_dataset
from .resumption import Checkpoint, load_checkpoint, save_checkpoint
from .scoring import binary_interval, paired_bootstrap_interval

MODULES = (
    "surprisal_gating",
    "mutable_rag",
    "affective_routing",
    "sleep_consolidation",
    "spreading_activation",
)


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    enabled_modules: tuple[str, ...]
    disabled_module: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled_modules": list(self.enabled_modules),
            "disabled_module": self.disabled_module,
        }


def _profile(name: str, disabled: str | None = None) -> ExperimentProfile:
    enabled = tuple(module for module in MODULES if module != disabled)
    return ExperimentProfile(name=name, enabled_modules=enabled, disabled_module=disabled)


DEFAULT_PROFILES = (
    ExperimentProfile("baseline", ()),
    ExperimentProfile("full", MODULES),
    *(_profile(f"ablate-{module}", module) for module in MODULES),
)


@dataclass(frozen=True)
class ExperimentConfig:
    profile: ExperimentProfile = DEFAULT_PROFILES[1]
    provider: str = "unconfigured"
    repetitions: int = 3
    top_k: int = 5
    max_context_tokens: int = 1200
    confidence: float = 0.95
    bootstrap_repetitions: int = 10_000
    bootstrap_seed: int = 20260916
    model_sha256: str = ""
    embedding_sha256: str = ""
    options_sha256: str = ""
    code_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.as_dict(),
            "provider": self.provider,
            "repetitions": self.repetitions,
            "top_k": self.top_k,
            "max_context_tokens": self.max_context_tokens,
            "confidence": self.confidence,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "model_sha256": self.model_sha256,
            "embedding_sha256": self.embedding_sha256,
            "options_sha256": self.options_sha256,
            "code_sha256": self.code_sha256,
        }

    def validate(self) -> None:
        if self.repetitions < 3:
            raise ValueError("T18 requires at least three real-model repetitions")
        if self.top_k <= 0 or self.max_context_tokens <= 0:
            raise ValueError("top_k and max_context_tokens must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.profile.name == "full" and set(self.profile.enabled_modules) != set(MODULES):
            raise ValueError("full profile must enable all five modules")
        if self.profile.name == "baseline" and self.profile.enabled_modules:
            raise ValueError("baseline profile cannot enable biomimetic modules")
        if set(self.profile.enabled_modules) - set(MODULES):
            raise ValueError("profile contains an unknown module")


@dataclass(frozen=True)
class RunObservation:
    """One adapter result for one session and repetition.

    The adapter must report measured values. Missing optional measurements stay
    null in the artifact and are never replaced with synthetic estimates.
    """

    session_id: str
    repetition: int
    critical_retention: tuple[bool | None, ...] = ()
    critical_retrieval: tuple[bool | None, ...] = ()
    supported_precision: tuple[bool | None, ...] = ()
    answerable_hits: tuple[bool | None, ...] = ()
    no_answer_correct: tuple[bool | None, ...] = ()
    state_pair_semantic_hits: tuple[bool | None, ...] = ()
    state_pair_state_hits: tuple[bool | None, ...] = ()
    surprisal_filter_kept: tuple[bool | None, ...] = ()
    length_filter_kept: tuple[bool | None, ...] = ()
    stale_fact_count: int | None = None
    retrieved_fact_count: int | None = None
    latency_ms: float | None = None
    cold_latency_ms: float | None = None
    token_cost: float | None = None
    bytes_before: int | None = None
    bytes_after: int | None = None
    provider: str | None = None
    model_revision: str | None = None
    embedding_revision: str | None = None
    execution_status: str = "partial"
    adapter_kind: str = "unknown"
    verified_embedding_provenance: bool = False
    verified_model_provenance: bool = False
    raw_output: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "repetition": self.repetition,
            "critical_retention": list(self.critical_retention),
            "critical_retrieval": list(self.critical_retrieval),
            "supported_precision": list(self.supported_precision),
            "answerable_hits": list(self.answerable_hits),
            "no_answer_correct": list(self.no_answer_correct),
            "state_pair_semantic_hits": list(self.state_pair_semantic_hits),
            "state_pair_state_hits": list(self.state_pair_state_hits),
            "surprisal_filter_kept": list(self.surprisal_filter_kept),
            "length_filter_kept": list(self.length_filter_kept),
            "stale_fact_count": self.stale_fact_count,
            "retrieved_fact_count": self.retrieved_fact_count,
            "latency_ms": self.latency_ms,
            "cold_latency_ms": self.cold_latency_ms,
            "token_cost": self.token_cost,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "provider": self.provider,
            "model_revision": self.model_revision,
            "embedding_revision": self.embedding_revision,
            "execution_status": self.execution_status,
            "adapter_kind": self.adapter_kind,
            "verified_embedding_provenance": self.verified_embedding_provenance,
            "verified_model_provenance": self.verified_model_provenance,
            "raw_output": dict(self.raw_output),
        }


class ExperimentAdapter(Protocol):
    """Boundary implemented by a real local-model/engine integration."""

    def evaluate_session(
        self, session: SessionScenario, profile: ExperimentProfile, repetition: int
    ) -> RunObservation | Mapping[str, Any]:
        """Run one session and return measured observations only."""


class ExperimentHarness:
    def __init__(self, dataset: HeldoutDataset, config: ExperimentConfig) -> None:
        config.validate()
        self.dataset = dataset
        self.config = config
        self.config_sha256 = json_digest(config.as_dict())

    def run(
        self,
        adapter: ExperimentAdapter,
        *,
        checkpoint_path: Path | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute held-out sessions and return an artifact-ready report."""
        identifier = run_id or str(uuid.uuid4())
        sessions = self.dataset.heldout_sessions
        checkpoint: Checkpoint | None = None
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = load_checkpoint(
                checkpoint_path,
                dataset_sha256=self.dataset.digest,
                config_sha256=self.config_sha256,
                profile=self.config.profile.name,
                repetitions=self.config.repetitions,
                model_sha256=self.config.model_sha256,
                embedding_sha256=self.config.embedding_sha256,
                options_sha256=self.config.options_sha256,
                code_sha256=self.config.code_sha256,
                run_id=run_id,
            )
            identifier = checkpoint.run_id
        elif checkpoint_path:
            checkpoint = Checkpoint(
                run_id=identifier,
                dataset_sha256=self.dataset.digest,
                config_sha256=self.config_sha256,
                profile=self.config.profile.name,
                repetitions=self.config.repetitions,
                model_sha256=self.config.model_sha256,
                embedding_sha256=self.config.embedding_sha256,
                options_sha256=self.config.options_sha256,
                code_sha256=self.config.code_sha256,
            )
            save_checkpoint(checkpoint_path, checkpoint)

        observations: list[RunObservation] = []
        if checkpoint:
            observations = [
                _coerce_observation(
                    item,
                    str(item.get("session_id", "")),
                    int(item.get("repetition", 0)),
                )
                for item in checkpoint.completed_observations
            ]
        start_index = checkpoint.next_session_index if checkpoint else 0
        completed = set(checkpoint.completed_observation_ids if checkpoint else ())
        for session_index, session in enumerate(sessions[start_index:], start=start_index):
            for repetition in range(self.config.repetitions):
                observation_id = f"{session.session_id}:r{repetition + 1}"
                if observation_id in completed:
                    continue
                observed = adapter.evaluate_session(session, self.config.profile, repetition + 1)
                result = _coerce_observation(observed, session.session_id, repetition + 1)
                _validate_observation_shape(session, result)
                observations.append(result)
                completed.add(observation_id)
                if checkpoint_path:
                    checkpoint = Checkpoint(
                        run_id=identifier,
                        dataset_sha256=self.dataset.digest,
                        config_sha256=self.config_sha256,
                        profile=self.config.profile.name,
                        repetitions=self.config.repetitions,
                        model_sha256=self.config.model_sha256,
                        embedding_sha256=self.config.embedding_sha256,
                        options_sha256=self.config.options_sha256,
                        code_sha256=self.config.code_sha256,
                        next_session_index=session_index,
                        completed_observation_ids=sorted(completed),
                        completed_observations=[item.as_dict() for item in observations],
                    )
                    save_checkpoint(checkpoint_path, checkpoint)
        if checkpoint_path:
            checkpoint = Checkpoint(
                run_id=identifier,
                dataset_sha256=self.dataset.digest,
                config_sha256=self.config_sha256,
                profile=self.config.profile.name,
                repetitions=self.config.repetitions,
                model_sha256=self.config.model_sha256,
                embedding_sha256=self.config.embedding_sha256,
                options_sha256=self.config.options_sha256,
                code_sha256=self.config.code_sha256,
                next_session_index=len(sessions),
                completed_observation_ids=sorted(completed),
                completed_observations=[item.as_dict() for item in observations],
                status=(
                    "complete"
                    if len(observations) == len(sessions) * self.config.repetitions
                    and all(item.execution_status == "complete" for item in observations)
                    else "incomplete"
                ),
            )
            save_checkpoint(checkpoint_path, checkpoint)
        return build_completed_artifact(self.dataset, self.config, observations, identifier)

    def run_profiles(
        self,
        adapter: ExperimentAdapter,
        profiles: Sequence[ExperimentProfile] = DEFAULT_PROFILES,
    ) -> dict[str, dict[str, Any]]:
        """Run baseline, full, and ablations with identical session ordering."""
        reports: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            config = ExperimentConfig(
                profile=profile,
                provider=self.config.provider,
                repetitions=self.config.repetitions,
                top_k=self.config.top_k,
                max_context_tokens=self.config.max_context_tokens,
                confidence=self.config.confidence,
                bootstrap_repetitions=self.config.bootstrap_repetitions,
                bootstrap_seed=self.config.bootstrap_seed,
                model_sha256=self.config.model_sha256,
                embedding_sha256=self.config.embedding_sha256,
                options_sha256=self.config.options_sha256,
                code_sha256=self.config.code_sha256,
            )
            reports[profile.name] = ExperimentHarness(self.dataset, config).run(adapter)
        baseline = reports.get("baseline")
        if baseline:
            baseline["paired_comparisons"] = {
                name: compare_observation_metrics(baseline, report, config=self.config)
                for name, report in reports.items()
                if name != "baseline"
            }
        return reports


def _coerce_observation(
    value: RunObservation | Mapping[str, Any], session_id: str, repetition: int
) -> RunObservation:
    if isinstance(value, RunObservation):
        if value.session_id != session_id or value.repetition != repetition:
            raise ValueError("adapter returned an observation for the wrong session or repetition")
        return value
    if not isinstance(value, Mapping):
        raise TypeError("adapter result must be RunObservation or a mapping")
    kwargs: dict[str, Any] = {"session_id": session_id, "repetition": repetition}
    fields = set(RunObservation.__dataclass_fields__)
    for key, raw in value.items():
        if key in fields and key not in {"session_id", "repetition"}:
            if key in {
                "critical_retention",
                "critical_retrieval",
                "supported_precision",
                "answerable_hits",
                "no_answer_correct",
                "state_pair_semantic_hits",
                "state_pair_state_hits",
                "surprisal_filter_kept",
                "length_filter_kept",
            }:
                kwargs[key] = _coerce_bool_sequence(raw, key)
            else:
                kwargs[key] = raw
    return RunObservation(**kwargs)


def _coerce_bool_sequence(raw: Any, field_name: str) -> tuple[bool | None, ...]:
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"adapter field {field_name} must be a list of JSON booleans")
    if any(item is not None and type(item) is not bool for item in raw):
        raise TypeError(f"adapter field {field_name} contains a non-boolean value")
    return tuple(raw)


def _validate_observation_shape(session: SessionScenario, observation: RunObservation) -> None:
    expected_lengths = {
        "critical_retention": sum(fact.critical for fact in session.facts),
        "critical_retrieval": sum(fact.critical for fact in session.facts),
        "answerable_hits": sum(query.kind == "answerable" for query in session.queries),
        "no_answer_correct": sum(query.kind == "no_answer" for query in session.queries),
        "state_pair_semantic_hits": len(session.state_pairs),
        "state_pair_state_hits": len(session.state_pairs),
    }
    for field_name, expected in expected_lengths.items():
        raw = getattr(observation, field_name)
        if any(item is not None and type(item) is not bool for item in raw):
            raise TypeError(f"observation {field_name} contains a non-boolean value")
        if len(raw) != expected:
            raise ValueError(
                f"observation {session.session_id} {field_name} has {len(raw)} values; expected {expected}"
            )
    if observation.execution_status not in {"complete", "partial", "rejected", "missing"}:
        raise ValueError(f"unknown execution_status {observation.execution_status!r}")
    if type(observation.verified_model_provenance) is not bool:
        raise TypeError("verified_model_provenance must be a boolean")
    if type(observation.verified_embedding_provenance) is not bool:
        raise TypeError("verified_embedding_provenance must be a boolean")
    for field_name in (
        "stale_fact_count",
        "retrieved_fact_count",
        "bytes_before",
        "bytes_after",
    ):
        value = getattr(observation, field_name)
        if value is not None and (type(value) is not int or value < 0):
            raise TypeError(f"observation {field_name} must be a non-negative integer or null")
    for field_name in ("latency_ms", "cold_latency_ms", "token_cost"):
        value = getattr(observation, field_name)
        if value is not None and (type(value) not in {int, float} or value < 0):
            raise TypeError(f"observation {field_name} must be a non-negative number or null")


def build_completed_artifact(
    dataset: HeldoutDataset,
    config: ExperimentConfig,
    observations: Sequence[RunObservation],
    run_id: str,
) -> dict[str, Any]:
    """Summarize measured observations with uncertainty and sample counts."""
    expected_keys = {
        (session.session_id, repetition)
        for session in dataset.heldout_sessions
        for repetition in range(1, config.repetitions + 1)
    }
    actual_keys = {(observation.session_id, observation.repetition) for observation in observations}
    if actual_keys != expected_keys or len(observations) != len(actual_keys):
        raise ValueError(
            "incomplete run: observations must cover every session and repetition exactly once"
        )
    sessions_by_id = {session.session_id: session for session in dataset.heldout_sessions}
    for observation in observations:
        _validate_observation_shape(sessions_by_id[observation.session_id], observation)
    metrics = _build_observation_metrics(observations, config.confidence)
    complete_observations = [
        observation for observation in observations if observation.execution_status == "complete"
    ]
    incomplete_count = len(observations) - len(complete_observations)
    for metric in metrics.values():
        if isinstance(metric, dict):
            metric["diagnostic_when_incomplete"] = incomplete_count > 0
            metric["aggregation_scope"] = "all_observations"
    complete_metrics = _build_observation_metrics(complete_observations, config.confidence)
    for metric in complete_metrics.values():
        if isinstance(metric, dict):
            metric["diagnostic_when_incomplete"] = False
            metric["parent_run_incomplete"] = incomplete_count > 0
            metric["aggregation_scope"] = "complete_observations"
    provider_values = sorted({item.provider for item in observations if item.provider})
    model_values = sorted({item.model_revision for item in observations if item.model_revision})
    embedding_values = sorted(
        {item.embedding_revision for item in observations if item.embedding_revision}
    )
    measurement_fields = (
        "critical_retention",
        "critical_retrieval",
        "supported_precision",
        "answerable_hits",
        "no_answer_correct",
        "state_pair_semantic_hits",
        "state_pair_state_hits",
    )
    no_unjudged_measurements = all(
        all(value is not None for value in getattr(item, field_name))
        for item in observations
        for field_name in measurement_fields
    )
    return {
        "schema_version": "local-memory-artifact-v1",
        "status": (
            "completed"
            if all(item.execution_status == "complete" for item in observations)
            else "incomplete"
        ),
        "run_id": run_id,
        "gates": {
            "real_embedding": (
                "OBSERVED"
                if all(
                    item.execution_status == "complete"
                    and item.adapter_kind in {"real_embedding_observed", "real_model"}
                    and item.verified_embedding_provenance
                    for item in observations
                )
                else "NOT VERIFIED"
            ),
            "real_model": (
                "OBSERVED"
                if all(
                    item.execution_status == "complete"
                    and item.adapter_kind == "real_model"
                    and item.verified_model_provenance
                    and item.provider
                    and item.model_revision
                    and item.embedding_revision
                    and config.model_sha256
                    and config.embedding_sha256
                    and config.options_sha256
                    and config.code_sha256
                    and no_unjudged_measurements
                    for item in observations
                )
                else "NOT VERIFIED"
            ),
            "controlled_state_ablation": "NOT MEASURED",
            "length_filter_decision": "NOT MEASURED",
            "human_acceptance": "NOT RUN",
        },
        "dataset": {
            "sha256": dataset.digest,
            "version": dataset.version,
            "bytes": file_digest(dataset.source_path)["bytes"] if dataset.source_path else None,
            "source_label": "synthetic_seed",
            "advertised_as_real": False,
        },
        "profile": config.profile.as_dict(),
        "config": config.as_dict(),
        "config_sha256": json_digest(config.as_dict()),
        "environment": environment_metadata(),
        "provider": provider_values,
        "model_revisions": model_values,
        "embedding_revisions": embedding_values,
        "repetitions": config.repetitions,
        "sample_counts": {
            "sessions": len(dataset.heldout_sessions),
            "observations": len(observations),
            "critical_facts": len(
                [
                    fact
                    for session in dataset.heldout_sessions
                    for fact in session.facts
                    if fact.critical
                ]
            ),
            "answerable_queries": len(
                [
                    query
                    for session in dataset.heldout_sessions
                    for query in session.queries
                    if query.kind == "answerable"
                ]
            ),
            "no_answer_queries": len(
                [
                    query
                    for session in dataset.heldout_sessions
                    for query in session.queries
                    if query.kind == "no_answer"
                ]
            ),
            "state_pairs": len(dataset.state_pairs),
        },
        "measurements": metrics,
        "complete_observation_diagnostics": {
            "observation_count": len(complete_observations),
            "excluded_incomplete_observation_count": incomplete_count,
            "parent_run_incomplete": incomplete_count > 0,
            "status": "ready" if incomplete_count == 0 else "partial_available",
            "measurements": complete_metrics,
        },
        "lifecycle_operations": _aggregate_lifecycle_operations(observations),
        "raw_outputs": [observation.as_dict() for observation in observations],
    }


def compare_observation_metrics(
    baseline_artifact: Mapping[str, Any],
    treatment_artifact: Mapping[str, Any],
    *,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Compare paired raw observations, reporting eligible and excluded pairs."""
    base_rows = {
        (str(row.get("session_id")), int(row.get("repetition", 0))): row
        for row in baseline_artifact.get("raw_outputs", [])
        if isinstance(row, Mapping)
    }
    treatment_rows = {
        (str(row.get("session_id")), int(row.get("repetition", 0))): row
        for row in treatment_artifact.get("raw_outputs", [])
        if isinstance(row, Mapping)
    }
    paired_keys = sorted(set(base_rows) & set(treatment_rows))
    result: dict[str, Any] = {
        "paired_observations": len(paired_keys),
        "metrics": {},
        "pair_counts": {},
    }
    for metric in (
        "answerable_hits",
        "no_answer_correct",
        "critical_retention",
        "critical_retrieval",
    ):
        baseline_values: list[float] = []
        treatment_values: list[float] = []
        paired_clusters: list[str] = []
        excluded_reasons = {
            "incomplete_execution": 0,
            "missing_components": 0,
            "no_applicable_observations": 0,
        }
        for key in paired_keys:
            baseline_row = base_rows[key]
            treatment_row = treatment_rows[key]
            baseline_status = baseline_row.get("execution_status")
            treatment_status = treatment_row.get("execution_status")
            if baseline_status != "complete" or treatment_status != "complete":
                excluded_reasons["incomplete_execution"] += 1
                continue
            baseline_raw = baseline_row.get(metric)
            treatment_raw = treatment_row.get(metric)
            if baseline_raw in ([], ()) or treatment_raw in ([], ()):
                excluded_reasons["no_applicable_observations"] += 1
                continue
            baseline_value = _mean_bool(baseline_raw)
            treatment_value = _mean_bool(treatment_raw)
            if baseline_value is not None and treatment_value is not None:
                baseline_values.append(baseline_value)
                treatment_values.append(treatment_value)
                paired_clusters.append(key[0])
            else:
                excluded_reasons["missing_components"] += 1
        eligible_count = len(baseline_values)
        result["pair_counts"][metric] = {
            "total_count": len(paired_keys),
            "eligible_count": eligible_count,
            "excluded_count": len(paired_keys) - eligible_count,
            "excluded_reasons": excluded_reasons,
        }
        if baseline_values:
            interval = paired_bootstrap_interval(
                baseline_values,
                treatment_values,
                confidence=config.confidence,
                repetitions=config.bootstrap_repetitions,
                seed=config.bootstrap_seed,
                cluster_ids=paired_clusters,
            ).as_dict()
            interval.update(result["pair_counts"][metric])
            result["metrics"][metric] = interval
        else:
            result["metrics"][metric] = {
                "status": "NOT MEASURED",
                "sample_count": 0,
                **result["pair_counts"][metric],
            }
    return result


def _mean_bool(raw: Any) -> float | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    if any(value is None for value in raw):
        return None
    return sum(bool(value) for value in raw) / len(raw)


def evaluate_with_adapter(
    dataset_path: Path,
    config: ExperimentConfig,
    adapter: ExperimentAdapter,
    *,
    output: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    artifact = ExperimentHarness(dataset, config).run(adapter, checkpoint_path=checkpoint_path)
    if output:
        rows = [row for row in artifact.get("raw_outputs", []) if isinstance(row, dict)]
        ArtifactRecorder(output).write_run(artifact, rows)
    return artifact


def run_experiment(
    dataset_path: Path,
    config: ExperimentConfig,
    adapter: ExperimentAdapter,
    *,
    output: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for callers wiring the real adapter."""
    return evaluate_with_adapter(
        dataset_path,
        config,
        adapter,
        output=output,
        checkpoint_path=checkpoint_path,
    )


def _build_observation_metrics(
    observations: Sequence[RunObservation], confidence: float
) -> dict[str, Any]:
    """Aggregate one explicitly named observation population."""
    fields = {
        "critical_retention": [
            item for observation in observations for item in observation.critical_retention
        ],
        "critical_retrieval": [
            item for observation in observations for item in observation.critical_retrieval
        ],
        "supported_precision": [
            item for observation in observations for item in observation.supported_precision
        ],
        "answerable_recall_at_k": [
            item for observation in observations for item in observation.answerable_hits
        ],
        "no_answer_accuracy": [
            item for observation in observations for item in observation.no_answer_correct
        ],
        "state_pair_semantic": [
            item for observation in observations for item in observation.state_pair_semantic_hits
        ],
        "state_pair_state": [
            item for observation in observations for item in observation.state_pair_state_hits
        ],
        "surprisal_filter_kept": [
            item for observation in observations for item in observation.surprisal_filter_kept
        ],
        "length_filter_kept": [
            item for observation in observations for item in observation.length_filter_kept
        ],
    }
    metrics: dict[str, Any] = {}
    for name, all_values in fields.items():
        values = [item for item in all_values if item is not None]
        missing_count = len(all_values) - len(values)
        metric = (
            binary_interval(values, confidence).as_dict()
            if values
            else {"status": "NOT MEASURED", "sample_count": 0}
        )
        metric.update(
            {
                "measured_count": len(values),
                "total_count": len(all_values),
                "unjudged_count": missing_count,
                "missing_count": missing_count,
            }
        )
        metrics[name] = metric
    metrics["latency_ms"] = _numeric_summary(
        [item.latency_ms for item in observations if item.latency_ms is not None]
    )
    metrics["latency_ms"]["measurement"] = "end_to_end_session_wall_clock"
    metrics["cold_latency_ms"] = _numeric_summary(
        [item.cold_latency_ms for item in observations if item.cold_latency_ms is not None]
    )
    metrics["cold_latency_ms"]["measurement"] = "service_construction_only"
    metrics["stale_fact_rate"] = _rate_from_counts(
        observations, "stale_fact_count", "retrieved_fact_count"
    )
    metrics["stale_fact_rate"]["measurement"] = "returned_row_substring_proxy"
    metrics["stale_fact_rate"][
        "interpretation"
    ] = "Returned-row substring proxy; it is not unique-fact or query-level accuracy."
    metrics["row_or_byte_growth"] = _byte_summary(observations)
    metrics["token_cost"] = _numeric_summary(
        [item.token_cost for item in observations if item.token_cost is not None]
    )
    metrics["extraction_precision"] = metrics["supported_precision"]
    metrics["retrieval_accuracy"] = metrics["answerable_recall_at_k"]
    return metrics


def _aggregate_lifecycle_operations(
    observations: Sequence[RunObservation],
) -> dict[str, dict[str, int]]:
    """Combine per-observation lifecycle outcomes without inventing results."""
    fields = ("attempted", "committed", "rejected", "mapping_invalid", "missing_target")
    totals = {operation: {field: 0 for field in fields} for operation in ("correct", "forget")}
    for observation in observations:
        summary = observation.raw_output.get("lifecycle_summary")
        if not isinstance(summary, Mapping):
            continue
        for operation in totals:
            counts = summary.get(operation)
            if not isinstance(counts, Mapping):
                continue
            for field in fields:
                value = counts.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    totals[operation][field] += value
    return totals


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"status": "NOT MEASURED", "sample_count": 0}
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * 0.95 + 0.999999) - 1)))
    return {
        "sample_count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _rate_from_counts(
    observations: Sequence[RunObservation], numerator: str, denominator: str
) -> dict[str, Any]:
    pairs = [(getattr(item, numerator), getattr(item, denominator)) for item in observations]
    pairs = [(int(n), int(d)) for n, d in pairs if n is not None and d is not None and d > 0]
    if not pairs:
        return {"status": "NOT MEASURED", "sample_count": 0}
    return {
        "estimate": sum(n for n, _ in pairs) / sum(d for _, d in pairs),
        "numerator": sum(n for n, _ in pairs),
        "denominator": sum(d for _, d in pairs),
        "sample_count": len(pairs),
    }


def _byte_summary(observations: Sequence[RunObservation]) -> dict[str, Any]:
    values = [
        (item.bytes_before, item.bytes_after)
        for item in observations
        if item.bytes_before is not None and item.bytes_after is not None
    ]
    if not values:
        return {"status": "NOT MEASURED", "sample_count": 0}
    growth = [after - before for before, after in values]
    return {
        "sample_count": len(growth),
        "total_growth_bytes": sum(growth),
        "mean_growth_bytes": statistics.fmean(growth),
        "actual_bytes_reported": True,
    }


def _select_profile(name: str, disabled: str | None) -> ExperimentProfile:
    if disabled and disabled not in MODULES:
        raise ValueError(f"unknown module {disabled!r}")
    if name == "baseline":
        if disabled:
            raise ValueError("--disable-module cannot be combined with --profile baseline")
        return DEFAULT_PROFILES[0]
    if name == "full":
        return _profile("full", disabled)
    for candidate in DEFAULT_PROFILES:
        if candidate.name == name:
            if disabled:
                raise ValueError("--disable-module is only valid with --profile full")
            return candidate
    raise ValueError(f"unknown profile {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mnemos-local-memory-benchmark")
    default_dataset = Path(__file__).with_name("datasets") / "local-memory-heldout-v2.jsonl"
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument(
        "--profile",
        choices=("baseline", "full", *(profile.name for profile in DEFAULT_PROFILES[2:])),
        default="full",
    )
    parser.add_argument("--disable-module", choices=MODULES, default=None)
    parser.add_argument("--provider", default="unconfigured")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("local-memory-artifact.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    profile = _select_profile(args.profile, args.disable_module)
    config = ExperimentConfig(profile=profile, provider=args.provider, repetitions=args.repetitions)
    config.validate()
    artifact = build_not_run_artifact(
        dataset_path=args.dataset,
        dataset_digest=dataset.digest,
        profile=profile.name,
        provider=args.provider,
        repetitions=args.repetitions,
        config=config.as_dict(),
    )
    ArtifactRecorder(args.output).write(artifact)
    # CLI preparation intentionally has no implicit provider calls.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
