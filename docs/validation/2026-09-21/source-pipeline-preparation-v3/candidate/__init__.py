"""Offline contracts and runner for the local-memory held-out evaluation.

The package deliberately does not construct an engine or make provider calls.
Callers supply an adapter at the boundary when they are ready to run a real
model experiment.
"""

from .artifacts import ArtifactRecorder, build_not_run_artifact, file_digest
from .contracts import (
    DATASET_VERSION,
    MINIMUMS,
    FactRecord,
    HeldoutDataset,
    QueryRecord,
    SessionScenario,
    SourceEvent,
    load_dataset,
    validate_dataset,
)
from .harness import (
    DEFAULT_PROFILES,
    ExperimentConfig,
    ExperimentHarness,
    ExperimentProfile,
    RunObservation,
    build_completed_artifact,
    compare_observation_metrics,
    evaluate_with_adapter,
    run_experiment,
)
from .resumption import Checkpoint, load_checkpoint, save_checkpoint
from .scoring import paired_bootstrap_interval, wilson_interval
from .ollama_adapter import OllamaEvaluatorAdapter, make_ollama_service_factory
from .module_challenge import load_module_challenge_dataset

__all__ = [
    "ArtifactRecorder",
    "Checkpoint",
    "DATASET_VERSION",
    "DEFAULT_PROFILES",
    "ExperimentConfig",
    "ExperimentHarness",
    "ExperimentProfile",
    "FactRecord",
    "HeldoutDataset",
    "MINIMUMS",
    "QueryRecord",
    "RunObservation",
    "SessionScenario",
    "SourceEvent",
    "build_not_run_artifact",
    "build_completed_artifact",
    "compare_observation_metrics",
    "evaluate_with_adapter",
    "run_experiment",
    "OllamaEvaluatorAdapter",
    "make_ollama_service_factory",
    "load_module_challenge_dataset",
    "file_digest",
    "load_checkpoint",
    "load_dataset",
    "paired_bootstrap_interval",
    "save_checkpoint",
    "validate_dataset",
    "wilson_interval",
]
