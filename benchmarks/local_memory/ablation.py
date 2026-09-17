"""Named profile matrix for T18 ablation runs.

The execution implementation stays in :mod:`benchmarks.local_memory.harness`;
this module gives experiment scripts a small stable import surface.
"""

from .harness import (
    DEFAULT_PROFILES,
    MODULES,
    ExperimentConfig,
    ExperimentHarness,
    ExperimentProfile,
    RunObservation,
    compare_observation_metrics,
)

__all__ = [
    "DEFAULT_PROFILES",
    "MODULES",
    "ExperimentConfig",
    "ExperimentHarness",
    "ExperimentProfile",
    "RunObservation",
    "compare_observation_metrics",
]
