"""Applied encoding comparator for a separate, source-bound T18 experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mnemos.engine import MnemosEngine
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeRegistry
from mnemos.types import Interaction

from .contracts import SessionScenario
from .harness import DEFAULT_PROFILES, ExperimentProfile, MODULES, RunObservation
from .ollama_adapter import (
    SHADOW_LENGTH_MIN_CHARS,
    SHADOW_NOISE_TEXT,
    OllamaEvaluatorAdapter,
    ServiceFactory,
    make_ollama_service_factory,
)

POLICY_VERSION = "applied-length-noise-v1"
POLICY_CONFIG = {
    "version": POLICY_VERSION,
    "minimum_chars": SHADOW_LENGTH_MIN_CHARS,
    "boilerplate_exact_casefold": sorted(SHADOW_NOISE_TEXT),
    "input": "persisted_source_after_write_firewall",
    "boundary": "automatic_extraction_job",
}
LENGTH_PROFILE = ExperimentProfile(
    "length-noise",
    tuple(module for module in MODULES if module != "surprisal_gating"),
    "surprisal_gating",
)
COMPARATOR_PROFILES = (
    DEFAULT_PROFILES[1],
    next(p for p in DEFAULT_PROFILES if p.name == "ablate-surprisal_gating"),
    LENGTH_PROFILE,
)


class LengthNoiseService(LocalMemoryService):
    """Substitute the frozen length/noise rule only at automatic encoding."""

    def __init__(self, engine: MnemosEngine, registry: ScopeRegistry) -> None:
        if engine.config.modules.surprisal:
            raise ValueError("The length/noise comparison must disable surprisal")
        super().__init__(engine, registry)

    async def _automatic_encoding_gate(
        self, interaction: Interaction, existing_facts: list[str] | None = None
    ) -> tuple[str | None, dict[str, Any]]:
        text = interaction.content.strip()
        noise = text.casefold() in SHADOW_NOISE_TEXT
        kept = len(text) >= SHADOW_LENGTH_MIN_CHARS and not noise
        return None if kept else "length_noise_filtered", {
            "policy": POLICY_VERSION,
            "kept": kept,
            "content_chars": len(text),
            "minimum_chars": SHADOW_LENGTH_MIN_CHARS,
            "reason": "exact_boilerplate" if noise else "long_enough" if kept else "too_short",
            "input": "persisted_source_after_write_firewall",
        }


def make_comparator_service_factory(
    db_path: Path, registry: ScopeRegistry, **options: Any
) -> ServiceFactory:
    ordinary = make_ollama_service_factory(db_path, registry, **options)
    length = make_ollama_service_factory(
        db_path, registry, service_type=LengthNoiseService, **options
    )

    def factory(profile: ExperimentProfile, run_key: str | None = None) -> LocalMemoryService:
        if profile not in COMPARATOR_PROFILES:
            raise ValueError("Unknown encoding comparator profile")
        selected = length if profile == LENGTH_PROFILE else ordinary
        return cast(LocalMemoryService, selected(profile, run_key))

    return factory


class EncodingComparatorAdapter(OllamaEvaluatorAdapter):
    """Read applied gate decisions from durable jobs, preserving missing data."""

    def evaluate_session(
        self, session: SessionScenario, profile: ExperimentProfile, repetition: int
    ) -> RunObservation:
        if profile not in COMPARATOR_PROFILES:
            raise ValueError("Unknown encoding comparator profile")
        self._final_row_counts: dict[str, int] | None = None
        observed = super().evaluate_session(session, profile, repetition)
        raw = dict(observed.raw_output)
        details = raw["surprisal_filter_decisions"]
        raw["encoding_policy_decisions"] = details
        raw["final_row_counts"] = self._final_row_counts
        raw["encoding_comparison"] = {
            "policy": POLICY_CONFIG,
            "arm": profile.name,
            "scope": "replacement_encoding_policy_with_other_modules_unchanged",
            "counts": {
                "source_events": len(session.events),
                "automatic_capture_operations": len(details),
                "completed_job_decisions": sum(row["kept"] is not None for row in details),
                "unobserved_decisions": sum(row["kept"] is None for row in details),
            },
        }
        # The superclass calls these fields surprisal decisions. Keep those
        # fields available, but do not relabel length decisions as surprisal.
        raw["surprisal_filter_decisions"] = [
            {**row, "kept": row["kept"] if row["policy"] == "surprisal" else None}
            for row in details
            if row["write_mode"] == "automatic"
        ]
        if profile == LENGTH_PROFILE:
            by_event = {row["event_id"]: row for row in details}
            length_values = tuple(
                by_event.get(event.event_id, {}).get("kept")
                for event in session.events
                if event.lifecycle_operation in {"capture", "noise", "response"}
                and event.write_mode != "none"
            )
            raw["length_filter_policy"] = {
                **POLICY_CONFIG,
                "applied": "durable_extraction_job",
                "source": "completed_job_encoding_diagnostics",
                "events": [row["event_id"] for row in details],
            }
            raw["measurement_notes"] = [
                note
                for note in raw["measurement_notes"]
                if not note.startswith("length_filter_kept is")
            ] + [
                "length_filter_kept reads applied durable job decisions; missing decisions stay null.",
                "Capture/noise/response operations create automatic jobs regardless of fixture write_mode; every such job is listed separately.",
            ]
            observed = replace(observed, length_filter_kept=length_values)
        return replace(observed, raw_output=raw)

    def _close_service(self, service: Any) -> None:
        store = getattr(getattr(service, "engine", None), "store", None)
        if store is not None:
            self._final_row_counts = {
                "facts": len(store.get_all()),
                "episodes": len(store.list_episodic_events(limit=1_000_000)),
                "jobs": len(store.list_jobs(limit=1_000_000)),
            }
        super()._close_service(service)

    @staticmethod
    def _surprisal_filter_decisions(
        service: Any | None, session: SessionScenario, operations: list[dict[str, Any]]
    ) -> tuple[list[bool | None], list[dict[str, Any]]]:
        by_event = {row.get("event_id"): row for row in operations if row.get("event_id")}
        store = getattr(getattr(service, "engine", None), "store", None)
        details: list[dict[str, Any]] = []
        values: list[bool | None] = []
        for event in session.events:
            if event.lifecycle_operation not in {"capture", "noise", "response"}:
                continue
            record = by_event.get(event.event_id, {})
            job_id = record.get("job_id")
            job = store.get_job(job_id) if store is not None and job_id else None
            result = job.result if job is not None and isinstance(job.result, Mapping) else {}
            diagnostics = result.get("encoding_diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                diagnostics = {}
            completed = job is not None and job.status == "completed"
            policy = "unobserved"
            kept: bool | None = None
            if completed and diagnostics.get("policy") == POLICY_VERSION:
                policy = "length_noise"
                value = diagnostics.get("kept")
                kept = value if isinstance(value, bool) else None
            elif completed and diagnostics.get("surprisal_status") == "disabled":
                policy, kept = "ungated", True
            elif completed and diagnostics.get("surprisal_score") is not None:
                policy = "surprisal"
                kept = result.get("decision") != "surprisal_filtered"
            details.append(
                {
                    "event_id": event.event_id,
                    "operation": event.lifecycle_operation,
                    "write_mode": event.write_mode,
                    "job_id": job_id,
                    "durable_event_id": job.payload.get("event_id") if job is not None else None,
                    "job_status": job.status if job is not None else "missing",
                    "decision": result.get("decision"),
                    "diagnostics": dict(diagnostics),
                    "policy": policy,
                    "kept": kept,
                    "source": "durable_job_result",
                    "model_call": (
                        "not_requested"
                        if policy == "length_noise" and kept is False
                        else "not_attributed"
                    ),
                    "token_usage": None,
                }
            )
            if event.write_mode == "automatic":
                values.append(kept if policy == "surprisal" else None)
        return values, details
