"""Service adapter for bounded Ollama-backed local-memory evaluations.

The adapter owns orchestration and measurement only. A caller supplies a
service factory so engine configuration, module switches, scope registration,
and the SQLite path remain explicit integration choices. No facts are written
directly by this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from .contracts import FactRecord, QueryRecord, SessionScenario, SourceEvent, StatePair
from .harness import ExperimentProfile, RunObservation
from mnemos.retrieval import score_candidate
from mnemos.scopes import ScopeContext, scope_allows
from mnemos.types import Interaction, MemoryChunk
from mnemos.runtime import embedding_task_contract
from mnemos.utils.embeddings import EmbeddingProvider, PrefixEmbeddingProvider
from mnemos.utils.embeddings import embed_text_async
from mnemos.utils.llm import LLMProvider, _validate_ollama_think

ServiceFactory = Callable[..., Any]
OllamaThink = bool | Literal["low", "medium", "high", "max"] | None


def parse_ollama_think(value: str) -> OllamaThink:
    """Parse the CLI spelling used by Ollama's ``think`` option."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return _validate_ollama_think(normalized)


def add_ollama_think(options: dict[str, Any], think: OllamaThink) -> dict[str, Any]:
    """Add an explicit thinking mode without changing the default shape."""
    selected = _validate_ollama_think(think)
    if selected is not None:
        options["think"] = selected
    return options


def ollama_options_sha256(options: Mapping[str, Any]) -> str:
    """Hash the exact serialized generation options used by an evaluation."""
    return hashlib.sha256(
        json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# These values are frozen protocol metadata for the comparison artifact. The
# length/noise rule is a counterfactual shadow measurement because the service
# does not expose a length filter decision. It must never be presented as an
# applied pipeline result.
SHADOW_LENGTH_MIN_CHARS = 50
SHADOW_NOISE_TEXT = frozenset({"ok", "okay", "thanks", "thanks!", "got it"})


class OllamaEvaluatorAdapter:
    """Run ordered dataset events through a real or injected memory service.

    ``service_factory`` is invoked once per evaluation and again at every
    restart event. The factory should return a fresh ``LocalMemoryService``
    over the same temporary SQLite database. Tests can inject a small fake
    service; real runs pass a factory that wires ``OllamaProvider`` and
    ``OllamaEmbeddingProvider``.
    """

    def __init__(
        self,
        service_factory: ServiceFactory,
        *,
        base_url: str = "http://192.168.86.35:11434",
        model: str = "gemma3:4b",
        embedding_model: str = "nomic-embed-text:latest",
        provider: str = "ollama",
        user_profile_id: str = "t18-synthetic-profile",
        top_k: int = 5,
        max_context_tokens: int = 1200,
        real_model: bool = False,
        model_revision: str | None = None,
        embedding_revision: str | None = None,
        model_sha256: str = "",
        embedding_sha256: str = "",
        options_sha256: str = "",
        code_sha256: str = "",
        unsupported_modules: tuple[str, ...] = (),
        model_provenance: Mapping[str, Any] | None = None,
        embedding_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.service_factory = service_factory
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model
        self.provider = provider
        self.user_profile_id = user_profile_id
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.real_model = real_model
        self.model_revision = model_revision
        self.embedding_revision = embedding_revision
        self.model_sha256 = model_sha256
        self.embedding_sha256 = embedding_sha256
        self.options_sha256 = options_sha256
        self.code_sha256 = code_sha256
        self.unsupported_modules = tuple(unsupported_modules)
        self.model_provenance = dict(model_provenance or {})
        self.embedding_provenance = dict(embedding_provenance or {})

    def evaluate_session(
        self, session: SessionScenario, profile: ExperimentProfile, repetition: int
    ) -> RunObservation:
        started = time.perf_counter()
        operations: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        unsupported: list[str] = []
        lifecycle_summary: dict[str, dict[str, int]] = {
            operation: {
                "attempted": 0,
                "committed": 0,
                "rejected": 0,
                "mapping_invalid": 0,
                "missing_target": 0,
            }
            for operation in ("correct", "forget")
        }

        def note_lifecycle(operation: str, status: str) -> None:
            summary = lifecycle_summary.get(operation)
            if summary is None:
                return
            summary["attempted"] += 1
            bucket = {
                "target_mapping_invalid": "mapping_invalid",
                "missing_accepted_target": "missing_target",
                "committed": "committed",
            }.get(status, "rejected")
            summary[bucket] += 1

        retrievals: dict[str, Mapping[str, Any] | None] = {}
        service: Any | None = None
        cold_latency_ms: float | None = None
        bytes_before: int | None = None

        try:
            cold_started = time.perf_counter()
            run_key = f"{session.session_id}-{profile.name}-r{repetition}"
            service = self._new_service(profile, run_key)
            cold_latency_ms = (time.perf_counter() - cold_started) * 1000
            bytes_before = self._store_bytes(service)
        except Exception as exc:
            errors.append(self._error("start", exc))

        context = self._context(session, profile, repetition)
        for event in session.events:
            operation = event.lifecycle_operation
            if operation == "restart":
                if service is None:
                    continue
                try:
                    self._close_service(service)
                    restart_started = time.perf_counter()
                    service = self._new_service(profile, run_key)
                    cold_latency_ms = (time.perf_counter() - restart_started) * 1000
                    operations.append({"operation": "restart", "status": "completed"})
                except Exception as exc:
                    errors.append(self._error("restart", exc, event.event_id))
                continue
            if service is None:
                errors.append({"operation": operation, "error": "service_unavailable"})
                note_lifecycle(operation, "rejected")
                continue
            try:
                if operation in {"correct", "forget"}:
                    # Resolve targets only after prior captures have completed
                    # their durable jobs. This still uses the service API and
                    # prevents a guessed or gold-only target ID.
                    self._run_maintenance(service, operations, errors, before=operation)
                result = self._event_operation(
                    service, event, session, context, profile, repetition
                )
                operation_record: dict[str, Any] = {
                    "operation": operation,
                    "event_id": event.event_id,
                    "status": (
                        str(result.get("status", "returned"))
                        if isinstance(result, Mapping)
                        else "returned"
                    ),
                }
                if isinstance(result, Mapping):
                    for field in (
                        "job_id",
                        "processing_status",
                        "fact_id",
                        "forgotten",
                        "expected_version",
                    ):
                        value = result.get(field)
                        if value is not None:
                            operation_record[field] = value
                operations.append(operation_record)
                if operation in {"correct", "forget"}:
                    result_status = (
                        str(result.get("status", "returned"))
                        if isinstance(result, Mapping)
                        else "returned"
                    )
                    note_lifecycle(operation, result_status)
                    if result_status != "committed":
                        if result_status in {"unsupported", "rejected"}:
                            unsupported.append(operation)
                        errors.append(
                            {
                                "operation": operation,
                                "event_id": event.event_id,
                                "error": "lifecycle_not_committed",
                                "status": result_status,
                            }
                        )
            except _TargetMappingError as exc:
                note_lifecycle(operation, exc.status)
                operations.append(
                    {
                        "operation": operation,
                        "event_id": event.event_id,
                        "status": exc.status,
                    }
                )
                errors.append(
                    {
                        "operation": operation,
                        "event_id": event.event_id,
                        "error": exc.status,
                        "reason": exc.reason,
                    }
                )
            except _UnsupportedOperation as exc:
                note_lifecycle(operation, "rejected")
                unsupported.append(operation)
                operations.append(
                    {"operation": operation, "event_id": event.event_id, "status": "unsupported"}
                )
                errors.append(
                    {"operation": operation, "event_id": event.event_id, "error": str(exc)}
                )
            except Exception as exc:
                note_lifecycle(operation, "rejected")
                errors.append(self._error(operation, exc, event.event_id))

        if service is not None:
            self._run_maintenance(service, operations, errors)

            for query in session.queries:
                retrievals[query.query_id] = self._run_retrieval(
                    service,
                    query.text,
                    self._target_context(context, query),
                    profile,
                    operations,
                    errors,
                    query.query_id,
                )
            for pair in session.state_pairs:
                retrievals[pair.pair_id + ":semantic"] = self._run_retrieval(
                    service,
                    pair.semantic_query,
                    self._target_context(context, pair),
                    profile,
                    operations,
                    errors,
                    pair.pair_id + ":semantic",
                )
                retrievals[pair.pair_id + ":state"] = self._run_retrieval(
                    service,
                    pair.state_query,
                    self._target_context(context, pair),
                    profile,
                    operations,
                    errors,
                    pair.pair_id + ":state",
                )

        extraction_values, extraction_details = self._extraction_judgements(service, session)
        profile_gaps = [
            module
            for module in self.unsupported_modules
            if module in profile.enabled_modules or profile.disabled_module == module
        ]
        for module in profile_gaps:
            errors.append(
                {
                    "operation": "profile",
                    "error": "module_behavior_unavailable",
                    "module": module,
                }
            )
        critical_values, retention_details = self._critical_retention(service, session)
        critical_retrieval_values = [
            self._fact_retrieved(fact, session, retrievals)
            for fact in session.facts
            if fact.critical
        ]
        answerable_values = [
            self._answerable_hit(query, retrievals.get(query.query_id), session)
            for query in session.queries
            if query.kind == "answerable"
        ]
        no_answer_values = [
            self._no_answer_correct(retrievals.get(query.query_id))
            for query in session.queries
            if query.kind == "no_answer"
        ]
        semantic_values = [
            self._state_hit(pair, retrievals.get(pair.pair_id + ":semantic"), semantic=True)
            for pair in session.state_pairs
        ]
        state_values = [
            self._state_hit(pair, retrievals.get(pair.pair_id + ":state"), semantic=False)
            for pair in session.state_pairs
        ]
        stale_count, retrieved_count = self._stale_counts(session, retrievals)
        controlled_state_details: list[dict[str, Any]] = []
        if service is not None and session.state_pairs:
            try:
                controlled_state_details = self._controlled_state_pairs(
                    service, session, context, profile
                )
            except Exception as exc:
                errors.append(self._error("controlled_state_pairs", exc))
        surprisal_values, surprisal_details = self._surprisal_filter_decisions(
            service, session, operations
        )
        shadow_events = [
            event
            for event in session.events
            if event.lifecycle_operation in {"capture", "noise", "response"}
            and event.write_mode != "none"
        ]
        length_values = [self._shadow_length_keep(event) for event in shadow_events]
        bytes_after = self._store_bytes(service)
        try:
            self._close_service(service)
        except Exception as exc:
            errors.append(self._error("close", exc))
        execution_status = "complete" if not errors else "partial"
        if service is None:
            execution_status = "missing"
        embedding_verified = bool(
            self.real_model
            and self.model_sha256
            and self.embedding_sha256
            and self.options_sha256
            and self.code_sha256
            and self.model_revision
            and self.embedding_revision
            and self.model_provenance
            and self.embedding_provenance
            and self._provenance_complete(self.model_provenance, self.model)
            and self._provenance_complete(self.embedding_provenance, self.embedding_model)
        )
        cognition_observed = self._trace_has_cognition_calls(service)
        verified = bool(embedding_verified and cognition_observed)
        return RunObservation(
            session_id=session.session_id,
            repetition=repetition,
            critical_retention=tuple(critical_values),
            critical_retrieval=tuple(critical_retrieval_values),
            supported_precision=tuple(extraction_values),
            answerable_hits=tuple(answerable_values),
            no_answer_correct=tuple(no_answer_values),
            state_pair_semantic_hits=tuple(semantic_values),
            state_pair_state_hits=tuple(state_values),
            surprisal_filter_kept=tuple(surprisal_values),
            length_filter_kept=tuple(length_values),
            stale_fact_count=stale_count,
            retrieved_fact_count=retrieved_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            cold_latency_ms=cold_latency_ms,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            provider=self.provider,
            model_revision=self.model_revision,
            embedding_revision=self.embedding_revision,
            execution_status=execution_status,
            adapter_kind=(
                "real_model"
                if self.real_model and cognition_observed
                else (
                    "real_embedding_observed"
                    if self.real_model and self._trace_has_embedding_calls(service)
                    else "test"
                )
            ),
            verified_embedding_provenance=embedding_verified,
            verified_model_provenance=verified,
            raw_output={
                "adapter": "ollama",
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "embedding_model": self.embedding_model,
                "profile": profile.as_dict(),
                "module_switches": {
                    "enabled": list(profile.enabled_modules),
                    "disabled": profile.disabled_module,
                    "source": "service_factory",
                },
                "operations": operations,
                "lifecycle_summary": lifecycle_summary,
                "errors": errors,
                "unsupported_operations": sorted(set(unsupported)),
                "unsupported_modules": profile_gaps,
                "extraction_judgements": extraction_details,
                "critical_retention_judgements": retention_details,
                "surprisal_filter_decisions": surprisal_details,
                "length_filter_policy": {
                    "minimum_chars": SHADOW_LENGTH_MIN_CHARS,
                    "boilerplate_exact_casefold": sorted(SHADOW_NOISE_TEXT),
                    "applied": "counterfactual_shadow_only",
                    "service_decision": "not_exposed",
                    "events": [event.event_id for event in shadow_events],
                },
                "controlled_state_pairs": controlled_state_details,
                "measurement_notes": [
                    "Missing retrievals remain null and are excluded from interval numerators.",
                    "Correction and forget use runtime fact IDs resolved from the service store.",
                    "Provider token counters are recorded when returned; billing cost is not inferred.",
                    "latency_ms is end-to-end session wall clock; cold_latency_ms covers service construction only.",
                    "Extraction precision is an exact frozen text, scope, and source-evidence proxy; unmatched text, including paraphrases, counts as false. It does not measure semantic factual precision.",
                    "State pairs use paired natural-language queries; no same-text embedding control is claimed.",
                    "surprisal_filter_kept is derived from durable job encoding_diagnostics and remains null when a job result is unavailable.",
                    "length_filter_kept is a frozen counterfactual shadow policy, not an applied service filter decision.",
                    "stale_fact_rate is a returned-row substring proxy, not a unique-fact or query-level accuracy measure.",
                ],
                "provenance": {
                    "model_sha256": self.model_sha256 or None,
                    "embedding_sha256": self.embedding_sha256 or None,
                    "options_sha256": self.options_sha256 or None,
                    "code_sha256": self.code_sha256 or None,
                    "model_metadata": self.model_provenance,
                    "embedding_metadata": self.embedding_provenance,
                    "embedding_verified": embedding_verified,
                    "cognition_observed": cognition_observed,
                    "verified": verified,
                },
                "provider_trace": self._provider_trace(service),
            },
        )

    def _event_operation(
        self,
        service: Any,
        event: SourceEvent,
        session: SessionScenario,
        context: Mapping[str, Any],
        profile: ExperimentProfile,
        repetition: int,
    ) -> Mapping[str, Any]:
        event_id = self._runtime_event_id(event.event_id, profile, repetition)
        if event.lifecycle_operation in {"capture", "noise", "response"}:
            event_context = self._target_context(context, event)
            return self._dispatch(
                service,
                "capture",
                {
                    "event_id": event_id,
                    "context": dict(event_context),
                    "occurred_at": event.occurred_at,
                    "role": event.role,
                    "content": event.content,
                    "source_kind": event.source_kind,
                },
            )
        if event.lifecycle_operation == "remember":
            return self._dispatch(
                service,
                "remember",
                {
                    "event_id": event_id,
                    "context": dict(context),
                    "content": event.content,
                    "scope": self._scope_for_event(event, session),
                },
            )
        if event.lifecycle_operation in {"correct", "forget"}:
            action_context = self._target_context(context, event)
            target = self._resolve_runtime_target(
                service,
                event,
                session,
                context=context,
                profile=profile,
                repetition=repetition,
            )
            if target is None:
                raise _UnsupportedOperation(
                    f"{event.lifecycle_operation} target is not resolvable from runtime facts"
                )
            payload = {
                "event_id": event_id,
                "context": dict(action_context),
                # v3 supplies the replacement separately from the action's
                # natural-language instruction. Keep the v2 payload behavior
                # when loading an older fixture.
                "content": event.replacement_content or event.content,
                "fact_id": target["id"],
                "expected_version": target.get("version", 1),
            }
            result = dict(self._dispatch(service, event.lifecycle_operation, payload))
            # Services may return only the job envelope for correction. Keep
            # the resolved identity and CAS version in the raw operation trace
            # so a lifecycle result remains auditable after the service closes.
            result.setdefault("fact_id", target["id"])
            result.setdefault("expected_version", target.get("version", 1))
            return result
        raise _UnsupportedOperation(f"lifecycle operation {event.lifecycle_operation!r}")

    def _run_retrieval(
        self,
        service: Any,
        query: str,
        context: Mapping[str, Any],
        profile: ExperimentProfile,
        operations: list[dict[str, Any]],
        errors: list[dict[str, Any]],
        query_id: str,
    ) -> Mapping[str, Any] | None:
        try:
            raw_result = self._dispatch(
                service,
                "retrieve",
                {
                    "query": query,
                    "context": dict(context),
                    "allowed_scopes": ["project", "global"],
                    "top_k": self.top_k,
                    "max_context_tokens": self.max_context_tokens,
                    "profile": "biomimetic" if profile.name != "baseline" else "baseline",
                },
            )
            result, scope_validation = self._validate_retrieval_scope(
                service, raw_result, context, errors, query_id
            )
            operation_record: dict[str, Any] = {
                "operation": "retrieve",
                "query_id": query_id,
                "status": scope_validation["status"],
                "scope_validation": scope_validation,
            }
            if isinstance(raw_result.get("facts"), list):
                # Keep the host response auditable while exposing only the
                # validated candidates to metric helpers and later callers.
                operation_record["observed_facts"] = raw_result["facts"]
            if isinstance(result, Mapping) and isinstance(result.get("facts"), list):
                # Preserve the observed candidate order and production score
                # breakdown, including graph energy, for module diagnostics.
                operation_record["facts"] = result["facts"]
            operations.append(operation_record)
            return result
        except Exception as exc:
            errors.append(self._error("retrieve", exc, query_id))
            return None

    def _validate_retrieval_scope(
        self,
        service: Any,
        result: Mapping[str, Any],
        context: Mapping[str, Any],
        errors: list[dict[str, Any]],
        query_id: str,
    ) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
        """Filter foreign candidates before any retrieval metric sees them.

        Production retrievals must return the profile and scope metadata that
        made a candidate eligible. Injected fixtures may omit that metadata so
        older offline tests remain usable, but the trace labels those rows as
        unverified instead of claiming a scope-safe hit.
        """
        raw_facts = result.get("facts")
        if not isinstance(raw_facts, list):
            errors.append(
                {
                    "operation": "retrieve",
                    "query_id": query_id,
                    "error": "scope_validation_failed",
                    "foreign_count": 0,
                    "missing_metadata_count": 0,
                    "malformed_count": 1,
                }
            )
            return None, {
                "status": "malformed_result",
                "returned_count": 0,
                "accepted_count": 0,
                "verified_count": 0,
                "missing_metadata_count": 0,
                "foreign_count": 0,
                "malformed_count": 1,
            }
        allowed_scopes = ("project", "global")
        query_context = ScopeContext.model_validate(context)
        filtered: list[Any] = []
        missing_metadata = 0
        foreign = 0
        malformed = 0
        verified = 0
        required = ("user_profile_id", "scope", "scope_id")
        store = getattr(getattr(service, "engine", None), "store", None)
        for item in raw_facts:
            if not isinstance(item, Mapping):
                malformed += 1
                continue
            metadata = item.get("metadata")
            # Service responses currently expose scope fields at the fact
            # level. Accepting a nested metadata mapping keeps this boundary
            # compatible with lower-level adapters without guessing identity.
            if not isinstance(metadata, Mapping):
                metadata = item
            # The public service response intentionally omits profile identity.
            # Resolve it from the authoritative store row before validating;
            # only fixture adapters without a store may use response metadata.
            item_id = item.get("id")
            get_chunk = getattr(store, "get", None)
            stored_chunk = get_chunk(str(item_id)) if callable(get_chunk) and item_id else None
            stored_metadata = getattr(stored_chunk, "metadata", None)
            if callable(get_chunk) and (not item_id or stored_chunk is None):
                missing_metadata += 1
                if not self.real_model:
                    filtered.append(item)
                continue
            if isinstance(stored_metadata, Mapping):
                response_scope = (item.get("scope"), item.get("scope_id"))
                stored_scope = (
                    stored_metadata.get("scope"),
                    stored_metadata.get("scope_id"),
                )
                if response_scope != stored_scope:
                    foreign += 1
                    continue
                metadata = stored_metadata
            if any(field not in metadata for field in required):
                missing_metadata += 1
                if not self.real_model:
                    filtered.append(item)
                continue
            if not scope_allows(metadata, query_context, allowed_scopes):
                foreign += 1
                continue
            verified += 1
            filtered.append(item)
        if foreign or (missing_metadata and self.real_model) or malformed:
            errors.append(
                {
                    "operation": "retrieve",
                    "query_id": query_id,
                    "error": "scope_validation_failed",
                    "foreign_count": foreign,
                    "missing_metadata_count": missing_metadata,
                    "malformed_count": malformed,
                }
            )
        if foreign:
            status = "scope_violation"
        elif missing_metadata:
            status = "not_scope_verified"
        elif malformed:
            status = "malformed_result"
        elif raw_facts:
            status = "verified"
        else:
            status = "no_returned_facts"
        sanitized = dict(result)
        sanitized["facts"] = filtered
        return sanitized, {
            "status": status,
            "returned_count": len(raw_facts),
            "accepted_count": len(filtered),
            "verified_count": verified,
            "missing_metadata_count": missing_metadata,
            "foreign_count": foreign,
            "malformed_count": malformed,
        }

    def _controlled_state_pairs(
        self,
        service: Any,
        session: SessionScenario,
        context: Mapping[str, Any],
        profile: ExperimentProfile,
    ) -> list[dict[str, Any]]:
        """Measure paired affective state wording with one embedding/pool.

        This supplemental diagnostic deliberately bypasses the full retrieval
        orchestration. It embeds each pair's semantic query once, builds one
        scoped candidate pool, obtains the production classifier output for
        both wordings, and calls the production ``score_candidate`` function.
        Full pipeline query metrics remain separate in ``state_pair_*``.
        """
        engine = getattr(service, "engine", None)
        router = getattr(engine, "affective_router", None)
        embedder = getattr(engine, "embedder", None)
        store: Any = getattr(engine, "store", None)
        if router is None or embedder is None or not callable(getattr(store, "retrieve", None)):
            return []
        config = getattr(engine, "config", None)
        modules = getattr(config, "modules", None)
        affective_module_enabled = (
            bool(modules.get("affective", True))
            if isinstance(modules, Mapping)
            else bool(getattr(modules, "affective", True))
        )
        affective_enabled = bool(
            profile.name != "baseline"
            and "affective_routing" in profile.enabled_modules
            and affective_module_enabled
        )
        similarity_weight = float(
            getattr(getattr(config, "affective", None), "weight_similarity", 0.7)
        )
        state_weight = float(getattr(getattr(config, "affective", None), "weight_state", 0.3))
        rows: list[dict[str, Any]] = []
        for pair in session.state_pairs:
            pair_context_data = self._target_context(context, pair)
            pair_context = ScopeContext.model_validate(pair_context_data)
            query_embedding = _run_awaitable(
                embed_text_async(embedder, pair.semantic_query, role="query")
            )

            def permitted(chunk: MemoryChunk) -> bool:
                return scope_allows(
                    chunk.metadata, pair_context, ("project", "workspace", "global")
                )

            setattr(
                permitted,
                "_mnemos_scope_filter",
                {
                    "user_profile_id": pair_context.user_profile_id,
                    "project_id": pair_context.project_id,
                    "workspace_id": pair_context.workspace_id,
                    "allowed_scopes": ["project", "workspace", "global"],
                },
            )
            candidates = list(
                store.retrieve(
                    query_embedding,
                    top_k=max(self.top_k * 3, 20),
                    filter_fn=permitted,
                )
            )
            direct_seed_ids = [chunk.id for chunk in candidates]
            states: dict[str, Any] = {}
            for label, wording in (
                ("semantic", pair.semantic_query),
                ("state", pair.state_query),
            ):
                state = _run_awaitable(
                    router.classify_state(
                        Interaction(
                            role="user",
                            content=wording,
                            metadata=dict(pair_context_data),
                        )
                    )
                )
                states[label] = state
            rankings: dict[str, list[str]] = {}
            score_rows: dict[str, list[dict[str, Any]]] = {}
            for label, state in states.items():
                scored = [
                    (
                        score_candidate(
                            chunk,
                            query_embedding,
                            current_state=state,
                            affective=affective_enabled,
                            similarity_weight=similarity_weight,
                            state_weight=state_weight,
                        ),
                        chunk,
                    )
                    for chunk in candidates
                ]
                scored.sort(key=lambda item: (-item[0].total, item[1].id))
                rankings[label] = [chunk.id for _, chunk in scored]
                score_rows[label] = [
                    {
                        "fact_id": chunk.id,
                        "content": chunk.content,
                        "score": breakdown.model_dump(mode="json"),
                    }
                    for breakdown, chunk in scored
                ]
            semantic_rank = rankings.get("semantic", [])
            state_rank = rankings.get("state", [])
            rank_delta = {
                fact_id: state_rank.index(fact_id) - semantic_rank.index(fact_id)
                for fact_id in semantic_rank
                if fact_id in state_rank
            }
            matched = pair.matched_content.casefold()
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "candidate_count": len(candidates),
                    "candidate_fact_ids": semantic_rank,
                    "direct_vector_seed_fact_ids_observed": direct_seed_ids,
                    "declared_candidate_fact_ids": list(pair.candidate_fact_ids),
                    "declared_direct_vector_seed_fact_ids": list(pair.direct_vector_seed_fact_ids),
                    "declared_graph_target_fact_ids": list(pair.graph_target_fact_ids),
                    "classifier_states": {
                        label: state.model_dump(mode="json") if state is not None else None
                        for label, state in states.items()
                    },
                    "affective_scoring_applied": affective_enabled,
                    "rankings": rankings,
                    "rank_delta_state_minus_semantic": rank_delta,
                    "top1_changed": bool(
                        semantic_rank and state_rank and semantic_rank[0] != state_rank[0]
                    ),
                    "semantic_fidelity": {
                        "semantic_top_k": any(
                            row["content"].strip().casefold() == matched
                            for row in score_rows["semantic"][: self.top_k]
                        ),
                        "state_top_k": any(
                            row["content"].strip().casefold() == matched
                            for row in score_rows["state"][: self.top_k]
                        ),
                        "judgement": "exact_frozen_text_only",
                    },
                    "scores": score_rows,
                    "measurement": "same_semantic_embedding_and_scoped_candidate_pool",
                    "graph_target_observation": {
                        "target_ranks_in_direct_seed": {
                            fact_id: direct_seed_ids.index(fact_id) + 1
                            for fact_id in pair.graph_target_fact_ids
                            if fact_id in direct_seed_ids
                        },
                        "targets_in_direct_seed": [
                            fact_id
                            for fact_id in pair.graph_target_fact_ids
                            if fact_id in direct_seed_ids
                        ],
                        "status": (
                            "not_discriminating_target_already_in_direct_seed"
                            if any(
                                fact_id in direct_seed_ids for fact_id in pair.graph_target_fact_ids
                            )
                            else "not_observed_graph_traversal"
                        ),
                    },
                }
            )
        return rows

    def _new_service(self, profile: ExperimentProfile, run_key: str) -> Any:
        try:
            signature = inspect.signature(self.service_factory)
            accepts_run_key = len(
                [
                    parameter
                    for parameter in signature.parameters.values()
                    if parameter.kind
                    in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
                ]
            ) >= 2 or any(
                parameter.kind == parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_run_key = True
        return (
            self.service_factory(profile, run_key)
            if accepts_run_key
            else self.service_factory(profile)
        )

    @staticmethod
    def _provider_trace(service: Any | None) -> Mapping[str, Any]:
        trace = getattr(service, "_benchmark_provider_trace", {})
        return dict(trace) if isinstance(trace, Mapping) else {}

    def _trace_has_provider_calls(self, service: Any | None) -> bool:
        trace = self._provider_trace(service)
        return bool(trace.get("predict_calls") or trace.get("embed_calls"))

    def _trace_has_cognition_calls(self, service: Any | None) -> bool:
        trace = self._provider_trace(service)
        return bool(trace.get("predict_calls") or trace.get("classify_calls"))

    def _trace_has_embedding_calls(self, service: Any | None) -> bool:
        trace = self._provider_trace(service)
        return bool(trace.get("embed_calls"))

    @staticmethod
    def _shadow_length_keep(event: SourceEvent) -> bool:
        """Return the frozen counterfactual length/noise policy decision."""
        text = event.content.strip()
        return len(text) >= SHADOW_LENGTH_MIN_CHARS and text.casefold() not in SHADOW_NOISE_TEXT

    @staticmethod
    def _surprisal_filter_decisions(
        service: Any | None,
        session: SessionScenario,
        operations: list[dict[str, Any]],
    ) -> tuple[list[bool | None], list[dict[str, Any]]]:
        """Read actual extraction decisions from durable job results.

        A missing job, pending job, failed job, or result without a decision is
        retained as an unjudged null. The adapter never infers a decision from
        gold facts or from whether a chunk happened to be retrieved.
        """
        by_event = {
            str(record.get("event_id")): record
            for record in operations
            if record.get("operation") in {"capture", "noise", "response"}
            and record.get("event_id")
        }
        store = getattr(getattr(service, "engine", None), "store", None)
        get_job = getattr(store, "get_job", None)
        values: list[bool | None] = []
        details: list[dict[str, Any]] = []
        for event in session.events:
            if event.lifecycle_operation not in {"capture", "noise", "response"}:
                continue
            if event.write_mode != "automatic":
                continue
            record = by_event.get(event.event_id)
            job_id = record.get("job_id") if record else None
            job = get_job(job_id) if callable(get_job) and job_id else None
            status = str(getattr(job, "status", "missing")).casefold() if job else "missing"
            result = getattr(job, "result", None) if job else None
            decision = result.get("decision") if isinstance(result, Mapping) else None
            value: bool | None
            if status == "completed" and isinstance(decision, str):
                value = decision != "surprisal_filtered"
            else:
                value = None
            values.append(value)
            details.append(
                {
                    "event_id": event.event_id,
                    "job_id": job_id,
                    "job_status": status,
                    "decision": decision,
                    "kept": value,
                    "source": "durable_job_result",
                }
            )
        return values, details

    @staticmethod
    def _provenance_complete(metadata: Mapping[str, Any], model: str) -> bool:
        digest = metadata.get("digest") or metadata.get("sha256") or metadata.get("digest_sha256")
        observed_model = metadata.get("name") or metadata.get("model")
        return (
            observed_model == model
            and isinstance(digest, str)
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", digest))
        )

    def _run_maintenance(
        self,
        service: Any,
        operations: list[dict[str, Any]],
        errors: list[dict[str, Any]],
        *,
        before: str | None = None,
    ) -> Mapping[str, Any] | None:
        try:
            result = self._dispatch(service, "maintenance", {})
            store = getattr(getattr(service, "engine", None), "store", None)
            list_jobs = getattr(store, "list_jobs", None)
            bad_statuses: list[str] = []
            job_details: list[dict[str, Any]] = []
            if callable(list_jobs):
                jobs = list_jobs()
                for job in jobs:
                    job_details.append(
                        {
                            "id": str(getattr(job, "id", "")),
                            "kind": str(getattr(job, "kind", "")),
                            "status": str(getattr(job, "status", "")),
                            "attempt_count": getattr(job, "attempt_count", None),
                            "last_error": getattr(job, "last_error", None),
                        }
                    )
                bad_statuses = sorted(
                    {
                        str(getattr(job, "status", "unknown"))
                        for job in jobs
                        if getattr(job, "status", "") != "completed"
                    }
                )
            operation: dict[str, Any] = {
                "operation": "maintenance",
                "status": "completed" if not bad_statuses else "incomplete",
                "pending_or_failed_statuses": bad_statuses,
                "jobs": job_details,
            }
            if before:
                operation["before"] = before
            operations.append(operation)
            if bad_statuses:
                errors.append(
                    {
                        "operation": "maintenance",
                        "error": "jobs_not_completed",
                        "statuses": bad_statuses,
                        "jobs": job_details,
                    }
                )
            return result
        except Exception as exc:
            errors.append(self._error("maintenance", exc))
            return None

    @staticmethod
    def _dispatch(service: Any, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        result = service.dispatch(operation, dict(payload))
        if inspect.isawaitable(result):
            result = _run_awaitable(result)
        if isinstance(result, Mapping):
            return result
        raise TypeError(f"service dispatch {operation!r} returned a non-mapping result")

    @staticmethod
    def _runtime_event_id(event_id: str, profile: ExperimentProfile, repetition: int) -> str:
        return f"{event_id}-{profile.name}-r{repetition}"

    def _resolve_runtime_target(
        self,
        service: Any,
        event: SourceEvent,
        session: SessionScenario,
        *,
        context: Mapping[str, Any] | None = None,
        profile: ExperimentProfile | None = None,
        repetition: int | None = None,
    ) -> Mapping[str, Any] | None:
        """Resolve a lifecycle target from durable evidence.

        v3 actions carry the identity of an earlier user source event. That
        identity is resolved through the service's persisted episodic event and
        fact evidence, then checked against the requested scope. Content and
        gold labels are deliberately not used for this path. The optional
        arguments keep the old v2 test seam available for compatibility.
        """
        strict = any(
            value is not None
            for value in (
                event.target_source_event_id,
                event.target_fact_id,
                event.replacement_content,
            )
        )
        if strict:
            return self._resolve_v3_runtime_target(
                service,
                event,
                session,
                context=context,
                profile=profile,
                repetition=repetition,
            )

        return self._resolve_v2_runtime_target(service, event, session)

    def _resolve_v3_runtime_target(
        self,
        service: Any,
        event: SourceEvent,
        session: SessionScenario,
        *,
        context: Mapping[str, Any] | None,
        profile: ExperimentProfile | None,
        repetition: int | None,
    ) -> Mapping[str, Any]:
        if (
            event.target_source_event_id is None
            or event.target_fact_id is None
            or (event.lifecycle_operation == "correct" and event.replacement_content is None)
            or (event.lifecycle_operation == "forget" and event.replacement_content is not None)
        ):
            raise _TargetMappingError(
                "target_mapping_invalid", "incomplete lifecycle target metadata"
            )

        target_fact = next(
            (fact for fact in session.facts if fact.fact_id == event.target_fact_id), None
        )
        if target_fact is None:
            raise _TargetMappingError("target_mapping_invalid", "declared target fact is unknown")
        target_event = next(
            (item for item in session.events if item.event_id == event.target_source_event_id),
            None,
        )
        if target_event is None or target_event.lifecycle_operation not in {"capture", "remember"}:
            raise _TargetMappingError(
                "target_mapping_invalid", "declared target source event is unknown"
            )
        if target_event.role != "user" or target_event.write_mode == "none":
            raise _TargetMappingError(
                "target_mapping_invalid", "target source event is not a user memory event"
            )
        try:
            target_occurred_at = datetime.fromisoformat(
                target_event.occurred_at.replace("Z", "+00:00")
            )
            action_occurred_at = datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise _TargetMappingError(
                "target_mapping_invalid", "lifecycle timestamps are not valid ISO-8601"
            ) from exc
        if target_occurred_at >= action_occurred_at:
            raise _TargetMappingError(
                "target_mapping_invalid", "target source event does not precede action"
            )
        if target_event.evidence_fact_ids.count(event.target_fact_id) != 1:
            raise _TargetMappingError(
                "target_mapping_invalid", "target source event does not uniquely evidence fact"
            )

        store = getattr(getattr(service, "engine", None), "store", None)
        get_all = getattr(store, "get_all", None)
        get_event = getattr(store, "get_episodic_event", None)
        list_evidence = getattr(store, "list_fact_evidence", None)
        if not callable(get_all) or not callable(get_event):
            raise _TargetMappingError(
                "missing_accepted_target", "durable source evidence is unavailable"
            )
        if context is None or profile is None or repetition is None:
            raise _TargetMappingError(
                "target_mapping_invalid", "runtime target context is unavailable"
            )

        target_context = self._target_context(context, event)
        for field in ("user_profile_id", "project_id", "workspace_id"):
            declared = getattr(target_event, "target_" + field)
            if declared is not None and declared != target_context.get(field):
                raise _TargetMappingError(
                    "target_mapping_invalid", "target source event context does not match action"
                )
        if target_fact.evidence_event_ids.count(event.target_source_event_id) != 1:
            raise _TargetMappingError(
                "target_mapping_invalid", "target fact does not uniquely cite source event"
            )
        runtime_source_id = self._runtime_event_id(
            event.target_source_event_id, profile, repetition
        )
        durable_source_id = self._durable_event_id(target_context, runtime_source_id)
        source_event = get_event(durable_source_id)
        if source_event is None:
            raise _TargetMappingError(
                "missing_accepted_target", "accepted source event is unavailable"
            )

        expected_identity = {
            "user_profile_id": target_context.get("user_profile_id"),
            "project_id": target_context.get("project_id"),
            "workspace_id": target_context.get("workspace_id"),
        }
        source_metadata = getattr(source_event, "metadata", {})
        if any(source_metadata.get(key) != value for key, value in expected_identity.items()):
            raise _TargetMappingError("target_mapping_invalid", "source event scope does not match")

        if target_fact.scope == "project":
            expected_scope_id = target_context.get("project_id")
        elif target_fact.scope == "workspace":
            expected_scope_id = target_context.get("workspace_id")
        elif target_fact.scope == "global":
            expected_scope_id = None
        else:
            raise _TargetMappingError("target_mapping_invalid", "target fact has an invalid scope")

        candidates: list[Mapping[str, Any]] = []
        invalid_scope = False
        for chunk in get_all():
            metadata = getattr(chunk, "metadata", {})
            if any(metadata.get(key) != value for key, value in expected_identity.items()):
                continue
            evidence_ids = metadata.get("source_event_ids") or ()
            if not isinstance(evidence_ids, (list, tuple, set, frozenset)):
                evidence_ids = (evidence_ids,)
            has_metadata_evidence = durable_source_id in evidence_ids
            has_table_evidence = False
            if callable(list_evidence):
                has_table_evidence = any(
                    getattr(item, "source_event_id", None) == durable_source_id
                    for item in list_evidence(str(getattr(chunk, "id")))
                )
            # A same-profile fact with a different scope is relevant to this
            # mapping decision only when it cites the declared source. This
            # keeps unrelated rows from changing a missing-target diagnosis.
            if not (has_metadata_evidence or has_table_evidence):
                continue
            if (
                metadata.get("scope") != target_fact.scope
                or metadata.get("scope_id") != expected_scope_id
            ):
                invalid_scope = True
                continue
            candidates.append(
                {
                    "id": str(getattr(chunk, "id")),
                    "version": int(getattr(chunk, "version", 1)),
                    "scope": metadata.get("scope", target_fact.scope),
                }
            )
        if not candidates:
            if invalid_scope:
                raise _TargetMappingError(
                    "target_mapping_invalid", "accepted fact evidence scope does not match target"
                )
            raise _TargetMappingError(
                "missing_accepted_target", "no accepted fact evidence matches target"
            )
        if len(candidates) != 1:
            raise _TargetMappingError(
                "target_mapping_invalid", "target evidence maps to multiple facts"
            )
        return candidates[0]

    @staticmethod
    def _durable_event_id(context: Mapping[str, Any], runtime_event_id: str) -> str:
        identity = [
            context.get("user_profile_id"),
            context.get("host"),
            context.get("session_id"),
            runtime_event_id,
        ]
        return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_v2_runtime_target(
        service: Any, event: SourceEvent, session: SessionScenario
    ) -> Mapping[str, Any] | None:
        store = getattr(getattr(service, "engine", None), "store", None)
        get_all = getattr(store, "get_all", None)
        if not callable(get_all):
            return None
        chunks = get_all()
        candidates = [fact for fact in session.facts if fact.fact_id in event.evidence_fact_ids]
        if not candidates:
            candidates = [
                fact for fact in session.facts if fact.gold_status in {"superseded", "forgotten"}
            ]
        for fact in candidates:
            for chunk in chunks:
                content = str(getattr(chunk, "content", ""))
                if content.strip().casefold() == fact.content.strip().casefold():
                    metadata = getattr(chunk, "metadata", {})
                    return {
                        "id": str(getattr(chunk, "id")),
                        "version": int(getattr(chunk, "version", 1)),
                        "scope": metadata.get("scope", fact.scope),
                    }
        return None

    def _fact_retrieved(
        self,
        fact: FactRecord,
        session: SessionScenario,
        retrievals: Mapping[str, Mapping[str, Any] | None],
    ) -> bool | None:
        relevant = [query for query in session.queries if fact.fact_id in query.relevant_fact_ids]
        if not relevant:
            return None
        measured = [
            self._fact_in_result(fact, retrievals.get(query.query_id)) for query in relevant
        ]
        if any(value is True for value in measured):
            return True
        if any(value is None for value in measured):
            return None
        return False

    def _critical_retention(
        self, service: Any | None, session: SessionScenario
    ) -> tuple[list[bool | None], list[dict[str, Any]]]:
        """Judge critical facts against the accepted post-lifecycle store.

        Retrieval query hits are recorded separately. This metric asks whether
        the expected current fact remains in the runtime store after all source
        events, corrections, forget operations, and maintenance have run.
        """
        critical = [fact for fact in session.facts if fact.critical]
        store = getattr(getattr(service, "engine", None), "store", None)
        get_all = getattr(store, "get_all", None)
        if not callable(get_all):
            return [None for _ in critical], [
                {"fact_id": fact.fact_id, "status": "unjudged", "reason": "store_unavailable"}
                for fact in critical
            ]
        chunks = list(get_all())
        values: list[bool | None] = []
        details: list[dict[str, Any]] = []
        for fact in critical:
            expected_identities = self._fact_identities(fact, session)
            matches = [
                chunk
                for chunk in chunks
                if str(getattr(chunk, "content", "")).strip().casefold()
                == fact.content.strip().casefold()
                and self._metadata_matches_identity(
                    getattr(chunk, "metadata", {}), expected_identities
                )
            ]
            expected_present = fact.gold_status == "supported"
            observed_present = bool(matches)
            values.append(observed_present == expected_present)
            details.append(
                {
                    "fact_id": fact.fact_id,
                    "gold_status": fact.gold_status,
                    "expected_present": expected_present,
                    "observed_present": observed_present,
                    "runtime_fact_ids": [str(getattr(chunk, "id", "")) for chunk in matches],
                    "expected_identities": sorted(expected_identities),
                    "judgement": "exact_content_and_scope_in_accepted_store",
                }
            )
        return values, details

    def _fact_identities(
        self, fact: FactRecord, session: SessionScenario
    ) -> set[tuple[str | None, str, str | None]]:
        identities: set[tuple[str | None, str, str | None]] = set()
        for event in session.events:
            if event.event_id not in fact.evidence_event_ids:
                continue
            profile = event.target_user_profile_id or self.user_profile_id
            if fact.scope == "global":
                identities.add((profile, "global", None))
            elif fact.scope == "workspace":
                identities.add(
                    (profile, "workspace", event.target_workspace_id or session.project_id)
                )
            else:
                identities.add((profile, "project", event.target_project_id or session.project_id))
        if not identities:
            identities.add((self.user_profile_id, fact.scope, session.project_id))
        return identities

    @staticmethod
    def _metadata_matches_identity(
        metadata: Any, identities: set[tuple[str | None, str, str | None]]
    ) -> bool:
        if not isinstance(metadata, Mapping):
            return False
        observed = (
            metadata.get("user_profile_id"),
            metadata.get("scope"),
            metadata.get("scope_id"),
        )
        return observed in identities

    def _extraction_judgements(
        self, service: Any | None, session: SessionScenario
    ) -> tuple[list[bool | None], list[dict[str, Any]]]:
        """Judge each actual runtime output against frozen evidence metadata."""
        store = getattr(getattr(service, "engine", None), "store", None)
        get_all = getattr(store, "get_all", None)
        if not callable(get_all):
            return [None], [{"status": "unjudged", "reason": "runtime_store_unavailable"}]
        events_by_source = {event.event_id: event for event in session.events}
        values: list[bool | None] = []
        details: list[dict[str, Any]] = []
        for chunk in get_all():
            content = str(getattr(chunk, "content", "")).strip()
            metadata = getattr(chunk, "metadata", {})
            matched = next(
                (
                    fact
                    for fact in session.facts
                    if fact.gold_status == "supported"
                    and content
                    and content.casefold() == fact.content.strip().casefold()
                    and self._metadata_matches_identity(
                        metadata, self._fact_identities(fact, session)
                    )
                ),
                None,
            )
            source_ids = metadata.get("source_event_ids", [])
            if not isinstance(source_ids, list):
                source_ids = []
            observed_events: list[SourceEvent] = []
            for source_id in source_ids:
                event = getattr(store, "get_episodic_event", lambda _: None)(source_id)
                original_id = (
                    getattr(event, "metadata", {}).get("source_event_id") if event else None
                )
                if isinstance(original_id, str):
                    for dataset_id, dataset_event in events_by_source.items():
                        if original_id == dataset_id or original_id.startswith(dataset_id + "-"):
                            observed_events.append(dataset_event)
                            break
            accepted = bool(
                matched
                and observed_events
                and all(
                    event.role == "user"
                    and event.write_mode in {"automatic", "explicit"}
                    and event.event_id in matched.evidence_event_ids
                    for event in observed_events
                )
                and str(metadata.get("scope", "")) == matched.scope
            )
            values.append(accepted)
            details.append(
                {
                    "runtime_fact_id": str(getattr(chunk, "id", "")),
                    "content_nonempty": bool(content),
                    "matched_supported_fact_id": matched.fact_id if matched else None,
                    "observed_source_event_ids": [event.event_id for event in observed_events],
                    "accepted": accepted,
                    "judgement": "exact_frozen_text_and_evidence_proxy",
                }
            )
        return values, details

    @staticmethod
    def _fact_in_result(fact: FactRecord, result: Mapping[str, Any] | None) -> bool | None:
        if result is None:
            return None
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return None
        target = fact.content.casefold()
        for item in facts:
            if not isinstance(item, Mapping):
                continue
            content = str(item.get("content", "")).casefold()
            if content and content == target:
                return True
        return False

    @staticmethod
    def _answerable_hit(
        query: QueryRecord, result: Mapping[str, Any] | None, session: SessionScenario
    ) -> bool | None:
        if result is None:
            return None
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return None
        expected = query.expected_answer.strip().casefold() if query.expected_answer else ""
        allowed = {expected} if expected else set()
        allowed.update(
            fact.content.strip().casefold()
            for fact in session.facts
            if fact.fact_id in query.relevant_fact_ids
        )
        return any(
            str(item.get("content", "")).strip().casefold() in allowed
            for item in facts
            if isinstance(item, Mapping) and str(item.get("content", "")).strip()
        )

    @staticmethod
    def _no_answer_correct(result: Mapping[str, Any] | None) -> bool | None:
        if result is None:
            return None
        facts = result.get("facts", [])
        return not facts if isinstance(facts, list) else None

    @staticmethod
    def _state_hit(
        pair: StatePair, result: Mapping[str, Any] | None, *, semantic: bool
    ) -> bool | None:
        if result is None:
            return None
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return None
        target = pair.matched_content.casefold()
        return any(
            str(item.get("content", "")).strip().casefold() == target
            for item in facts
            if isinstance(item, Mapping)
        )

    @staticmethod
    def _stale_counts(
        session: SessionScenario, retrievals: Mapping[str, Mapping[str, Any] | None]
    ) -> tuple[int | None, int | None]:
        if not retrievals or any(result is None for result in retrievals.values()):
            return None, None
        stale_text = {
            fact.content.casefold()
            for fact in session.facts
            if fact.gold_status in {"superseded", "forgotten"}
        }
        returned = [
            item
            for result in retrievals.values()
            for item in (result.get("facts", []) if isinstance(result, Mapping) else [])
            if isinstance(item, Mapping)
        ]
        stale = sum(
            any(text in str(item.get("content", "")).casefold() for text in stale_text)
            for item in returned
        )
        return stale, len(returned)

    def _context(
        self, session: SessionScenario, profile: ExperimentProfile, repetition: int
    ) -> Mapping[str, Any]:
        return {
            "user_profile_id": self.user_profile_id,
            "project_id": session.project_id,
            "workspace_id": None,
            "session_id": f"{session.session_id}-{profile.name}-r{repetition}",
            "host": session.host,
        }

    @staticmethod
    def _target_context(context: Mapping[str, Any], target: Any) -> Mapping[str, Any]:
        result = dict(context)
        for field in ("user_profile_id", "project_id", "workspace_id"):
            value = getattr(target, "target_" + field, None)
            if value is not None:
                result[field] = value
        return result

    @staticmethod
    def _scope_for_event(event: SourceEvent, session: SessionScenario) -> str:
        for fact_id in event.evidence_fact_ids:
            for fact in session.facts:
                if fact.fact_id == fact_id:
                    return fact.scope
        return "project"

    @staticmethod
    def _close_service(service: Any) -> None:
        store = getattr(getattr(service, "engine", None), "store", None)
        close = getattr(store, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _store_bytes(service: Any | None) -> int | None:
        store = getattr(getattr(service, "engine", None), "store", None)
        path = getattr(store, "db_path", None)
        if not path or path == ":memory:":
            return None
        try:
            database = Path(path)
            paths = [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
            size = 0
            for item in paths:
                try:
                    size += item.stat().st_size
                except FileNotFoundError:
                    continue
        except OSError:
            return None
        return size

    @staticmethod
    def _error(operation: str, exc: Exception, event_id: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operation": operation,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if event_id:
            result["event_id"] = event_id
        return result


class _UnsupportedOperation(RuntimeError):
    pass


class _TargetMappingError(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _run_awaitable(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "OllamaEvaluatorAdapter cannot run synchronously inside an active event loop"
    )


def make_ollama_service_factory(
    db_path: Path,
    registry: Any,
    *,
    base_url: str = "http://192.168.86.35:11434",
    model: str = "gemma3:4b",
    embedding_model: str = "nomic-embed-text:latest",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int = 512,
    think: OllamaThink = None,
    query_prefix: str = "",
    document_prefix: str = "",
    symmetric_prefix: str = "",
) -> ServiceFactory:
    """Build a profile-aware Ollama service factory over one SQLite file.

    The registry must already contain an enabled project matching dataset
    contexts. Parent integration can replace this helper when its service
    constructor grows new module settings.
    """

    selected_think = _validate_ollama_think(think)
    task_contract = embedding_task_contract(
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        symmetric_prefix=symmetric_prefix,
    )
    traces: dict[tuple[str, str], dict[str, Any]] = {}

    def factory(profile: ExperimentProfile, run_key: str | None = None) -> Any:
        from mnemos.config import MnemosConfig
        from mnemos.engine import MnemosEngine
        from mnemos.memory_service import LocalMemoryService
        from mnemos.utils.embeddings import OllamaEmbeddingProvider
        from mnemos.utils.llm import OllamaProvider
        from mnemos.utils.storage import SQLiteStore

        config = MnemosConfig()
        enabled = set(profile.enabled_modules)
        # Keep both the current per-module switches and the parent integration
        # switch object aligned when either API is present.
        config.retrieval.affective_enabled = "affective_routing" in enabled
        config.retrieval.spreading_enabled = "spreading_activation" in enabled
        modules = getattr(config, "modules", None)
        if isinstance(modules, dict):
            for name, module in {
                "surprisal": "surprisal_gating",
                "mutable_rag": "mutable_rag",
                "affective": "affective_routing",
                "sleep": "sleep_consolidation",
                "spreading": "spreading_activation",
            }.items():
                modules[name] = module in enabled
        elif modules is not None:
            for name, module in {
                "surprisal": "surprisal_gating",
                "mutable_rag": "mutable_rag",
                "affective": "affective_routing",
                "sleep": "sleep_consolidation",
                "spreading": "spreading_activation",
            }.items():
                if hasattr(modules, name):
                    setattr(modules, name, module in enabled)
        trace_key = (run_key or "<no-run-key>", profile.name)
        trace = traces.get(trace_key)
        if trace is None:
            trace = {
                "config": {
                    "endpoint": base_url.rstrip("/"),
                    "model": model,
                    "embedding_model": embedding_model,
                    "timeout_seconds": timeout,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                "trace_key": {
                    "run_key": run_key,
                    "profile": profile.name,
                },
                "service_instances": 0,
                "predict_calls": [],
                "classify_calls": [],
                "embed_calls": [],
                "response_metadata": [],
            }
            add_ollama_think(trace["config"], selected_think)
            if task_contract is not None:
                trace["config"]["embedding_task_contract"] = task_contract
            traces[trace_key] = trace
        trace["service_instances"] = int(trace.get("service_instances", 0)) + 1

        def observe_llm(data: dict[str, Any]) -> None:
            trace["response_metadata"].append(
                {
                    "provider": "ollama",
                    "kind": "generate",
                    "model": data.get("model"),
                    "created_at": data.get("created_at"),
                    "done": data.get("done"),
                    "done_reason": data.get("done_reason"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "prompt_eval_duration": data.get("prompt_eval_duration"),
                    "eval_count": data.get("eval_count"),
                    "eval_duration": data.get("eval_duration"),
                }
            )

        def observe_embedding(data: dict[str, Any]) -> None:
            trace["response_metadata"].append(
                {
                    "provider": "ollama",
                    "kind": "embedding",
                    "model": data.get("model"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                }
            )

        llm = _TracingLLM(
            OllamaProvider(
                model=model,
                base_url=base_url,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                response_observer=observe_llm,
                think=selected_think,
            ),
            trace,
        )
        base_embedder: Any = OllamaEmbeddingProvider(
            model=embedding_model,
            base_url=base_url,
            timeout=timeout,
            response_observer=observe_embedding,
        )
        if task_contract is not None:
            base_embedder = PrefixEmbeddingProvider(
                base_embedder,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
                symmetric_prefix=symmetric_prefix,
            )
        embedder = _TracingEmbedder(
            base_embedder,
            trace,
        )
        store_path = Path(db_path)
        if run_key:
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_key)
            store_path = store_path.with_name(store_path.stem + "-" + safe_key + store_path.suffix)
        engine = MnemosEngine(
            config=config,
            llm=llm,
            embedder=embedder,
            store=SQLiteStore(str(store_path)),
            scope_registry=registry,
            require_scope_context=True,
        )
        service = LocalMemoryService(engine, registry)
        processor = service.maintenance.processor

        async def traced_processor(job: Any) -> None:
            try:
                await processor(job)
            except Exception as exc:
                detail: dict[str, Any] = {
                    "job_id": str(getattr(job, "id", "")),
                    "kind": str(getattr(job, "kind", "")),
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                errors = getattr(exc, "errors", None)
                if callable(errors):
                    try:
                        # Pydantic errors() can retain exception objects in
                        # ctx. Its JSON serializer preserves their messages
                        # without making the durable trace unserializable.
                        detail["validation_errors"] = (
                            json.loads(exc.json())
                            if isinstance(exc, ValidationError)
                            else json.loads(json.dumps(errors()))
                        )
                    except Exception:
                        detail["validation_errors"] = "unavailable"
                trace.setdefault("job_exceptions", []).append(detail)
                raise

        service.maintenance.processor = traced_processor
        setattr(service, "_benchmark_provider_trace", trace)
        return service

    return factory


def _latest_token_usage(trace: Mapping[str, Any]) -> Mapping[str, int] | None:
    metadata = trace.get("response_metadata", [])
    if not isinstance(metadata, list) or not metadata:
        return None
    item = metadata[-1]
    if not isinstance(item, Mapping):
        return None
    values: dict[str, int] = {}
    for key in ("prompt_eval_count", "eval_count"):
        value = item.get(key)
        if isinstance(value, int):
            values[key] = value
    return values or None


class _TracingLLM(LLMProvider):
    def __init__(self, provider: Any, trace: dict[str, Any]) -> None:
        self._provider = provider
        self._trace = trace
        self.supports_structured_output = bool(
            getattr(self._provider, "supports_structured_output", False)
        )

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        started = time.perf_counter()
        result = cast(str, await self._provider.predict(prompt, **kwargs))
        record: dict[str, Any] = {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_chars": len(prompt),
            "response": result,
            "token_usage": _latest_token_usage(self._trace),
        }
        response_format = kwargs.get("format")
        if response_format is not None:
            record["structured_output_requested"] = True
            record["format_schema_sha256"] = hashlib.sha256(
                json.dumps(response_format, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        self._trace["predict_calls"].append(record)
        return result

    async def classify(self, prompt: str, labels: list[str]) -> dict[str, float]:
        started = time.perf_counter()
        result = cast(dict[str, float], await self._provider.classify(prompt, labels))
        self._trace["classify_calls"].append(
            {
                "latency_ms": (time.perf_counter() - started) * 1000,
                "prompt_chars": len(prompt),
                "response": result,
                "token_usage": _latest_token_usage(self._trace),
            }
        )
        return result


class _TracingEmbedder(EmbeddingProvider):
    def __init__(self, provider: Any, trace: dict[str, Any]) -> None:
        self._provider = provider
        self._trace = trace

    def embed(self, text: str) -> list[float]:
        return self._embed_role(text, "symmetric")

    def _embed_role(self, text: str, role: str) -> list[float]:
        started = time.perf_counter()
        method = getattr(self._provider, f"embed_{role}", self._provider.embed)
        result = cast(list[float], method(text))
        self._trace["embed_calls"].append(
            {
                "latency_ms": (time.perf_counter() - started) * 1000,
                "role": role,
                "text_chars": len(text),
                "vector_dimensions": len(result),
                "token_usage": _latest_token_usage(self._trace),
            }
        )
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "symmetric")

    def _embed_batch_role(self, texts: list[str], role: str) -> list[list[float]]:
        started = time.perf_counter()
        method_name = {
            "query": "embed_batch_queries",
            "document": "embed_batch_documents",
            "symmetric": "embed_batch_symmetric",
        }[role]
        method = getattr(self._provider, method_name, self._provider.embed_batch)
        result = cast(list[list[float]], method(texts))
        self._trace["embed_calls"].append(
            {
                "latency_ms": (time.perf_counter() - started) * 1000,
                "role": role,
                "text_count": len(texts),
                "text_chars": sum(len(text) for text in texts),
                "vector_dimensions": [len(row) for row in result],
                "token_usage": _latest_token_usage(self._trace),
            }
        )
        return result

    def embed_query(self, text: str) -> list[float]:
        return self._embed_role(text, "query")

    def embed_document(self, text: str) -> list[float]:
        return self._embed_role(text, "document")

    def embed_symmetric(self, text: str) -> list[float]:
        return self._embed_role(text, "symmetric")

    def embed_batch_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "query")

    def embed_batch_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "document")

    def embed_batch_symmetric(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_role(texts, "symmetric")

    @property
    def dim(self) -> int:
        return cast(int, self._provider.dim)

    @property
    def should_offload_in_async(self) -> bool:
        return bool(self._provider.should_offload_in_async)
