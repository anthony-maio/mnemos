"""Scope-validated domain operations owned by the persistent local service."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .contracts import (
    CaptureEvent,
    FactProposal,
    ForgetPreviewRequest,
    ForgetRelatedRequest,
    RetrievalRequest,
)
from .claude_identity import legacy_event_id, scoped_event_id
from .hermes_identity import scoped_event_id as hermes_scoped_event_id
from .engine import MnemosEngine
from .jobs import DurableJob, FactEvidence, FactRevision
from .maintenance import MaintenanceScheduler
from .memory_safety import MemoryWriteFirewall
from .retrieval import assertion_status, render_context_chunk
from .scopes import ScopeContext, ScopeRegistry, context_metadata, scope_allows
from .types import EpisodicEvent, Interaction, MemoryChunk, RetrievalFeedbackEvent
from .utils.embeddings import embed_text_async
from .utils.storage import SQLiteStore, compute_canonical_key, normalize_memory_text


class LocalMemoryService:
    """No host-provided metadata can override resolved scope or write policy."""

    def __init__(self, engine: MnemosEngine, registry: ScopeRegistry) -> None:
        self.engine = engine
        self.registry = registry
        self.engine._scope_registry = registry
        self.engine._require_scope_context = True
        self.firewall = MemoryWriteFirewall(engine.config.safety)
        self.maintenance = MaintenanceScheduler(
            engine.store, self.process_job, after_batch=engine.rebuild_graph_batch
        )
        self._cognition_probe: dict[str, Any] = {"status": "not_observed"}

    async def close(self) -> None:
        """Release engine resources after the owner has drained pending tasks."""
        self.maintenance.stop()
        await self.engine.aclose()

    def _original_source_status(
        self,
        fact: MemoryChunk,
        evidence: FactEvidence,
        context: ScopeContext,
        allowed_scopes: tuple[str, ...],
    ) -> str:
        """Report only whether the retained local source episode is still present."""

        source_event_id = evidence.source_event_id
        if not source_event_id:
            return "no_source_evidence"
        source_event = self.engine.store.get_episodic_event(source_event_id)
        if source_event is None:
            # This includes the documented processed-event retention expiry. An
            # external transcript or host file is never probed here.
            return "unavailable_locally"
        if not scope_allows(source_event.metadata, context, allowed_scopes):
            return "unavailable_locally"
        fact_scope = fact.metadata.get("scope")
        fact_scope_id = fact.metadata.get("scope_id")
        if (
            source_event.metadata.get("scope") != fact_scope
            or source_event.metadata.get("scope_id") != fact_scope_id
        ):
            return "unavailable_locally"
        return "retained_locally"

    @staticmethod
    def _source_status_summary(statuses: list[str]) -> str:
        if "retained_locally" in statuses:
            return "retained_locally"
        if "unavailable_locally" in statuses:
            return "unavailable_locally"
        return "no_source_evidence"

    def capture(
        self,
        event: CaptureEvent,
        *,
        scope: str | None = None,
        explicit: bool = False,
        correction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.registry.validate_context(event.context)
        scope = scope or event.scope
        if not self.engine._capture_channel_allowed("manual" if explicit else "hook"):
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "Capture channel is disabled",
            }
        metadata = context_metadata(event.context, scope)
        for identity_field in (event.event_id, event.context.session_id, event.source_kind):
            decision = self.firewall.apply(identity_field)
            if not decision.allowed or decision.content != identity_field:
                return {
                    "schema_version": 1,
                    "status": "rejected",
                    "reason": "Source metadata violates write policy",
                }
        safety = self.firewall.apply(event.content)
        if not safety.allowed:
            return {"schema_version": 1, "status": "rejected", "reason": safety.reason}
        claude_provisional = (
            event.identity_version == 2
            and event.context.host == "claude-code"
            and event.source_kind == "claude_code_transcript"
        )
        hermes_provisional = (
            event.identity_version == 1
            and event.context.host == "hermes"
            and event.source_kind == "hermes_memory_provider"
            and event.source_ref is not None
            and event.source_ref.startswith("provisional:v1:")
            and event.event_id
            == hermes_scoped_event_id(event.context, event.source_ref, event.role, event.content)
        )
        if event.identity_status == "provisional_unassociated" and (
            not (claude_provisional or hermes_provisional)
            or explicit
            or correction is not None
            or event.role not in {"user", "assistant"}
            or (scope or event.scope) != "project"
            or event.source_ref is None
            or not event.source_ref.strip()
        ):
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "Provisional identity requires a verified non-explicit host capture",
            }
        if (
            event.context.host == "hermes"
            and event.source_kind == "hermes_memory_provider"
            and (event.source_ref or "").startswith("provisional:v1:")
            and event.identity_status != "provisional_unassociated"
        ):
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "A provisional Hermes receipt cannot become extraction evidence",
            }
        source_ref = event.source_ref
        if source_ref is not None:
            safe_ref = self.firewall.apply(source_ref)
            if not safe_ref.allowed:
                return {
                    "schema_version": 1,
                    "status": "rejected",
                    "reason": "Source reference violates write policy",
                }
            source_ref = safe_ref.content
        identity = [
            event.context.user_profile_id,
            event.context.host,
            event.context.session_id,
            event.event_id,
        ]
        key = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()
        if event.identity_version == 2 or event.event_id.startswith("claude:v2:"):
            # An alias is derived from the verified capture, never selected by
            # a caller. Redaction makes old content-derived identities ambiguous.
            if (
                event.identity_version != 2
                or event.context.host != "claude-code"
                or event.source_kind != "claude_code_transcript"
                or explicit
                or correction is not None
                or source_ref is None
                or not source_ref.strip()
                or event.role not in {"user", "assistant"}
                or scope != "project"
                or safety.content != event.content
                or event.content != event.content.strip()
                or source_ref != event.source_ref
                or "[REDACTED_" in event.content
                or "[REDACTED_" in source_ref
                or event.event_id
                != scoped_event_id(event.context, source_ref, event.role, event.content)
            ):
                return {
                    "schema_version": 1,
                    "status": "rejected",
                    "reason": "Claude scoped identity cannot be verified",
                }
        canonical = compute_canonical_key(
            safety.content, scope, metadata["scope_id"], event.context.user_profile_id
        )
        if self.engine.store.is_suppressed(source_event_id=key) or (
            not explicit and self.engine.store.is_suppressed(canonical_key=canonical)
        ):
            return {
                "schema_version": 1,
                "status": "committed",
                "event_id": key,
                "processing_status": "suppressed",
                "job_id": None,
            }
        suppression_source_ids: tuple[str, ...] = ()
        if event.event_id.startswith("claude:v2:") and event.identity_status == "normal":
            assert source_ref is not None
            old_adapter_id = legacy_event_id(
                event.context.session_id, source_ref, event.role, event.content
            )
            old_key = hashlib.sha256(
                json.dumps([*identity[:3], old_adapter_id]).encode("utf-8")
            ).hexdigest()
            suppression_source_ids = (old_key,)
            if self.engine.store.is_suppressed(source_event_id=old_key):
                # Old tombstones lack project ownership after source erasure.
                # Keep the suppression even if this causes a conservative block.
                return {
                    "schema_version": 1,
                    "status": "committed",
                    "event_id": key,
                    "processing_status": "suppressed",
                    "reason": "legacy_source_suppressed",
                    "job_id": None,
                }
            old_event = self.engine.store.get_episodic_event(old_key)
            old_job = self.engine.store.get_job("event:" + old_key)
            expected_payload = {
                "event_id": old_key,
                "context": event.context.model_dump(),
                "scope": scope,
                "source_digest": hashlib.sha256(
                    json.dumps([event.role, safety.content, scope, metadata["scope_id"]]).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
            exact_event = old_event is None or (
                old_event.content == safety.content
                and old_event.role == event.role
                and old_event.scope == scope
                and old_event.scope_id == metadata["scope_id"]
                and all(old_event.metadata.get(k) == v for k, v in metadata.items())
                and old_event.metadata.get("source_ref") == source_ref
                and old_event.metadata.get("source_kind") == event.source_kind
            )
            if (
                old_job is not None
                and old_job.kind == "extract"
                and old_job.payload == expected_payload
                and exact_event
            ):
                if old_event is None and old_job.status != "completed":
                    return {
                        "schema_version": 1,
                        "status": "rejected",
                        "reason": "Legacy Claude source is unavailable",
                    }
                return {
                    "schema_version": 1,
                    "status": "committed",
                    "event_id": old_key,
                    "processing_status": (
                        "consolidated"
                        if old_job.status == "completed"
                        else old_event.status if old_event is not None else "pending"
                    ),
                    "job_id": old_job.id,
                    "identity_status": event.identity_status,
                    "fact_eligible": event.identity_status == "normal",
                }
        metadata.update(
            {
                "source_event_id": event.event_id,
                "source_ref": source_ref,
                "source_kind": event.source_kind,
                "explicit_remember": explicit,
                "identity_status": event.identity_status,
            }
        )
        record = EpisodicEvent(
            id=key,
            event_fingerprint=key,
            role=event.role,
            content=safety.content,
            normalized_content=normalize_memory_text(safety.content),
            scope=scope,
            scope_id=metadata["scope_id"],
            source=event.context.host,
            metadata=metadata,
            occurred_at=event.occurred_at.astimezone(timezone.utc),
            status=(
                "deferred" if event.identity_status == "provisional_unassociated" else "pending"
            ),
        )
        job_payload = {
            "event_id": key,
            "context": event.context.model_dump(),
            "scope": scope,
            "source_digest": hashlib.sha256(
                json.dumps([event.role, safety.content, scope, metadata["scope_id"]]).encode(
                    "utf-8"
                )
            ).hexdigest(),
            **(correction or {}),
        }
        if event.identity_status == "provisional_unassociated":
            job_payload["identity_status"] = event.identity_status
        job = DurableJob(
            id="event:" + key,
            kind=(
                "capture_only"
                if event.identity_status == "provisional_unassociated"
                else "correct" if correction else "remember" if explicit else "extract"
            ),
            payload=job_payload,
        )
        existing_job = self.engine.store.get_job(job.id)
        if existing_job is not None:
            same_payload = existing_job.payload == job.payload
            legacy_payload = dict(job.payload)
            legacy_payload.pop("identity_status", None)
            is_legacy_payload = existing_job.payload == legacy_payload
            if is_legacy_payload and event.identity_status == "provisional_unassociated":
                return {
                    "schema_version": 1,
                    "status": "rejected",
                    "reason": "Existing legacy capture cannot be reclassified as provisional",
                }
            if existing_job.kind != job.kind or not (same_payload or is_legacy_payload):
                raise ValueError("Source event identity already has a different payload")
            if existing_job.status == "completed":
                return {
                    "schema_version": 1,
                    "status": "committed",
                    "event_id": key,
                    "processing_status": (
                        "deferred"
                        if event.identity_status == "provisional_unassociated"
                        else "consolidated"
                    ),
                    "job_id": job.id,
                    "identity_status": event.identity_status,
                    "fact_eligible": event.identity_status == "normal",
                }
        stored = self.engine.store.capture_event_and_job(
            record, job, suppression_source_ids=suppression_source_ids
        )
        if stored is None:
            return {
                "schema_version": 1,
                "status": "committed",
                "event_id": key,
                "processing_status": "suppressed",
                "job_id": None,
            }
        self.maintenance.note_activity()
        return {
            "schema_version": 1,
            "status": "committed",
            "event_id": stored.id,
            "processing_status": stored.status,
            "job_id": job.id,
            "identity_status": event.identity_status,
            "fact_eligible": event.identity_status == "normal",
        }

    async def _automatic_encoding_gate(
        self, interaction: Interaction
    ) -> tuple[str | None, dict[str, Any]]:
        """Return an automatic encoding rejection and its durable diagnostics.

        Capture, scope validation, and explicit writes remain outside this
        decision. Benchmark services can substitute a comparison policy here
        while retaining the same extraction and transactional commit path.
        """
        surprise, status, prediction, mode = (
            await self.engine.surprisal_gate._compute_surprisal(interaction)
            if self.engine.config.modules.surprisal
            else (None, "disabled", None, "disabled")
        )
        diagnostics = {
            "surprisal_score": surprise,
            "surprisal_status": status,
            "prediction": prediction,
            "prediction_mode": mode,
            "threshold": self.engine.config.surprisal.threshold,
        }
        if (
            self.engine.config.modules.surprisal
            and surprise is None
            and self.engine.config.surprisal.threshold > 0
        ):
            raise ConnectionError("Surprisal prediction is unavailable")
        rejected = surprise is not None and surprise <= self.engine.config.surprisal.threshold
        return "surprisal_filtered" if rejected else None, diagnostics

    async def process_job(self, job: DurableJob) -> None:
        """Model work happens outside the transaction that accepts its results."""
        if job.kind == "reconsolidate_review":
            await self.engine.mutable_rag.process_job(job)
            chunk = self.engine.store.get(job.payload["target_fact_id"])
            if chunk is not None:
                self.engine.refresh_graph_chunk(chunk)
            return
        context = self.registry.validate_context(
            ScopeContext.model_validate(job.payload["context"])
        )
        event = self.engine.store.get_episodic_event(job.payload["event_id"])
        if event is None:
            raise ValueError("Source event is unavailable")
        event_identity_status = event.metadata.get("identity_status", "normal")
        if event_identity_status == "provisional_unassociated" and job.kind != "capture_only":
            raise ValueError("Provisional source events cannot run extraction jobs")
        if job.kind == "capture_only" and event_identity_status != "provisional_unassociated":
            raise ValueError("Capture-only jobs require provisional source events")
        scope = job.payload["scope"]
        if not scope_allows(event.metadata, context, (scope,)):
            raise ValueError("Source event is outside the requested scope")
        if not job.lease_owner or not job.lease_token:
            raise ValueError("A live job lease is required")
        if event.status == "consolidated":
            if not self.engine.store.complete_job(
                job.id,
                owner=job.lease_owner,
                lease_token=job.lease_token,
                result={"decision": "already_processed"},
            ):
                raise RuntimeError("Job lease expired")
            return
        if job.kind == "capture_only":
            committed = self.engine.store.commit_fact_and_job(
                job.id,
                owner=job.lease_owner,
                lease_token=job.lease_token,
                chunks=[],
                evidence=[],
                revisions=[],
                processed_event_ids=[event.id],
                result={"decision": "identity_pending", "fact_eligible": False},
            )
            if not committed:
                raise RuntimeError("Job lease expired before provisional capture was recorded")
            return
        interaction = Interaction(
            role=event.role,
            content=event.content,
            timestamp=event.occurred_at,
            metadata=event.metadata,
        )
        diagnostics: dict[str, Any] = {}
        decision = "facts"
        captured_passthrough = job.kind == "extract" and not self.engine.config.modules.sleep
        if job.kind in {"remember", "correct"}:
            proposals = [
                FactProposal(
                    content=event.content,
                    evidence_ids=[event.id],
                    operation="correct" if job.kind == "correct" else "add",
                    target_fact_id=job.payload.get("target_fact_id"),
                    expected_version=job.payload.get("expected_version"),
                )
            ]
        elif job.kind == "extract":
            rejected, diagnostics = await self._automatic_encoding_gate(interaction)
            if rejected is not None:
                proposals = []
                decision = rejected
            else:

                def same_partition(chunk: MemoryChunk) -> bool:
                    return scope_allows(chunk.metadata, context, (scope,))

                setattr(
                    same_partition,
                    "_mnemos_scope_filter",
                    {
                        "user_profile_id": context.user_profile_id,
                        "project_id": context.project_id,
                        "workspace_id": context.workspace_id,
                        "allowed_scopes": [scope],
                    },
                )
                existing = (
                    self.engine.store.retrieve(
                        await embed_text_async(self.engine.embedder, event.content, role="query"),
                        top_k=5,
                        filter_fn=same_partition,
                    )
                    if self.engine.config.modules.sleep
                    else []
                )
                proposals = (
                    await self.engine.sleep_daemon.propose_facts(
                        [event], self.engine.llm, existing_facts=existing
                    )
                    if self.engine.config.modules.sleep
                    else [
                        FactProposal(
                            content=event.content, evidence_ids=[event.id], operation="add"
                        )
                    ]
                )
                if not proposals:
                    decision = "no_durable_facts"
        else:
            raise ValueError("Unsupported maintenance job kind")
        state = (
            await self.engine.affective_router.classify_state(interaction)
            if proposals and self.engine.config.modules.affective
            else None
        )
        chunks = []
        evidence = []
        revisions = []
        expected_versions = {}
        seen_keys = set()
        for proposal in proposals:
            if set(proposal.evidence_ids) != {event.id}:
                raise ValueError("Fact proposal has unauthorized evidence")
            proposed_content = proposal.content
            if proposal.operation == "support":
                assert proposal.target_fact_id is not None
                supported = self.engine.store.get(proposal.target_fact_id)
                if supported is None or not scope_allows(supported.metadata, context, (scope,)):
                    raise ValueError("Support target is unavailable in this scope")
                proposed_content = supported.content
            if (
                proposal.operation == "correct"
                and job.kind != "correct"
                and not self.engine.config.modules.mutable_rag
            ):
                proposal.operation = "conflict"
            raw_safe = self.firewall.apply(proposed_content)
            if not raw_safe.allowed:
                raise ValueError("Fact proposal violates write policy")
            raw_canonical_key = compute_canonical_key(
                raw_safe.content, scope, event.scope_id, context.user_profile_id
            )
            if job.kind == "extract" and self.engine.store.is_suppressed(
                canonical_key=raw_canonical_key, source_event_id=event.id
            ):
                decision = "suppressed"
                continue
            proposed_content = raw_safe.content
            if event.role != "user":
                proposed_content = f"{event.role.capitalize()} reported: {proposed_content}"
            safe = self.firewall.apply(proposed_content)
            if not safe.allowed:
                raise ValueError("Fact proposal violates write policy")
            canonical_key = compute_canonical_key(
                safe.content, scope, event.scope_id, context.user_profile_id
            )
            if canonical_key in seen_keys:
                continue
            seen_keys.add(canonical_key)
            if job.kind == "extract" and self.engine.store.is_suppressed(
                canonical_key=canonical_key, source_event_id=event.id
            ):
                decision = "suppressed"
                continue
            metadata = {
                **context_metadata(context, scope),
                "source": (
                    "explicit_remember"
                    if job.kind == "remember"
                    else (
                        "explicit_correction"
                        if job.kind == "correct"
                        else (
                            "captured_passthrough"
                            if captured_passthrough
                            else "sleep_consolidation"
                        )
                    )
                ),
                "source_roles": [event.role],
                "source_event_ids": [event.id],
                "encoding_diagnostics": diagnostics,
                "assertion_status": (
                    "conflict"
                    if proposal.operation == "conflict"
                    else "unknown" if captured_passthrough else "asserted"
                ),
            }
            embedding = await embed_text_async(self.engine.embedder, safe.content, role="document")
            if proposal.operation == "correct":
                previous, _ = self.engine.mutable_rag._validate_proposal(proposal)
                if previous is None or not scope_allows(previous.metadata, context, (scope,)):
                    raise ValueError("Correction target is unavailable in this scope")
                if previous.version != proposal.expected_version:
                    raise ValueError("Correction version conflict")
                if previous.id in expected_versions:
                    raise ValueError("A single extraction cannot correct a fact twice")
                expected_versions[previous.id] = previous.version
                chunk = previous.reconsolidate(safe.content)
                chunk.embedding = embedding
                chunk.metadata = {**chunk.metadata, **metadata}
                chunk.canonical_key = canonical_key
                revisions.append(
                    FactRevision(
                        fact_id=chunk.id,
                        version=previous.version,
                        content=previous.content,
                        reason="Before source-backed correction",
                    )
                )
            else:
                chunk = MemoryChunk(
                    content=safe.content,
                    embedding=embedding,
                    metadata=metadata,
                    canonical_key=canonical_key,
                )
            chunk.last_observed_at = event.occurred_at
            self.engine.affective_router.tag_chunk(chunk, state)
            chunks.append(chunk)
            evidence.append(
                FactEvidence(
                    fact_id=chunk.id,
                    source_event_id=event.id,
                    excerpt=event.content[:1000],
                    source_ref=event.metadata.get("source_ref"),
                )
            )
        committed = self.engine.store.commit_fact_and_job(
            job.id,
            owner=job.lease_owner,
            lease_token=job.lease_token,
            chunks=chunks,
            evidence=evidence,
            revisions=revisions,
            processed_event_ids=[event.id],
            expected_version=job.payload.get("expected_version"),
            expected_versions=expected_versions,
            result={"decision": decision, "encoding_diagnostics": diagnostics},
        )
        if not committed:
            raise RuntimeError("Job lease expired before results could be committed")
        self.engine.surprisal_gate.add_to_history(interaction)
        committed_job = self.engine.store.get_job(job.id)
        accepted_ids = (committed_job.result or {}).get("fact_ids", []) if committed_job else []
        for fact_id in accepted_ids:
            current = self.engine.store.get(fact_id)
            if current is not None:
                self.engine.refresh_graph_chunk(current)

    async def retrieve(self, request: RetrievalRequest) -> dict[str, Any]:
        self.registry.validate_context(request.context)
        chunks = await self.engine.retrieve(
            request.query,
            top_k=request.top_k,
            context=request.context,
            allowed_scopes=tuple(request.allowed_scopes),
            profile=request.profile,
            max_context_tokens=request.max_context_tokens,
        )
        facts = []
        for chunk in chunks:
            support = self.engine.store.list_fact_evidence(chunk.id)
            sources = []
            source_statuses = []
            for item in support:
                source_status = self._original_source_status(
                    chunk, item, request.context, tuple(request.allowed_scopes)
                )
                source_statuses.append(source_status)
                sources.append(
                    {
                        **item.model_dump(mode="json"),
                        "original_source_status": source_status,
                    }
                )
            facts.append(
                {
                    "id": chunk.id,
                    "version": chunk.version,
                    "content": chunk.content,
                    "scope": chunk.metadata.get("scope"),
                    "scope_id": chunk.metadata.get("scope_id"),
                    "assertion_status": assertion_status(chunk),
                    "sources": sources,
                    "support_available": bool(support),
                    "original_source_status": self._source_status_summary(source_statuses),
                    "scores": chunk.metadata.get("retrieval_trace", {}),
                }
            )
        return {
            "schema_version": 1,
            "profile": request.profile,
            "facts": facts,
            "context": "".join(render_context_chunk(chunk) for chunk in chunks),
            "token_count_method": "utf8_bytes_upper_bound",
            "context_token_upper_bound": sum(
                len(render_context_chunk(chunk).encode("utf-8")) for chunk in chunks
            ),
        }

    async def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation == "configure_project":
            from pathlib import Path
            from .scopes import default_registry_path

            path = Path(payload["path"]).expanduser()
            if not path.is_dir() or not isinstance(payload.get("enabled"), bool):
                raise ValueError("Project path must be a directory and enabled must be a boolean")
            profile = payload["user_profile_id"]
            if not isinstance(profile, str) or not profile.strip():
                raise ValueError("A nonempty user profile is required")
            updated = self.registry.model_copy(deep=True)
            project = updated.register_project(
                path, user_profile_id=profile, enabled=payload["enabled"]
            )
            project.enabled = payload["enabled"]
            if payload.get("workspace_id"):
                updated.add_membership(project.id, payload["workspace_id"])
            updated.save(default_registry_path())
            self.registry = updated
            self.engine._scope_registry = updated
            return {
                "schema_version": 1,
                "status": "committed",
                "project": project.model_dump(mode="json"),
            }
        if operation == "probe_cognition":
            from .utils.llm import MockLLMProvider, UnavailableLLMProvider

            context = self.registry.validate_context(
                ScopeContext.model_validate(payload["context"])
            )
            self._cognition_probe = {"status": "failed", "input_specific_extraction": False}
            if isinstance(self.engine.llm, (MockLLMProvider, UnavailableLLMProvider)):
                return {
                    "schema_version": 1,
                    **self._cognition_probe,
                    "reason": "A configured real cognition provider is required",
                }
            label = "mnemos-probe-" + uuid4().hex
            content = f"The validation project uses deployment label {label}."
            probe_event = EpisodicEvent(
                id=str(uuid4()),
                role="user",
                content=content,
                normalized_content=normalize_memory_text(content),
                event_fingerprint=label,
                scope="project",
                scope_id=context.project_id,
                metadata=context_metadata(context, "project"),
            )
            try:
                proposals = await self.engine.sleep_daemon.propose_facts(
                    [probe_event], self.engine.llm
                )
                passed = any(
                    label in fact.content and probe_event.id in fact.evidence_ids
                    for fact in proposals
                )
                self._cognition_probe = {
                    "status": "observed" if passed else "failed",
                    "input_specific_extraction": passed,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "provider": type(self.engine.llm).__name__,
                }
            except Exception as exc:
                self._cognition_probe["error"] = type(exc).__name__
            return {"schema_version": 1, **self._cognition_probe}
        if operation == "backup":
            from .backup import create_backup

            if not isinstance(self.engine.store, SQLiteStore):
                raise ValueError("Backup requires SQLite storage")
            manifest = create_backup(self.engine.store, payload["destination"], self.registry)
            return {
                "schema_version": 1,
                "status": "committed",
                "manifest": manifest,
                "destination": payload["destination"],
            }
        if operation == "restore":
            from .backup import restore_backup

            if not isinstance(self.engine.store, SQLiteStore):
                raise ValueError("Restore requires SQLite storage")
            manifest, registry = restore_backup(
                payload["archive"],
                payload["destination"],
                current_store=self.engine.store,
                current_registry=self.registry,
            )
            return {
                "schema_version": 1,
                "status": "committed",
                "manifest": manifest,
                "destination": payload["destination"],
                "registry": registry.model_dump(mode="json"),
                "activated": False,
            }
        if operation == "reload_registry":
            from .scopes import default_registry_path

            self.registry = ScopeRegistry.load(default_registry_path())
            self.engine._scope_registry = self.registry
            return {
                "schema_version": 1,
                "status": "committed",
                "projects": len(self.registry.projects),
            }
        if operation in {
            "list",
            "legacy_review",
            "feedback",
            "feedback_list",
            "job",
            "jobs",
            "retry_job",
        }:
            context = self.registry.validate_context(
                ScopeContext.model_validate(payload["context"])
            )
            allowed = payload.get("allowed_scopes", ["project", "global"])
            if any(scope not in {"project", "workspace", "global"} for scope in allowed):
                raise ValueError("Invalid allowed scopes")

            def permitted(chunk: MemoryChunk) -> bool:
                return scope_allows(chunk.metadata, context, allowed)

            setattr(
                permitted,
                "_mnemos_scope_filter",
                {
                    "user_profile_id": context.user_profile_id,
                    "project_id": context.project_id,
                    "workspace_id": context.workspace_id,
                    "allowed_scopes": allowed,
                },
            )
            if operation == "list":
                chunks = self.engine.store.list_chunks(
                    limit=min(max(int(payload.get("limit", 50)), 0), 500),
                    after_id=payload.get("after_id", ""),
                    filter_fn=permitted,
                )
                return {
                    "schema_version": 1,
                    "facts": [
                        chunk.model_dump(mode="json", exclude={"embedding"}) for chunk in chunks
                    ],
                    "jobs": self.engine.store.job_summary(
                        now=datetime.now(timezone.utc),
                        user_profile_id=context.user_profile_id,
                        project_id=context.project_id,
                        workspace_id=context.workspace_id,
                    ),
                    "next_cursor": chunks[-1].id if chunks else None,
                }
            if operation == "legacy_review":
                limit = min(max(int(payload.get("limit", 50)), 1), 500)
                chunks = self.engine.store.list_legacy_chunks(
                    limit=limit + 1,
                    after_id=str(payload.get("after_id") or ""),
                )
                page = chunks[:limit]
                return {
                    "schema_version": 1,
                    "status": "committed",
                    "read_only": True,
                    "migration_required": True,
                    "facts": [
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "version": chunk.version,
                            "created_at": chunk.created_at.isoformat(),
                            "updated_at": chunk.updated_at.isoformat(),
                        }
                        for chunk in page
                    ],
                    "truncated": len(chunks) > limit,
                    "next_cursor": page[-1].id if len(chunks) > limit and page else None,
                    "limit": limit,
                }
            if operation == "jobs":
                status = payload.get("status")
                if status not in {None, "pending", "leased", "completed", "failed", "exhausted"}:
                    raise ValueError("Invalid job status")
                limit = min(max(int(payload.get("limit", 50)), 1), 500)
                jobs = self.engine.store.list_jobs(
                    status=status,
                    limit=limit + 1,
                    user_profile_id=context.user_profile_id,
                    project_id=context.project_id,
                    workspace_id=context.workspace_id,
                )
                return {
                    "schema_version": 1,
                    "jobs": [
                        job.model_dump(mode="json", exclude={"payload", "lease_token"})
                        for job in jobs[:limit]
                    ],
                    "truncated": len(jobs) > limit,
                    "limit": limit,
                }
            if operation in {"job", "retry_job"}:
                job = self.engine.store.get_job(payload["job_id"])
                if (
                    job is None
                    or job.payload.get("context", {}).get("user_profile_id")
                    != context.user_profile_id
                ):
                    raise KeyError("Job not found in the requested profile")
                job_context = ScopeContext.model_validate(job.payload["context"])
                if (
                    job_context.project_id != context.project_id
                    or job_context.workspace_id != context.workspace_id
                ):
                    raise KeyError("Job not found in the requested scope")
                if operation == "retry_job":
                    if not self.engine.store.retry_job(job.id):
                        raise ValueError("Only exhausted jobs can be manually retried")
                    retried_job = self.engine.store.get_job(job.id)
                    assert retried_job is not None
                    return {
                        "schema_version": 1,
                        "status": "committed",
                        "retried": True,
                        "job": retried_job.model_dump(
                            mode="json", exclude={"payload", "lease_token"}
                        ),
                    }
                return {
                    "schema_version": 1,
                    "job": job.model_dump(mode="json", exclude={"payload", "lease_token"}),
                }
            if operation == "feedback_list":
                events = self.engine.store.list_feedback_events(
                    user_profile_id=context.user_profile_id, event_type=payload.get("event_type")
                )
                events = [
                    event
                    for event in events
                    if scope_allows(
                        {
                            "user_profile_id": event.user_profile_id,
                            "scope": event.scope,
                            "scope_id": event.scope_id,
                        },
                        context,
                        allowed,
                    )
                ]
                return {
                    "schema_version": 1,
                    "events": [event.model_dump(mode="json") for event in events],
                }
            chunk_ids = payload.get("chunk_ids", [])
            for fact_id in chunk_ids:
                fact = self.engine.store.get(fact_id)
                if fact is None or not permitted(fact):
                    raise ValueError("Feedback references a fact outside the permitted scope")
            query, notes = self.firewall.apply(payload["query"]), self.firewall.apply(
                payload.get("notes", "")
            )
            if not query.allowed or not notes.allowed:
                return {"schema_version": 1, "status": "rejected", "stored": False}
            metadata = context_metadata(context, payload.get("scope", "project"))
            feedback = RetrievalFeedbackEvent(
                event_type=payload["event_type"],
                query=query.content,
                notes=notes.content,
                chunk_ids=chunk_ids,
                scope=metadata["scope"],
                scope_id=metadata["scope_id"],
                user_profile_id=context.user_profile_id,
            )
            self.engine.store.store_feedback_event(feedback)
            return {
                "schema_version": 1,
                "status": "committed",
                "stored": True,
                "event_type": feedback.event_type,
                "chunk_ids": chunk_ids,
                "id": feedback.id,
            }
        if operation == "capture":
            return self.capture(CaptureEvent.model_validate(payload))
        if operation == "remember":
            context = ScopeContext.model_validate(payload["context"])
            event = CaptureEvent(
                event_id=payload.get("event_id") or str(uuid4()),
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role=payload.get("role", "user"),
                content=payload["content"],
                source_kind="explicit_remember",
                source_ref=payload.get("source_ref"),
            )
            return self.capture(event, scope=payload.get("scope", "project"), explicit=True)
        if operation == "retrieve":
            return await self.retrieve(RetrievalRequest.model_validate(payload))
        if operation == "correct":
            context = self.registry.validate_context(
                ScopeContext.model_validate(payload["context"])
            )
            previous = self.engine.store.get(payload["fact_id"])
            if previous is None or not scope_allows(
                previous.metadata, context, ("project", "workspace", "global")
            ):
                raise KeyError("Correction target not found in the requested scope")
            if previous.version != payload["expected_version"]:
                raise ValueError("Correction version conflict")
            event = CaptureEvent(
                event_id=payload.get("event_id") or str(uuid4()),
                context=context,
                occurred_at=datetime.now(timezone.utc),
                role="user",
                content=payload["content"],
                source_kind="explicit_correction",
                source_ref=payload.get("source_ref"),
            )
            return self.capture(
                event,
                scope=previous.metadata["scope"],
                explicit=True,
                correction={"target_fact_id": previous.id, "expected_version": previous.version},
            )
        if operation == "maintenance":
            return {
                "schema_version": 1,
                "completed": await self.maintenance.run_once(force=True),
                "maintenance": self.maintenance.health(),
                "graph": self._graph_status(),
            }
        if operation == "forget_preview":
            preview_request = ForgetPreviewRequest.model_validate(payload)
            context = self.registry.validate_context(preview_request.context)
            if context.project_id is None:
                raise ValueError("Related forget requires a project context")
            preview = self.engine.store.preview_related_forget(
                fact_id=preview_request.fact_id,
                expected_version=preview_request.expected_version,
                user_profile_id=context.user_profile_id,
                project_id=context.project_id,
                limit=preview_request.limit,
            )
            return {"schema_version": 1, "status": "preview", **preview}
        if operation == "forget_related":
            related_request = ForgetRelatedRequest.model_validate(payload)
            context = self.registry.validate_context(related_request.context)
            if context.project_id is None:
                raise ValueError("Related forget requires a project context")
            result = self.engine.store.forget_related(
                fact_id=related_request.fact_id,
                expected_version=related_request.expected_version,
                user_profile_id=context.user_profile_id,
                project_id=context.project_id,
                preview_digest=related_request.preview_digest,
                limit=related_request.preview_limit,
                selected_fact_ids=related_request.selected_fact_ids,
                selected_source_event_ids=related_request.selected_source_event_ids,
            )
            for selected_id in result["selected_fact_ids"]:
                self.engine.spreading_activation.remove_node(selected_id)
            self.engine.surprisal_gate.clear_history()
            self.engine.sleep_daemon._episodic_buffer.clear()
            self.engine.mutable_rag.clear_labile_queue()
            self.engine._retrieval_traces.clear()
            return {"schema_version": 1, **result}
        if operation == "forget":
            context = self.registry.validate_context(
                ScopeContext.model_validate(payload["context"])
            )
            chunk = self.engine.store.get(payload["fact_id"])
            if chunk is None or not scope_allows(
                chunk.metadata, context, ("project", "workspace", "global")
            ):
                raise KeyError("Memory not found in the requested scope")
            forgotten = self.engine.store.forget_fact(
                chunk.id, expected_version=payload["expected_version"]
            )
            if forgotten:
                self.engine.spreading_activation.remove_node(chunk.id)
                self.engine.surprisal_gate.clear_history()
                self.engine.sleep_daemon._episodic_buffer.clear()
                self.engine.mutable_rag.clear_labile_queue()
                self.engine._retrieval_traces.clear()
            return {
                "schema_version": 1,
                "status": "committed",
                "forgotten": forgotten,
                "fact_id": chunk.id,
                "erasure_scope": "live_memory_and_replay_suppression",
            }
        if operation == "inspect":
            context = self.registry.validate_context(
                ScopeContext.model_validate(payload["context"])
            )
            chunk = self.engine.store.get(payload["fact_id"])
            if chunk is None or not scope_allows(
                chunk.metadata, context, ("project", "workspace", "global")
            ):
                raise KeyError("Memory not found in the requested scope")
            evidence = [
                {
                    **item.model_dump(mode="json"),
                    "original_source_status": self._original_source_status(
                        chunk, item, context, ("project", "workspace", "global")
                    ),
                }
                for item in self.engine.store.list_fact_evidence(chunk.id)
            ]
            return {
                "schema_version": 1,
                "fact": chunk.model_dump(mode="json", exclude={"embedding"}),
                "evidence": evidence,
                "original_source_status": self._source_status_summary(
                    [item["original_source_status"] for item in evidence]
                ),
                "revisions": [
                    item.model_dump(mode="json")
                    for item in self.engine.store.list_fact_revisions(chunk.id)
                ],
            }
        if operation == "health":
            return {
                "schema_version": 1,
                "status": "running",
                "store": self.engine.store.get_stats(),
                "runtime": {
                    "store_type": type(self.engine.store).__name__,
                    "database_path": getattr(self.engine.store, "db_path", None),
                    "cognition_provider": type(self.engine.llm).__name__,
                    "embedding_provider": type(self.engine.embedder).__name__,
                },
                "capabilities": {
                    "capture": True,
                    "retrieve": True,
                    "background_maintenance": self.maintenance.health()["scheduler_running"],
                },
                "maintenance": self.maintenance.health(),
                "graph": self._graph_status(),
                "cognition_provider": type(self.engine.llm).__name__,
                "provider_state": {"cognition": dict(self._cognition_probe)},
            }
        raise ValueError(f"Unsupported memory operation: {operation}")

    def _graph_status(self) -> dict[str, Any]:
        reader = getattr(self.engine.store, "graph_derivation_status", None)
        state = reader() if callable(reader) else {"ready": False, "status": "unavailable"}
        return {"enabled": self.engine.config.modules.spreading, **state}
