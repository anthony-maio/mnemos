"""
mnemos/modules/mutable_rag.py — Mutable RAG (Memory Reconsolidation).

Neuroscience Basis:
    Human memory is not read-only. Every time a memory is retrieved, it enters
    a 'labile' (unstable) state — it becomes open to modification before being
    re-stabilized (reconsolidated) into long-term storage (Nader, Schafe &
    LeDoux, 2000; Misanin et al., 1968).

    This lability is not a bug — it's a feature. It allows memories to be
    updated with current information, preventing the accumulation of outdated
    facts. The memory trace is physically altered each time it is recalled.

    Key insight: In humans, the very act of remembering modifies what is remembered.

Mnemos Implementation:
    Retrieval records access metadata only. A later review must cite durable
    same-scope source events and a typed expected-version proposal before the
    LLM can produce a correction.

    This creates a *self-healing* database that naturally adapts to new information
    without accumulating contradictory entries — solving the classic RAG problem
    of stale facts competing with fresh ones in the context window.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..config import JudgmentConfig, MutableRAGConfig
from ..contracts import FactProposal
from ..errors import VersionConflict
from ..jobs import DurableJob, FactEvidence, FactRevision
from ..judgment import (
    Choice,
    ChoiceAnswer,
    Criteria,
    Description,
    Judge,
    Noul,
    NoulAnswer,
    UnavailableJudge,
)
from ..memory_safety import MemoryWriteFirewall
from ..types import EpisodicEvent, MemoryChunk
from ..utils.embeddings import EmbeddingProvider, embed_text_async
from ..utils.llm import LLMProvider
from ..utils.storage import MemoryStore, compute_canonical_key

_MAX_REVISION_HISTORY = 10


class MutableRAG:
    """
    Memory reconsolidation module — makes the vector database self-healing.

    Retrieval does not provide correcting evidence. A reconsolidation pass can
    evaluate a chunk only after explicit source events have been validated.

    Args:
        llm: LLM provider used for staleness evaluation.
        embedder: Embedding provider for re-embedding updated chunks.
        store: Memory store containing the chunks to maintain.
        config: MutableRAGConfig controlling reconsolidation behavior.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        store: MemoryStore,
        config: MutableRAGConfig | None = None,
        write_firewall: MemoryWriteFirewall | None = None,
        judge: Judge | None = None,
        judgment_config: JudgmentConfig | None = None,
    ) -> None:
        self._llm = llm
        self._judge: Judge = judge if judge is not None else UnavailableJudge()
        self._judgment_config = judgment_config or JudgmentConfig()
        self._embedder = embedder
        self._store = store
        self._config = config or MutableRAGConfig()
        self._write_firewall = write_firewall or MemoryWriteFirewall()
        # Labile chunk queue: chunk_id → (chunk, explicit source event IDs).
        self._labile_chunks: dict[str, tuple[MemoryChunk, list[str]]] = {}
        # Last reconsolidation timestamp per chunk (seconds since epoch)
        self._last_reconsolidation_at: dict[str, float] = {}
        # Stats
        self._total_retrieved: int = 0
        self._total_reconsolidated: int = 0
        self._total_unchanged: int = 0

    def _in_cooldown(self, chunk_id: str) -> bool:
        cooldown = self._config.reconsolidation_cooldown_seconds
        if cooldown <= 0:
            return False
        last = self._last_reconsolidation_at.get(chunk_id)
        if last is None:
            return False
        return (time.time() - last) < cooldown

    def _mark_reconsolidated_now(self, chunk_id: str) -> None:
        self._last_reconsolidation_at[chunk_id] = time.time()

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        current_context: str = "",
    ) -> list[MemoryChunk]:
        """
        Retrieve chunks by similarity and update access metadata.

        The query is deliberately not queued as correcting evidence. Use
        ``review`` or ``mark_labile(..., evidence_refs=...)`` for an explicit
        source-backed review.

        Args:
            query_embedding: Embedding of the current query.
            top_k: Maximum number of chunks to return.
            current_context: Retained for compatibility and ignored for writes.

        Returns:
            List of top-k MemoryChunks by cosine similarity.
        """
        chunks = self._store.retrieve(query_embedding, top_k=top_k)
        self._total_retrieved += len(chunks)
        now = datetime.now(timezone.utc)

        # Increment access counts
        for chunk in chunks:
            if self._store.mark_retrieved(chunk.id, retrieved_at=now, reconsolidation_due_at=now):
                refreshed = self._store.get(chunk.id)
                if refreshed is not None:
                    chunk.access_count = refreshed.access_count
                    chunk.updated_at = refreshed.updated_at
                    chunk.last_retrieved_at = refreshed.last_retrieved_at
                    chunk.reconsolidation_due_at = refreshed.reconsolidation_due_at

        return chunks

    def mark_labile(
        self,
        chunk: MemoryChunk,
        context: str = "",
        *,
        evidence_refs: Sequence[str] | None = None,
    ) -> None:
        """
        Flag a chunk as labile for future source-backed reconsolidation.

        Called by MnemosEngine after retrieval to queue chunks for
        staleness evaluation without accessing private internals.

        Args:
            chunk: The retrieved chunk to mark as labile.
            context: Retained for compatibility and ignored as evidence.
        """
        del context  # Legacy context is never treated as correcting evidence.
        refs = [str(item) for item in (evidence_refs or []) if str(item)]
        if not refs:
            return
        self._validate_evidence(chunk, refs)
        if self._config.enabled and not self._in_cooldown(chunk.id):
            self._labile_chunks[chunk.id] = (chunk, refs)

    def _validate_evidence(
        self, chunk: MemoryChunk, evidence_refs: Sequence[str]
    ) -> list[EpisodicEvent]:
        """Resolve source events and require an exact profile and scope match."""
        events: list[EpisodicEvent] = []
        chunk_profile = chunk.metadata.get("user_profile_id")
        chunk_scope = chunk.metadata.get("scope", "global")
        chunk_scope_id = chunk.metadata.get("scope_id")
        seen: set[str] = set()
        for ref in evidence_refs:
            event_id = str(ref)
            if event_id in seen:
                continue
            seen.add(event_id)
            event = self._store.get_episodic_event(event_id)
            if event is None:
                raise ValueError(f"Evidence event does not exist: {event_id}")
            event_profile = event.metadata.get("user_profile_id")
            event_scope = event.metadata.get("scope", event.scope)
            event_scope_id = event.metadata.get("scope_id", event.scope_id)
            if not chunk_profile or not event_profile or event_profile != chunk_profile:
                raise ValueError(f"Evidence event is outside the fact profile: {event_id}")
            if event_scope != chunk_scope or event_scope_id != chunk_scope_id:
                raise ValueError(f"Evidence event is outside the fact scope: {event_id}")
            events.append(event)
        if not events:
            raise ValueError("At least one explicit evidence event is required")
        return events

    def _validate_proposal(
        self,
        proposal: FactProposal,
        *,
        evidence_refs: Sequence[str] | None = None,
    ) -> tuple[MemoryChunk, list[EpisodicEvent]]:
        if proposal.operation != "correct" or not proposal.target_fact_id:
            raise ValueError("Mutable review requires a typed correction proposal")
        refs = list(evidence_refs or proposal.evidence_ids)
        if set(refs) != set(proposal.evidence_ids):
            raise ValueError("Evidence references must match the typed proposal")
        target = self._store.get(proposal.target_fact_id)
        if target is None:
            raise ValueError("Correction target does not exist")
        if proposal.expected_version is None:
            raise ValueError("Correction proposal requires an expected version")
        if target.version != proposal.expected_version:
            raise VersionConflict("Correction version conflict")
        return target, self._validate_evidence(target, refs)

    async def review(
        self,
        proposal: FactProposal,
        *,
        evidence_refs: Sequence[str] | None = None,
    ) -> tuple[MemoryChunk, bool]:
        """Validate and persist one source-backed typed correction proposal."""
        target, _events = self._validate_proposal(proposal, evidence_refs=evidence_refs)
        refs = list(evidence_refs or proposal.evidence_ids)
        job = DurableJob(
            kind="reconsolidate_review",
            payload={
                "content": proposal.content,
                "evidence_ids": refs,
                "operation": proposal.operation,
                "target_fact_id": target.id,
                "expected_version": proposal.expected_version,
            },
        )
        if self._store.__class__.commit_fact_and_job is MemoryStore.commit_fact_and_job:
            return await self._apply_review_job(job)

        self._store.enqueue_job(job)
        owner = f"mutable-rag:{id(self)}"
        claimed = self._store.claim_jobs(owner=owner, job_id=job.id, limit=1)
        if not claimed:
            raise RuntimeError("Mutable review job could not be claimed")
        lease = claimed[0]
        try:
            return await self._apply_review_job(lease)
        except Exception as exc:
            self._store.fail_job(
                lease.id,
                owner=owner,
                lease_token=lease.lease_token,
                error=str(exc),
            )
            raise

    async def process_job(self, job: DurableJob) -> tuple[MemoryChunk, bool]:
        """Resume a claimed source-backed review job after restart."""
        if job.kind != "reconsolidate_review":
            raise ValueError(f"Unsupported MutableRAG job kind: {job.kind}")
        return await self._apply_review_job(job)

    async def _apply_review_job(self, job: DurableJob) -> tuple[MemoryChunk, bool]:
        payload = job.payload
        proposal = FactProposal.model_validate(
            {
                "content": payload.get("content"),
                "evidence_ids": payload.get("evidence_ids"),
                "operation": payload.get("operation"),
                "target_fact_id": payload.get("target_fact_id"),
                "expected_version": payload.get("expected_version"),
            }
        )
        target, events = self._validate_proposal(proposal)
        safe = self._write_firewall.apply(proposal.content)
        if not safe.allowed:
            raise ValueError(safe.reason or "Correction violates write policy")
        if safe.content == target.content:
            if job.lease_owner and job.lease_token:
                if not self._store.complete_job(
                    job.id,
                    owner=job.lease_owner,
                    lease_token=job.lease_token,
                    result={"decision": "unchanged"},
                ):
                    raise RuntimeError("Mutable review lease expired")
            self._total_unchanged += 1
            self._mark_reconsolidated_now(target.id)
            return target, False

        updated = target.reconsolidate(safe.content)
        updated.embedding = await embed_text_async(self._embedder, safe.content, role="document")
        updated.canonical_key = compute_canonical_key(
            safe.content,
            str(updated.metadata.get("scope", "global")),
            updated.metadata.get("scope_id"),
            str(updated.metadata["user_profile_id"]),
        )
        updated.metadata["reconsolidation_context"] = "source-backed review"
        updated.metadata["source_event_ids"] = [event.id for event in events]
        updated.metadata["safety_redactions"] = [match.label for match in safe.matches]
        history = updated.metadata.get("revision_history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "from_version": target.version,
                "to_version": updated.version,
                "previous_content": target.content,
                "new_content": safe.content,
                "context": "source-backed review",
                "changed_at": updated.updated_at.isoformat(),
            }
        )
        updated.metadata["revision_history"] = history[-_MAX_REVISION_HISTORY:]
        evidence = [
            FactEvidence(
                fact_id=updated.id,
                source_event_id=event.id,
                excerpt=event.content[:1000],
                source_ref=event.metadata.get("source_ref"),
            )
            for event in events
        ]
        revision = FactRevision(
            fact_id=updated.id,
            version=target.version,
            content=target.content,
            reason="Before explicit source-backed correction",
        )
        if job.lease_owner and job.lease_token:
            committed = self._store.commit_fact_and_job(
                job.id,
                owner=job.lease_owner,
                lease_token=job.lease_token,
                chunks=[updated],
                evidence=evidence,
                revisions=[revision],
                expected_version=target.version,
                result={"decision": "corrected"},
            )
            if not committed:
                raise RuntimeError("Mutable review lease expired before commit")
        else:
            current = self._store.get(target.id)
            if current is None or current.version != target.version:
                raise VersionConflict("Correction version conflict")
            self._store.update(target.id, updated)

        self._total_reconsolidated += 1
        self._mark_reconsolidated_now(target.id)
        # Atomic canonical storage can absorb captured evidence and enrich the
        # committed metadata. Return that row, not the pre-commit proposal.
        persisted = self._store.get(target.id)
        if persisted is None:
            raise RuntimeError("Corrected fact is unavailable after commit")
        return persisted, True

    def _build_reconsolidation_prompt(self, chunk: MemoryChunk, context: str) -> str:
        """
        Build the LLM prompt for evaluating whether a chunk needs updating.

        Args:
            chunk: The memory chunk to evaluate.
            context: Current conversation context.

        Returns:
            Formatted prompt string.
        """
        return self._config.staleness_check_prompt.format(
            content=chunk.content,
            context=context if context else "(No new context available)",
        )

    async def reconsolidate(
        self,
        chunk: MemoryChunk,
        new_context: str,
        *,
        evidence_refs: Sequence[str] | None = None,
        raise_on_error: bool = False,
    ) -> tuple[MemoryChunk, bool]:
        """
        Evaluate a chunk for staleness and update if necessary.

        Asks the LLM about validated source events. A query alone cannot alter
        a fact, and an unchanged or malformed response performs no write.

        Args:
            chunk: The labile chunk to evaluate.
            new_context: Compatibility label; source event content supplies the review context.

        Returns:
            Tuple of (resulting_chunk, was_changed: bool).
        """
        refs = [str(item) for item in (evidence_refs or []) if str(item)]
        if not refs:
            # A query or free-form context is never a correction source.
            return chunk, False
        try:
            evidence = self._validate_evidence(chunk, refs)
        except Exception:
            if raise_on_error:
                raise
            return chunk, False
        if self._judge.provider != "unavailable":
            return await self._reconsolidate_with_judge(
                chunk, evidence, refs, raise_on_error=raise_on_error
            )

        # The review prompt receives source events, never the retrieval query.
        prompt = self._build_reconsolidation_prompt(
            chunk, "\n".join(event.content for event in evidence)
        )

        try:
            response = await self._llm.predict(prompt)
        except Exception:
            if raise_on_error:
                raise
            return chunk, False

        response = response.strip()

        if response.upper().startswith("UNCHANGED"):
            # An unchanged review must not make the fact appear newly observed.
            stable_chunk = chunk
            self._total_unchanged += 1
            self._mark_reconsolidated_now(chunk.id)
            return stable_chunk, False

        elif response.upper().startswith("CHANGED"):
            # Extract the updated content (after "CHANGED: ")
            if ":" in response:
                new_content = response.split(":", 1)[1].strip()
            else:
                # Response is just "CHANGED" without content -- do not modify.
                stable_chunk = chunk
                self._total_unchanged += 1
                self._mark_reconsolidated_now(chunk.id)
                return stable_chunk, False

            if not new_content:
                stable_chunk = chunk
                self._total_unchanged += 1
                self._mark_reconsolidated_now(chunk.id)
                return stable_chunk, False

            safe_content = new_content
            if self._write_firewall is not None:
                safety = self._write_firewall.apply(new_content)
                if not safety.allowed:
                    stable_chunk = chunk
                    self._total_unchanged += 1
                    self._mark_reconsolidated_now(chunk.id)
                    return stable_chunk, False
                safe_content = safety.content

            proposal = FactProposal(
                content=safe_content,
                evidence_ids=refs,
                operation="correct",
                target_fact_id=chunk.id,
                expected_version=chunk.version,
            )
            return await self.review(proposal, evidence_refs=refs)

        else:
            # Unexpected response format — conservative: leave unchanged
            stable_chunk = chunk
            self._total_unchanged += 1
            self._mark_reconsolidated_now(chunk.id)
            return stable_chunk, False

    async def _reconsolidate_with_judge(
        self,
        chunk: MemoryChunk,
        evidence: Sequence[EpisodicEvent],
        refs: list[str],
        *,
        raise_on_error: bool,
    ) -> tuple[MemoryChunk, bool]:
        """Source-bound reconsolidation: the judge picks the replacing span.

        No rewrite is generated. If the evidence shows the fact is outdated
        and one user-spoken unit states the current replacement, that unit's
        exact text becomes the correction; otherwise the fact stands.
        """
        from ..extraction.source_selection import source_units

        units: list[tuple[str, str, str]] = []  # (id, role, text)
        for position, event in enumerate(evidence):
            for unit in source_units(event.content):
                if unit.text.strip():
                    units.append((f"e{position}-u{unit.index}", event.role, unit.text.strip()))
        units = units[: self._judgment_config.max_units_per_request]
        state = {
            "stored_fact": {"text": chunk.content},
            "evidence": [{"id": uid, "role": role, "text": text} for uid, role, text in units],
        }
        replacement_options: list[tuple[str, Description]] = [
            ("none", "no evidence unit states a complete current replacement")
        ]
        replacement_options.extend(
            (uid, {"text": text}) for uid, role, text in units if role == "user"
        )
        questions: dict[str, Noul | Choice] = {
            "outdated": Noul(
                {
                    "stored_fact": chunk.content,
                    "question": (
                        "Does the evidence show stored_fact is no longer accurate, has been "
                        "replaced, or was explicitly withdrawn?"
                    ),
                },
                Criteria(
                    true=(
                        "evidence states a different current value or rule for the same "
                        "subject, or explicitly withdraws it"
                    ),
                    false=(
                        "evidence is unrelated, repeats the fact, asks a question, or is "
                        "hypothetical or quoted"
                    ),
                ),
            )
        }
        if len(replacement_options) > 1:
            questions["replacement"] = Choice(
                {
                    "stored_fact": chunk.content,
                    "question": (
                        "Which evidence unit states the complete current replacement for "
                        "stored_fact, spoken by the user?"
                    ),
                },
                tuple(replacement_options),
            )
        try:
            judgment = await self._judge.ask(state, questions, purpose="reconsolidate")
        except Exception:
            if raise_on_error:
                raise
            return chunk, False
        outdated = judgment.answers["outdated"]
        assert isinstance(outdated, NoulAnswer)
        replacement = judgment.answers.get("replacement")
        chosen = replacement.choice if isinstance(replacement, ChoiceAnswer) else "none"
        if (
            outdated.probability < self._judgment_config.outdated_min_probability
            or chosen == "none"
        ):
            self._total_unchanged += 1
            self._mark_reconsolidated_now(chunk.id)
            if outdated.probability >= self._judgment_config.outdated_min_probability:
                chunk.metadata["reconsolidation_status"] = "outdated_without_replacement"
            return chunk, False
        new_content = next(text for uid, _, text in units if uid == chosen)
        safety = self._write_firewall.apply(new_content)
        if not safety.allowed or safety.content != new_content:
            self._total_unchanged += 1
            self._mark_reconsolidated_now(chunk.id)
            return chunk, False
        proposal = FactProposal(
            content=new_content,
            evidence_ids=refs,
            operation="correct",
            target_fact_id=chunk.id,
            expected_version=chunk.version,
        )
        return await self.review(proposal, evidence_refs=refs)

    async def process_labile_chunks(
        self, max_chunks: int | None = None
    ) -> list[tuple[MemoryChunk, bool]]:
        """
        Batch reconsolidate all queued labile chunks.

        This is designed to run asynchronously after a conversation turn,
        not blocking the main response pipeline — mirroring how biological
        reconsolidation occurs in the background after retrieval.

        Args:
            max_chunks: Maximum number of labile chunks to process in this pass.
                        Defaults to config.max_labile_chunks.

        Returns:
            List of (chunk, was_changed) tuples for each processed chunk.
        """
        limit = max_chunks or self._config.max_labile_chunks

        if not self._labile_chunks:
            pending_jobs = self._store.list_reconsolidation_jobs(status="pending", limit=limit)
            if not pending_jobs:
                return []
            results: list[tuple[MemoryChunk, bool]] = []
            for job in pending_jobs:
                chunk = self._store.get(job.chunk_id)
                if chunk is None:
                    self._store.delete_reconsolidation_job(job.chunk_id)
                    continue
                # Historical queue rows contain only retrieval context and cannot
                # authorize a write. Keep the row visible for migration review.
                self._store.update_reconsolidation_job(
                    job.chunk_id,
                    status="failed",
                    error="Explicit source evidence is required for mutable review",
                )
            return results

        to_process = list(self._labile_chunks.items())[:limit]

        results = []
        for chunk_id, (chunk, evidence_refs) in to_process:
            result = await self.reconsolidate(chunk, "", evidence_refs=evidence_refs)
            results.append(result)
            # Remove from labile queue after processing
            self._labile_chunks.pop(chunk_id, None)

        return results

    def clear_labile_queue(self) -> int:
        """
        Clear the labile chunk queue without reconsolidating.

        Returns:
            Number of chunks that were queued and discarded.
        """
        count = len(self._labile_chunks)
        self._labile_chunks.clear()
        return count

    def get_labile_count(self) -> int:
        """Return the number of chunks currently in the labile queue."""
        return len(self._labile_chunks)

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the MutableRAG module."""
        return {
            "module": "MutableRAG",
            "enabled": self._config.enabled,
            "total_retrieved": self._total_retrieved,
            "total_reconsolidated": self._total_reconsolidated,
            "total_unchanged": self._total_unchanged,
            "labile_queue_size": len(self._labile_chunks),
        }
