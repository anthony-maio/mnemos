"""Adapters for the fixed QA-v2 and contrast-v5 development components."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from mnemos.answer_support import AnswerSupportConfig, _snapshot, assess_answer_support
from mnemos.judgment import Judge, JudgmentContractError
from mnemos.scopes import ScopeContext, ScopeRegistry
from mnemos.support_intersection import (
    ComponentSelection,
    SelectedEvidence,
    SnapshotFence,
    SupportSnapshot,
)
from mnemos.utils.storage import MemoryStore

from .extractive_support import decode_source_span, span_matches_fact


@dataclass(frozen=True)
class QAOutput:
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    sequence_ids: tuple[int | None, ...]
    attention_mask: tuple[int, ...]
    start_logits: tuple[float, ...]
    end_logits: tuple[float, ...]
    cls_index: int
    question_token_count: int


QAPredictor = Callable[[str, str], Awaitable[QAOutput]]


class ExtractiveSourceSelector:
    """Decode every retained excerpt with the original corrected QA policy.

    The predictor owns the pinned local model and must tokenize without truncation.
    It must validate the 64-question-token and 512-total-token limits before model
    execution; this adapter checks the returned token arrays too. Raw observations
    can be recorded before the composition performs its current-store fence.
    """

    def __init__(
        self,
        predict: QAPredictor,
        observe: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.predict = predict
        self.observe = observe

    async def __call__(
        self, snapshot: SupportSnapshot, revalidate: SnapshotFence
    ) -> ComponentSelection:
        selected = []
        prefix = (
            f"Reference date: {snapshot.reference_date.isoformat()}\nRetained source excerpt:\n"
        )
        for candidate in snapshot.candidates:
            witnesses = []
            for evidence in candidate.evidence:
                context = prefix + evidence.excerpt
                record: dict[str, Any] = {
                    "snapshot_fingerprint": snapshot.fingerprint,
                    "fact_id": candidate.fact_id,
                    "evidence_id": evidence.evidence_id,
                    "query": snapshot.query,
                    "context": context,
                    "source_region": (len(prefix), len(context)),
                }
                try:
                    revalidate()
                    output = await self.predict(snapshot.query, context)
                    revalidate()
                    if not isinstance(output, QAOutput):
                        raise JudgmentContractError("QA returned an invalid response type")
                    record["raw_output"] = asdict(output)
                    if (
                        type(output.question_token_count) is not int
                        or not 0 < output.question_token_count <= 64
                        or not 0 < len(output.input_ids) <= 512
                        or len(output.input_ids) != len(output.offsets)
                        or any(type(value) is not int or value < 0 for value in output.input_ids)
                    ):
                        raise JudgmentContractError(
                            "QA returned malformed or oversized token input"
                        )
                    try:
                        decision = decode_source_span(
                            context,
                            (len(prefix), len(context)),
                            output.offsets,
                            output.sequence_ids,
                            output.attention_mask,
                            output.start_logits,
                            output.end_logits,
                            cls_index=output.cls_index,
                            max_answer_tokens=30,
                            null_margin=0.0,
                            clip_boundary_whitespace=True,
                        )
                    except ValueError as exc:
                        raise JudgmentContractError("QA returned invalid span data") from exc
                    matches = span_matches_fact(decision, candidate.content)
                    record.update(
                        status="assessed", decision=asdict(decision), literal_fact_match=matches
                    )
                    if matches:
                        witnesses.append(evidence.evidence_id)
                except Exception as exc:
                    record.update(status="unavailable", error_type=type(exc).__name__)
                    raise
                finally:
                    if self.observe is not None:
                        self.observe(record)
            if witnesses:
                selected.append(SelectedEvidence(candidate.fact_id, tuple(witnesses)))
        return ComponentSelection("assessed", snapshot.fingerprint, tuple(selected))


class ContrastSourceSelector:
    """Use the existing contrast-v5 prompts and full-cohort revalidation."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        registry: ScopeRegistry,
        context: ScopeContext,
        judge: Judge,
        allowed_scopes: Sequence[str] = ("project", "global"),
    ) -> None:
        self.store = store
        self.registry = registry
        self.context = context.model_copy(deep=True)
        self.judge = judge
        self.allowed_scopes = tuple(allowed_scopes)

    async def __call__(
        self, snapshot: SupportSnapshot, revalidate: SnapshotFence
    ) -> ComponentSelection:
        revalidate()
        config = AnswerSupportConfig(protocol="contrast_v5", batching="all")
        for candidate in snapshot.candidates:
            current, _ = _snapshot(
                self.store, candidate.fact_id, self.context, self.allowed_scopes, config
            )
            if current is None or current.fingerprint != candidate.fingerprint:
                raise JudgmentContractError("Contrast input differs from the shared snapshot")
        result = await assess_answer_support(
            snapshot.query,
            tuple(item.fact_id for item in snapshot.candidates),
            store=self.store,
            registry=self.registry,
            context=self.context,
            judge=self.judge,
            allowed_scopes=self.allowed_scopes,
            config=config,
            reference_date=snapshot.reference_date,
        )
        revalidate()
        if result.status != "assessed":
            return ComponentSelection(
                "unavailable", snapshot.fingerprint, error_type=result.error_type
            )
        expected = {item.fact_id: item.fingerprint for item in snapshot.candidates}
        if any(expected.get(item.fact_id) != item.fingerprint for item in result.supported):
            raise JudgmentContractError("Contrast output differs from the shared snapshot")
        if any(
            item.disposition in {"ineligible", "unassessed", "stale"} for item in result.decisions
        ):
            raise JudgmentContractError("Contrast eligibility differs from the shared snapshot")
        return ComponentSelection(
            "assessed",
            snapshot.fingerprint,
            tuple(
                SelectedEvidence(item.fact_id, item.evidence_ids)
                for item in result.decisions
                if item.disposition == "supported"
            ),
        )
