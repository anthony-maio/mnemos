"""Two-pass development harness; no labels, production writes or implicit retries."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import time
from typing import Any

from mnemos.support_intersection import SnapshotFence, SupportSnapshot
from .relation_inventory_cache_v2 import RelationInventoryCache
from .relation_query_v2 import interpret_query
from .source_qualification import QualificationIdentity
from .source_relation_producer_v2 import prepare_source, produce_source


@dataclass(frozen=True)
class InventoryCase:
    key: str
    snapshot: SupportSnapshot


def prepare_run(cases: Sequence[InventoryCase], source_identity: QualificationIdentity,
                query_identity: QualificationIdentity) -> dict[str, Any]:
    """Validate every source request before the first provider call."""
    if not isinstance(source_identity, QualificationIdentity) or not isinstance(query_identity, QualificationIdentity):
        raise ValueError('Exact producer identities required')
    keys = [case.key for case in cases]
    if not keys or len(keys) > 200 or keys != sorted(set(keys)) or any(not k.strip() for k in keys):
        raise ValueError('Expected 1-200 unique case keys in lexical order')
    requests = {}
    for case in cases:
        snapshot = case.snapshot
        if not snapshot.query.strip() or not snapshot.fingerprint or type(snapshot.reference_date) is not date:
            raise ValueError('Incomplete query snapshot')
        ids = [c.fact_id for c in snapshot.candidates]
        if len(ids) != len(set(ids)) or len(ids) > 20:
            raise ValueError('Duplicate or excessive candidate cohort')
        for candidate in snapshot.candidates:
            if len(candidate.evidence) > 8:
                raise ValueError('Excessive source cohort')
            for evidence in candidate.evidence:
                prompt, key = prepare_source(candidate, evidence.evidence_id, source_identity)
                requests[key] = prompt
    if len(requests) > 256:
        raise ValueError('Run would exceed frozen cache capacity')
    payload = dict(cases=[asdict(c) for c in cases], source_identity=asdict(source_identity),
                   query_identity=asdict(query_identity), source_requests=requests,
                   passes=['first', 'reuse'], cache_capacity=256)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')
    return dict(manifest_sha256=hashlib.sha256(encoded).hexdigest(), **payload)


async def run_inventory_screen(
    cases: Sequence[InventoryCase], source_identity: QualificationIdentity,
    query_identity: QualificationIdentity,
    transport: Callable[[str, str], Awaitable[dict[str, Any]]],
    emit: Callable[[str, dict[str, Any]], None], fence: SnapshotFence,
) -> dict[str, Any]:
    """Owner freezes runtime/token limits and durably appends emitted records.

    Transport receives stage and exact unrendered prompt, and returns the full
    reply with done=True and response text. Owner records any rendering and wire
    metadata. An uncertain completion stops both passes; no requests are retried.
    Timers include awaited source work but exclude initial retrieval and startup.
    """
    manifest = prepare_run(cases, source_identity, query_identity)
    fence()
    emit('manifest', manifest)
    cache = RelationInventoryCache(max_records=256)
    attempted_sources = set()
    attempted_queries = []
    stopped = False
    for phase in ('first', 'reuse'):
        for case in cases:
            fence()
            started = time.perf_counter_ns()
            uncertain = False
            journal_failed = None
            emit('case_start', dict(phase=phase, key=case.key))

            async def provider(stage: str, prompt: str) -> str:
                nonlocal uncertain, journal_failed
                completed = False
                request_sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                base = dict(phase=phase, key=case.key, stage=stage, request_sha256=request_sha)
                def record(kind, data):
                    nonlocal journal_failed
                    try:
                        emit(kind, data)
                    except BaseException as exc:
                        journal_failed = exc
                        raise
                record('request_start', dict(**base, prompt=prompt))
                try:
                    wire = await transport(stage, prompt)
                    completed = isinstance(wire, dict) and wire.get('done') is True
                    record('provider_reply', dict(**base, reply=wire))
                    if not completed or not isinstance(wire.get('response'), str):
                        raise ValueError('Provider did not confirm a complete text response')
                    return wire['response']
                except BaseException as exc:
                    uncertain = not completed
                    record('provider_error', dict(**base, error_type=type(exc).__name__,
                                                  completion_uncertain=uncertain))
                    raise

            if phase == 'first':
                for candidate in case.snapshot.candidates:
                    for evidence in candidate.evidence:
                        _, source_key = prepare_source(candidate, evidence.evidence_id, source_identity)
                        if source_key in attempted_sources:
                            continue
                        attempted_sources.add(source_key)
                        source_start = time.perf_counter_ns()
                        result = await produce_source(candidate, evidence.evidence_id, source_identity,
                            cache, lambda prompt: provider('source', prompt), fence)
                        if journal_failed is not None:
                            raise journal_failed
                        emit('source_observation', dict(phase=phase, key=case.key, source_key=source_key,
                            elapsed_ms=(time.perf_counter_ns() - source_start) / 1_000_000, **asdict(result)))
                        if uncertain:
                            break
                    if uncertain:
                        break
            if not uncertain:
                query_start = time.perf_counter_ns()
                result = await interpret_query(case.snapshot, source_identity, query_identity, cache,
                    lambda prompt: provider('query', prompt), fence)
                if journal_failed is not None:
                    raise journal_failed
                attempted_queries.append((phase, case.key))
                emit('query_observation', dict(phase=phase, key=case.key, **asdict(result),
                    query_elapsed_ms=(time.perf_counter_ns() - query_start) / 1_000_000,
                    case_elapsed_ms=(time.perf_counter_ns() - started) / 1_000_000))
            emit('case_end', dict(phase=phase, key=case.key, completion_uncertain=uncertain,
                                 elapsed_ms=(time.perf_counter_ns() - started) / 1_000_000))
            if uncertain:
                stopped = True
                break
            fence()
        if stopped:
            break
    summary = dict(status='stopped_uncertain_completion' if stopped else 'completed',
        manifest_sha256=manifest['manifest_sha256'], attempted_sources=sorted(attempted_sources),
        attempted_queries=attempted_queries,
        unattempted_queries=[(phase, c.key) for phase in ('first', 'reuse') for c in cases
                             if (phase, c.key) not in attempted_queries])
    emit('summary', summary)
    return summary
