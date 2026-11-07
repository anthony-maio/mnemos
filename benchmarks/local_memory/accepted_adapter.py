"""Convert terminal gold state to accepted retrieval inputs without ingestion.

The legacy evaluator queries after all events. This adapter preserves that
phase and both state-pair wordings. Gold IDs and version 1 describe fixtures,
not measured runtime fact identities, versions or lifecycle outcomes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
from typing import Any, Literal

from mnemos.contracts import RetrievalRequest
from mnemos.jobs import FactEvidence
from mnemos.scopes import (
    ProjectIdentity,
    ScopeContext,
    ScopeRegistry,
    WorkspaceIdentity,
    context_metadata,
)
from mnemos.types import EpisodicEvent, MemoryChunk

from .accepted_corpus import (
    AcceptedCase,
    AcceptedCorpus,
    AcceptedQuery,
    CorpusOptions,
    validate_corpus,
)
from .accepted_retrieval import AcceptedSnapshot, _digest
from .contracts import (
    CONTRACT_V4_VERSION,
    CONTRACT_V5_VERSION,
    CONTRACT_V6_VERSION,
    FACT_STATUSES,
    SCENARIO_TAGS,
    HeldoutDataset,
    QueryRecord,
    SessionScenario,
    SourceEvent,
    StatePair,
    _parse_session,
    _validate_v3_lifecycle_targets,
    _validate_v4_contract,
    load_dataset,
    validate_dataset,
)

BASE_PROFILE = "t18-synthetic-profile"


@dataclass(frozen=True)
class AdaptedAcceptedCorpus:
    corpus: AcceptedCorpus
    provenance: dict[str, Any]


def _context(
    session: SessionScenario, target: SourceEvent | QueryRecord | StatePair
) -> ScopeContext:
    return ScopeContext.model_validate(
        {
            "user_profile_id": target.target_user_profile_id or BASE_PROFILE,
            "project_id": target.target_project_id or session.project_id,
            "workspace_id": target.target_workspace_id,
            "session_id": session.session_id,
            "host": session.host,
        }
    )


def _registry(sessions: tuple[SessionScenario, ...]) -> ScopeRegistry:
    projects: dict[str, str] = {}
    workspaces: dict[str, str] = {}
    memberships: dict[str, set[str]] = defaultdict(set)

    def bind(values: dict[str, str], identity: str, profile: str) -> None:
        if identity in values and values[identity] != profile:
            raise ValueError("A declared scope identity has conflicting profiles")
        values[identity] = profile

    for session in sessions:
        bind(projects, session.project_id, BASE_PROFILE)
        targets: tuple[SourceEvent | QueryRecord | StatePair, ...] = (
            *session.events,
            *session.queries,
            *session.state_pairs,
        )
        for target in targets:
            context = _context(session, target)
            assert context.project_id is not None
            bind(projects, context.project_id, context.user_profile_id)
            if context.workspace_id is not None:
                bind(workspaces, context.workspace_id, context.user_profile_id)
                memberships[context.project_id].add(context.workspace_id)
    return ScopeRegistry(
        profiles=sorted(set(projects.values()) | set(workspaces.values())),
        projects={
            key: ProjectIdentity(
                id=key,
                root="fixture:" + key,
                user_profile_id=value,
                enabled=True,
                workspace_ids=sorted(memberships[key]),
            )
            for key, value in projects.items()
        },
        workspaces={
            key: WorkspaceIdentity(id=key, name=key, user_profile_id=value)
            for key, value in workspaces.items()
        },
    )


def _validate_session(session: SessionScenario, version: str) -> None:
    # Reuse the original parsing and lifecycle guards without weakening the
    # full-dataset minimums to accommodate a small development fixture.
    if _parse_session(json.loads(json.dumps(asdict(session))), 0) != session:
        raise ValueError("Session must already use the canonical source contract")
    for label, values in (
        ("facts", [f.fact_id for f in session.facts]),
        ("events", [e.event_id for e in session.events]),
        ("queries", [q.query_id for q in session.queries]),
        ("pairs", [p.pair_id for p in session.state_pairs]),
    ):
        if len(set(values)) != len(values):
            raise ValueError("Duplicate " + label)
    if (
        not session.events
        or session.human_acceptance != "not_run"
        or session.source_label != "synthetic_seed"
    ):
        raise ValueError("Source events and the original unreviewed label are required")
    if (
        len(set(session.scenario_types)) != len(session.scenario_types)
        or set(session.scenario_types) - SCENARIO_TAGS
    ):
        raise ValueError("Scenario tags must preserve the original contract")
    if not any(e.lifecycle_operation == "restart" for e in session.events):
        raise ValueError("A source session needs its declared restart boundary")
    times = [datetime.fromisoformat(e.occurred_at.replace("Z", "+00:00")) for e in session.events]
    if any(t.tzinfo is None or t.utcoffset() is None for t in times):
        raise ValueError("Source timestamps must be timezone-aware")
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("Source events must be strictly ordered")
    facts = {f.fact_id: f for f in session.facts}
    events = {e.event_id: e for e in session.events}
    for fact in session.facts:
        if fact.gold_status not in FACT_STATUSES or fact.scope not in {
            "project",
            "workspace",
            "global",
        }:
            raise ValueError("Unknown fact status or scope")
        if not fact.evidence_event_ids or len(set(fact.evidence_event_ids)) != len(
            fact.evidence_event_ids
        ):
            raise ValueError("Fact evidence IDs must be nonempty and unique")
        if not set(fact.evidence_event_ids) <= set(events):
            raise ValueError("Fact evidence references a missing event")
        if fact.gold_status in {"supported", "conflict"} and any(
            fact.fact_id not in events[key].evidence_fact_ids for key in fact.evidence_event_ids
        ):
            raise ValueError("Retained fact and event evidence links must agree")
        if fact.gold_status == "supported" and not any(
            events[key].role == "user" and events[key].write_mode in {"automatic", "explicit"}
            for key in fact.evidence_event_ids
        ):
            raise ValueError("Supported gold facts need user evidence")
    for event in session.events:
        if not set(event.evidence_fact_ids) <= set(facts):
            raise ValueError("An event references a missing fact")
    _validate_v3_lifecycle_targets(session, events, facts)
    _validate_v4_contract(
        session, events, facts, check_repaired_canaries=version == CONTRACT_V4_VERSION
    )
    for query in session.queries:
        if bool(query.relevant_fact_ids) != (query.kind == "answerable"):
            raise ValueError("Query kind and labels disagree")
    for pair in session.state_pairs:
        if pair.semantic_query == pair.state_query:
            raise ValueError("Legacy state-pair wordings must remain distinct")


def _payload(dataset: HeldoutDataset) -> dict[str, Any]:
    return {
        "version": dataset.version,
        "sessions": [asdict(s) for s in dataset.sessions],
        "development_session_ids": dataset.development_session_ids,
        "manifest": dataset.manifest,
    }


def adapt_terminal_gold(
    dataset: HeldoutDataset,
    *,
    split: Literal["development", "heldout"],
    query_phase: Literal["after_all_events"],
    options: CorpusOptions | None = None,
    allowed_scopes: tuple[Literal["project", "workspace", "global"], ...] = ("project", "global"),
) -> AdaptedAcceptedCorpus:
    """Preserve the legacy post-event query phase; never infer earlier states."""
    options = options or CorpusOptions()
    if (
        dataset.manifest.get("source_label") != "synthetic_seed"
        or dataset.manifest.get("human_acceptance") != "not_run"
        or dataset.manifest.get("advertised_as_real") is not False
    ):
        raise ValueError("Source manifest must preserve synthetic, unreviewed provenance")
    if dataset.version not in {CONTRACT_V4_VERSION, CONTRACT_V5_VERSION, CONTRACT_V6_VERSION}:
        raise ValueError("Adapter requires the explicit v4, v5 or v6 source contract")
    if query_phase != "after_all_events" or split not in {"development", "heldout"}:
        raise ValueError("Choose an explicit split and the supported post-event query phase")
    if not allowed_scopes or len(set(allowed_scopes)) != len(allowed_scopes):
        raise ValueError("Allowed scopes must be explicit and unique")
    if len(dataset.digest) != 64 or any(ch not in "0123456789abcdef" for ch in dataset.digest):
        raise ValueError("Source digest must be a canonical SHA-256")
    file_sha256 = None
    if dataset.source_path is not None:
        reloaded = load_dataset(dataset.source_path)
        if dataset.digest != reloaded.digest or _payload(dataset) != _payload(reloaded):
            raise ValueError("Source dataset object differs from its bound file")
        file_sha256 = hashlib.sha256(dataset.source_path.read_bytes()).hexdigest()
    if split == "heldout":
        if file_sha256 is None:
            raise ValueError("Held-out conversion requires a validated source file")
        validate_dataset(dataset)
    sessions = tuple(session for session in dataset.sessions if session.split == split)
    if not sessions or len({s.session_id for s in sessions}) != len(sessions):
        raise ValueError("Selected split must contain unique sessions")
    if set(dataset.development_session_ids) != {
        s.session_id for s in dataset.sessions if s.split == "development"
    }:
        raise ValueError("Development membership differs from the source declaration")
    for session in sessions:
        _validate_session(session, dataset.version)
        if dataset.version != CONTRACT_V6_VERSION and any(
            e.lifecycle_operation == "replay" for e in session.events
        ):
            raise ValueError("Original source replay requires the v6 contract")
    identities = [[fact.fact_id for session in sessions for fact in session.facts]]
    if dataset.version in {CONTRACT_V5_VERSION, CONTRACT_V6_VERSION}:
        identities.extend(
            [
                [event.event_id for session in sessions for event in session.events],
                [query.query_id for session in sessions for query in session.queries],
                [pair.pair_id for session in sessions for pair in session.state_pairs],
            ]
        )
    if any(len(values) != len(set(values)) for values in identities):
        raise ValueError("Source identities must be unique across selected sessions")
    registry = _registry(sessions)
    cases = []
    dispositions: Counter[str] = Counter()
    source_events, retained_events = 0, 0
    for session in sessions:
        events = {e.event_id: e for e in session.events}
        cutoff = datetime.fromisoformat(session.events[-1].occurred_at.replace("Z", "+00:00"))
        query_time = cutoff + timedelta(microseconds=1)
        facts, evidence = [], []
        sources: dict[str, EpisodicEvent] = {}
        for fact in session.facts:
            dispositions[fact.gold_status] += 1
            if fact.gold_status not in {"supported", "conflict"}:
                continue
            contexts = [_context(session, events[key]) for key in fact.evidence_event_ids]
            metadata = [context_metadata(context, fact.scope) for context in contexts]
            partitions = {(m["user_profile_id"], m["scope"], m["scope_id"]) for m in metadata}
            if len(partitions) != 1:
                raise ValueError("A fact's retained evidence has ambiguous scope identity")
            times = []
            for event_id, source_metadata in zip(fact.evidence_event_ids, metadata):
                event = events[event_id]
                if event.lifecycle_operation == "replay":
                    raise ValueError(
                        "Gold evidence must cite the original source, not a replay delivery"
                    )
                at = datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))
                times.append(at)
                source = EpisodicEvent(
                    id=event.event_id,
                    event_fingerprint=_digest([dataset.digest, session.session_id, event.event_id]),
                    role=event.role,
                    content=event.content,
                    normalized_content=event.content,
                    scope=fact.scope,
                    scope_id=source_metadata["scope_id"],
                    source="declared_terminal_gold",
                    status="consolidated",
                    metadata=source_metadata,
                    occurred_at=at,
                    created_at=at,
                )
                if event_id in sources and sources[event_id] != source:
                    raise ValueError(
                        "One source event cannot be assigned multiple scope partitions"
                    )
                sources[event_id] = source
                evidence.append(
                    FactEvidence(
                        evidence_id=_digest([fact.fact_id, event_id]),
                        fact_id=fact.fact_id,
                        source_event_id=event_id,
                        excerpt=event.content,
                        source_ref="fixture:" + session.session_id + ":" + event_id,
                        created_at=at,
                    )
                )
            facts.append(
                MemoryChunk(
                    id=fact.fact_id,
                    content=fact.content,
                    version=1,
                    metadata={
                        **metadata[0],
                        "assertion_status": (
                            "asserted" if fact.gold_status == "supported" else "conflict"
                        ),
                    },
                    created_at=min(times),
                    updated_at=max(times),
                    last_observed_at=max(times),
                )
            )
        snapshot = AcceptedSnapshot(
            snapshot_id=session.session_id + ":terminal-gold",
            as_of=cutoff,
            registry=registry,
            facts=tuple(facts),
            sources=tuple(sources.values()),
            evidence=tuple(evidence),
        )

        def request(text: str, target: QueryRecord | StatePair) -> RetrievalRequest:
            return RetrievalRequest(
                query=text,
                context=_context(session, target),
                allowed_scopes=list(allowed_scopes),
                top_k=options.top_k,
                max_context_tokens=options.max_context_tokens,
                profile="baseline",
            )

        queries = [
            AcceptedQuery(
                query_id=q.query_id,
                snapshot_id=snapshot.snapshot_id,
                observed_at=query_time,
                request=request(q.text, q),
                expected_fact_ids=q.relevant_fact_ids,
            )
            for q in session.queries
        ]
        for pair in session.state_pairs:
            wordings: tuple[tuple[Literal["semantic", "state"], str], ...] = (
                ("semantic", pair.semantic_query),
                ("state", pair.state_query),
            )
            for variant, text in wordings:
                queries.append(
                    AcceptedQuery(
                        query_id=pair.pair_id + ":" + variant,
                        snapshot_id=snapshot.snapshot_id,
                        observed_at=query_time,
                        request=request(text, pair),
                        expected_fact_ids=pair.relevant_fact_ids,
                        pair_id=pair.pair_id,
                        pair_variant=variant,
                        pair_mode="source_wordings",
                        declared_state=variant + ":" + pair.semantic_state,
                    )
                )
        cases.append(
            AcceptedCase(
                case_id=session.session_id,
                session_id=session.session_id,
                snapshots=(snapshot,),
                queries=tuple(queries),
                critical_fact_ids=tuple(
                    f.fact_id
                    for f in session.facts
                    if f.critical and f.gold_status in {"supported", "conflict"}
                ),
            )
        )
        source_events += len(session.events)
        retained_events += len(sources)
    corpus = AcceptedCorpus(
        version="accepted-retrieval-corpus-v2",
        dataset_id=dataset.version + ":" + dataset.digest + ":" + split,
        stage=split,
        source_label="synthetic_declared_terminal_gold",
        human_acceptance="not_run",
        cases=tuple(cases),
    )
    coverage = validate_corpus(corpus, options)
    return AdaptedAcceptedCorpus(
        corpus,
        {
            "source_version": dataset.version,
            "source_digest": dataset.digest,
            "source_file_sha256": file_sha256,
            "source_object_sha256": _digest(_payload(dataset)),
            "selected_split": split,
            "query_phase": query_phase,
            "allowed_scopes": list(allowed_scopes),
            "source_events": source_events,
            "retained_source_events": retained_events,
            "gold_dispositions": dict(dispositions),
            "coverage": coverage,
            "corpus_sha256": _digest(corpus.model_dump(mode="json")),
            "versions": "gold identities start at declared version 1, not measured runtime versions",
            "lifecycle_executed": False,
            "extraction_executed": False,
            "quality_gate_claimed": False,
        },
    )
