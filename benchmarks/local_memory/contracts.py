"""Versioned held-out data contract for the five-module evaluation.

Rows are intentionally labeled synthetic until a human acceptance pass has
approved the scenarios. They are evaluation fixtures, not evidence of real
conversations or host behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DATASET_VERSION = "local-memory-heldout-v2"
LIFECYCLE_V3_VERSION = "local-memory-heldout-v3"
CONTRACT_V4_VERSION = "local-memory-heldout-v4"
LIFECYCLE_TARGET_VERSIONS = {LIFECYCLE_V3_VERSION, CONTRACT_V4_VERSION}
SUPPORTED_DATASET_VERSIONS = {DATASET_VERSION, *LIFECYCLE_TARGET_VERSIONS}
# This version repairs a fixed, audited v3 family; future datasets get new versions.
V4_REPAIRED_CANARY_SESSIONS = {f"heldout-{number:03d}" for number in range(61, 81)}
MINIMUMS = {
    "heldout_sessions": 200,
    "critical_facts": 100,
    "answerable_queries": 100,
    "no_answer_queries": 100,
    "state_pairs": 50,
}
HOSTS = {"codex", "claude-code", "hermes"}
SPLITS = {"heldout", "development"}
QUERY_KINDS = {"answerable", "no_answer"}
EVENT_ROLES = {"user", "assistant", "system"}
WRITE_MODES = {"none", "automatic", "explicit"}
LIFECYCLE_OPERATIONS = {"capture", "remember", "correct", "forget", "restart", "noise", "response"}
FACT_STATUSES = {"supported", "superseded", "forgotten", "rejected", "conflict"}
SCENARIO_TAGS = {
    "conflicting",
    "hypothetical",
    "assistant_claim",
    "noise",
    "critical_explicit",
    "forgotten_replay",
    "long_session",
    "cross_profile",
    "scope_canary",
    "correction",
}


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    content: str
    critical: bool
    scope: str
    source_label: str
    gold_status: str
    evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceEvent:
    event_id: str
    occurred_at: str
    role: str
    content: str
    source_kind: str
    write_mode: str
    lifecycle_operation: str
    evidence_fact_ids: tuple[str, ...]
    target_user_profile_id: str | None = None
    target_project_id: str | None = None
    target_workspace_id: str | None = None
    target_source_event_id: str | None = None
    target_fact_id: str | None = None
    replacement_content: str | None = None


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str
    kind: str
    relevant_fact_ids: tuple[str, ...]
    expected_answer: str | None = None
    target_user_profile_id: str | None = None
    target_project_id: str | None = None
    target_workspace_id: str | None = None


@dataclass(frozen=True)
class StatePair:
    pair_id: str
    semantic_query: str
    state_query: str
    relevant_fact_ids: tuple[str, ...]
    semantic_state: str
    matched_content: str
    target_user_profile_id: str | None = None
    target_project_id: str | None = None
    target_workspace_id: str | None = None
    candidate_fact_ids: tuple[str, ...] = ()
    direct_vector_seed_fact_ids: tuple[str, ...] = ()
    graph_target_fact_ids: tuple[str, ...] = ()

    @property
    def semantic_content(self) -> str:
        """The semantic query text kept constant when state is varied."""
        return self.semantic_query


@dataclass(frozen=True)
class SessionScenario:
    session_id: str
    split: str
    host: str
    project_id: str
    facts: tuple[FactRecord, ...]
    queries: tuple[QueryRecord, ...]
    state_pairs: tuple[StatePair, ...]
    source_label: str
    human_acceptance: str
    events: tuple[SourceEvent, ...]
    scenario_types: tuple[str, ...]


@dataclass(frozen=True)
class HeldoutDataset:
    version: str
    sessions: tuple[SessionScenario, ...]
    development_session_ids: tuple[str, ...]
    digest: str
    manifest: dict[str, Any]
    source_path: Path | None = None

    @property
    def heldout_sessions(self) -> tuple[SessionScenario, ...]:
        return tuple(session for session in self.sessions if session.split == "heldout")

    @property
    def facts(self) -> tuple[FactRecord, ...]:
        return tuple(fact for session in self.sessions for fact in session.facts)

    @property
    def queries(self) -> tuple[QueryRecord, ...]:
        return tuple(query for session in self.sessions for query in session.queries)

    @property
    def state_pairs(self) -> tuple[StatePair, ...]:
        return tuple(pair for session in self.sessions for pair in session.state_pairs)


def _required_text(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value.strip()


def _parse_fact(raw: Any, location: str) -> FactRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: fact must be an object")
    if not isinstance(raw.get("critical"), bool):
        raise ValueError(f"{location}: critical must be a JSON boolean")
    status = _required_text(raw.get("gold_status"), "gold_status", location)
    if status not in FACT_STATUSES:
        raise ValueError(f"{location}: unsupported gold_status {status!r}")
    evidence = raw.get("evidence_event_ids")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{location}: evidence_event_ids must be a non-empty list")
    return FactRecord(
        fact_id=_required_text(raw.get("fact_id"), "fact_id", location),
        content=_required_text(raw.get("content"), "content", location),
        critical=raw.get("critical") is True,
        scope=_required_text(raw.get("scope"), "scope", location),
        source_label=_required_text(raw.get("source_label"), "source_label", location),
        gold_status=status,
        evidence_event_ids=tuple(
            _required_text(item, "evidence_event_id", location) for item in evidence
        ),
    )


def _parse_event(raw: Any, location: str) -> SourceEvent:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: event must be an object")
    evidence = raw.get("evidence_fact_ids", [])
    if not isinstance(evidence, list) or any(
        not isinstance(item, str) or not item.strip() for item in evidence
    ):
        raise ValueError(f"{location}: evidence_fact_ids must be a list of non-empty strings")
    occurred_at = _required_text(raw.get("occurred_at"), "occurred_at", location)
    try:
        datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{location}: occurred_at must be ISO-8601") from exc
    role = _required_text(raw.get("role"), "role", location)
    if role not in EVENT_ROLES:
        raise ValueError(f"{location}: unsupported role {role!r}")
    write_mode = _required_text(raw.get("write_mode"), "write_mode", location)
    if write_mode not in WRITE_MODES:
        raise ValueError(f"{location}: unsupported write_mode {write_mode!r}")
    operation = _required_text(raw.get("lifecycle_operation"), "lifecycle_operation", location)
    if operation not in LIFECYCLE_OPERATIONS:
        raise ValueError(f"{location}: unsupported lifecycle operation {operation!r}")
    return SourceEvent(
        event_id=_required_text(raw.get("event_id"), "event_id", location),
        occurred_at=occurred_at,
        role=role,
        content=_required_text(raw.get("content"), "content", location),
        source_kind=_required_text(raw.get("source_kind"), "source_kind", location),
        write_mode=write_mode,
        lifecycle_operation=operation,
        evidence_fact_ids=tuple(str(item).strip() for item in evidence),
        target_user_profile_id=(
            _required_text(raw.get("target_user_profile_id"), "target_user_profile_id", location)
            if raw.get("target_user_profile_id") is not None
            else None
        ),
        target_project_id=(
            _required_text(raw.get("target_project_id"), "target_project_id", location)
            if raw.get("target_project_id") is not None
            else None
        ),
        target_workspace_id=(
            _required_text(raw.get("target_workspace_id"), "target_workspace_id", location)
            if raw.get("target_workspace_id") is not None
            else None
        ),
        target_source_event_id=(
            _required_text(raw.get("target_source_event_id"), "target_source_event_id", location)
            if raw.get("target_source_event_id") is not None
            else None
        ),
        target_fact_id=(
            _required_text(raw.get("target_fact_id"), "target_fact_id", location)
            if raw.get("target_fact_id") is not None
            else None
        ),
        replacement_content=(
            _required_text(raw.get("replacement_content"), "replacement_content", location)
            if raw.get("replacement_content") is not None
            else None
        ),
    )


def _parse_query(raw: Any, location: str) -> QueryRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: query must be an object")
    kind = _required_text(raw.get("kind"), "kind", location)
    if kind not in QUERY_KINDS:
        raise ValueError(f"{location}: kind must be answerable or no_answer")
    ids = raw.get("relevant_fact_ids", [])
    if not isinstance(ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in ids
    ):
        raise ValueError(f"{location}: relevant_fact_ids must be a list of non-empty strings")
    if kind == "answerable" and not ids:
        raise ValueError(f"{location}: answerable query needs a relevant fact")
    if kind == "no_answer" and ids:
        raise ValueError(f"{location}: no_answer query cannot have relevant facts")
    answer = raw.get("expected_answer")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError(f"{location}: expected_answer must be null or non-empty")
    return QueryRecord(
        query_id=_required_text(raw.get("query_id"), "query_id", location),
        text=_required_text(raw.get("text"), "text", location),
        kind=kind,
        relevant_fact_ids=tuple(str(value).strip() for value in ids),
        expected_answer=answer.strip() if isinstance(answer, str) else None,
        target_user_profile_id=(
            _required_text(raw.get("target_user_profile_id"), "target_user_profile_id", location)
            if raw.get("target_user_profile_id") is not None
            else None
        ),
        target_project_id=(
            _required_text(raw.get("target_project_id"), "target_project_id", location)
            if raw.get("target_project_id") is not None
            else None
        ),
        target_workspace_id=(
            _required_text(raw.get("target_workspace_id"), "target_workspace_id", location)
            if raw.get("target_workspace_id") is not None
            else None
        ),
    )


def _parse_pair(raw: Any, location: str) -> StatePair:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: state pair must be an object")
    ids = raw.get("relevant_fact_ids", [])
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{location}: state pair needs relevant facts")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{location}: state pair fact IDs must be non-empty strings")
    candidates = raw.get("candidate_fact_ids", [])
    direct_seeds = raw.get("direct_vector_seed_fact_ids", [])
    graph_targets = raw.get("graph_target_fact_ids", [])
    for field, values in (
        ("candidate_fact_ids", candidates),
        ("direct_vector_seed_fact_ids", direct_seeds),
        ("graph_target_fact_ids", graph_targets),
    ):
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{location}: {field} must be a list of non-empty strings")
    return StatePair(
        pair_id=_required_text(raw.get("pair_id"), "pair_id", location),
        semantic_query=_required_text(raw.get("semantic_query"), "semantic_query", location),
        state_query=_required_text(raw.get("state_query"), "state_query", location),
        relevant_fact_ids=tuple(str(value).strip() for value in ids),
        semantic_state=_required_text(raw.get("semantic_state"), "semantic_state", location),
        matched_content=_required_text(raw.get("matched_content"), "matched_content", location),
        target_user_profile_id=(
            _required_text(raw.get("target_user_profile_id"), "target_user_profile_id", location)
            if raw.get("target_user_profile_id") is not None
            else None
        ),
        target_project_id=(
            _required_text(raw.get("target_project_id"), "target_project_id", location)
            if raw.get("target_project_id") is not None
            else None
        ),
        target_workspace_id=(
            _required_text(raw.get("target_workspace_id"), "target_workspace_id", location)
            if raw.get("target_workspace_id") is not None
            else None
        ),
        candidate_fact_ids=tuple(
            _required_text(item, "candidate_fact_id", location) for item in candidates
        ),
        direct_vector_seed_fact_ids=tuple(
            _required_text(item, "direct_vector_seed_fact_id", location) for item in direct_seeds
        ),
        graph_target_fact_ids=tuple(
            _required_text(item, "graph_target_fact_id", location) for item in graph_targets
        ),
    )


def _parse_session(raw: dict[str, Any], index: int) -> SessionScenario:
    location = f"session[{index}]"
    session_id = _required_text(raw.get("session_id"), "session_id", location)
    split = _required_text(raw.get("split"), "split", location)
    if split not in SPLITS:
        raise ValueError(f"{location}: split must be heldout or development")
    host = _required_text(raw.get("host"), "host", location)
    if host not in HOSTS:
        raise ValueError(f"{location}: unsupported host {host!r}")
    facts_raw = raw.get("facts")
    queries_raw = raw.get("queries")
    pairs_raw = raw.get("state_pairs", [])
    events_raw = raw.get("events")
    scenario_types_raw = raw.get("scenario_types")
    if not isinstance(facts_raw, list) or not facts_raw:
        raise ValueError(f"{location}: facts must be a non-empty list")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise ValueError(f"{location}: queries must be a non-empty list")
    if not isinstance(pairs_raw, list):
        raise ValueError(f"{location}: state_pairs must be a list")
    if not isinstance(events_raw, list) or not events_raw:
        raise ValueError(f"{location}: events must be a non-empty list")
    if not isinstance(scenario_types_raw, list) or not scenario_types_raw:
        raise ValueError(f"{location}: scenario_types must be a non-empty list")
    return SessionScenario(
        session_id=session_id,
        split=split,
        host=host,
        project_id=_required_text(raw.get("project_id"), "project_id", location),
        facts=tuple(
            _parse_fact(item, f"{location}.facts[{i}]") for i, item in enumerate(facts_raw)
        ),
        queries=tuple(
            _parse_query(item, f"{location}.queries[{i}]") for i, item in enumerate(queries_raw)
        ),
        state_pairs=tuple(
            _parse_pair(item, f"{location}.state_pairs[{i}]") for i, item in enumerate(pairs_raw)
        ),
        source_label=_required_text(raw.get("source_label"), "source_label", location),
        human_acceptance=_required_text(raw.get("human_acceptance"), "human_acceptance", location),
        events=tuple(
            _parse_event(item, f"{location}.events[{i}]") for i, item in enumerate(events_raw)
        ),
        scenario_types=tuple(
            _required_text(item, "scenario_type", location) for item in scenario_types_raw
        ),
    )


def _canonical_lines(path: Path) -> bytes:
    # Digest the bytes as shipped. This catches accidental line or encoding edits.
    return path.read_bytes()


def _validate_v3_lifecycle_targets(
    session: SessionScenario,
    events_by_id: dict[str, SourceEvent],
    facts_by_id: dict[str, FactRecord],
) -> None:
    """Validate explicit action targets added by the v3 amendment."""

    event_positions = {event.event_id: index for index, event in enumerate(session.events)}

    def resolved_context(event: SourceEvent) -> dict[str, str | None]:
        return {
            "target_user_profile_id": event.target_user_profile_id,
            "target_project_id": event.target_project_id or session.project_id,
            "target_workspace_id": event.target_workspace_id,
        }

    for event in session.events:
        has_target = any(
            value is not None
            for value in (
                event.target_source_event_id,
                event.target_fact_id,
                event.replacement_content,
            )
        )
        if event.lifecycle_operation not in {"correct", "forget"}:
            if has_target:
                raise ValueError(
                    f"event {event.event_id}: target metadata is only valid for correct or forget"
                )
            continue
        if event.target_source_event_id is None or event.target_fact_id is None:
            raise ValueError(f"event {event.event_id}: lifecycle target metadata is required")
        target_event = events_by_id.get(event.target_source_event_id)
        target_fact = facts_by_id.get(event.target_fact_id)
        if target_event is None:
            raise ValueError(
                f"event {event.event_id}: target_source_event_id does not exist before action"
            )
        if target_fact is None:
            raise ValueError(f"event {event.event_id}: target_fact_id does not exist")
        if event_positions[target_event.event_id] >= event_positions[event.event_id]:
            raise ValueError(
                f"event {event.event_id}: target event must precede action in event order"
            )
        if target_event.occurred_at >= event.occurred_at:
            raise ValueError(f"event {event.event_id}: target event must precede action")
        if (
            target_event.role != "user"
            or target_event.lifecycle_operation not in {"capture", "remember"}
            or (
                target_event.lifecycle_operation == "remember"
                and target_event.write_mode != "explicit"
            )
        ):
            raise ValueError(
                f"event {event.event_id}: target must be an earlier user capture or explicit user remember"
            )
        target_context = resolved_context(target_event)
        action_context = resolved_context(event)
        for field in target_context:
            if target_context[field] != action_context[field]:
                raise ValueError(
                    f"event {event.event_id}: target event context mismatch for {field}"
                )
        if target_event.evidence_fact_ids.count(event.target_fact_id) != 1:
            raise ValueError(
                f"event {event.event_id}: target event must evidence target_fact_id exactly once"
            )
        if event.target_source_event_id not in target_fact.evidence_event_ids:
            raise ValueError(
                f"event {event.event_id}: target source event is not gold evidence for target_fact_id"
            )
        if event.lifecycle_operation == "forget":
            if event.replacement_content is not None:
                raise ValueError(f"event {event.event_id}: forget cannot have replacement_content")
            if target_fact.gold_status != "forgotten":
                raise ValueError(f"event {event.event_id}: forget target must be forgotten")
        else:
            replacement = event.replacement_content
            if replacement is None or not replacement.strip():
                raise ValueError(f"event {event.event_id}: correct needs replacement_content")
            if target_fact.gold_status != "superseded":
                raise ValueError(f"event {event.event_id}: correct target must be superseded")
            replacement_facts = [
                fact
                for fact in facts_by_id.values()
                if fact.fact_id != target_fact.fact_id and fact.content == replacement
            ]
            if len(replacement_facts) != 1:
                raise ValueError(
                    f"event {event.event_id}: replacement_content must identify one new fact"
                )
            replacement_fact = replacement_facts[0]
            if replacement_fact.gold_status != "supported":
                raise ValueError(f"event {event.event_id}: replacement fact must be supported")
            if replacement_fact.fact_id not in event.evidence_fact_ids:
                raise ValueError(
                    f"event {event.event_id}: replacement fact must be evidenced by correct action"
                )
            if event.event_id not in replacement_fact.evidence_event_ids:
                raise ValueError(
                    f"event {event.event_id}: replacement fact must be evidenced by correct action"
                )


def _validate_v4_contract(
    session: SessionScenario,
    events_by_id: dict[str, SourceEvent],
    facts_by_id: dict[str, FactRecord],
) -> None:
    """Reject source/body and scope-label contradictions before inference."""

    def identity(
        target: SourceEvent | QueryRecord | StatePair, scope: str
    ) -> tuple[str, str, str | None]:
        profile = target.target_user_profile_id or "t18-synthetic-profile"
        scope_id: str | None
        if scope == "project":
            scope_id = target.target_project_id or session.project_id
        elif scope == "workspace":
            scope_id = target.target_workspace_id
            if scope_id is None:
                raise ValueError("workspace fact needs an explicit workspace identity")
        elif scope == "global":
            scope_id = None
        else:
            raise ValueError(f"unsupported fact scope {scope!r}")
        return profile, scope, scope_id

    def check_target(target: QueryRecord | StatePair, fact_ids: tuple[str, ...]) -> None:
        for fact_id in fact_ids:
            fact = facts_by_id.get(fact_id)
            if fact is None or fact.gold_status != "supported":
                raise ValueError("answerable target needs a supported fact in its own session")
            sources = [
                events_by_id[event_id]
                for event_id in fact.evidence_event_ids
                if events_by_id[event_id].role == "user"
                and events_by_id[event_id].write_mode != "none"
                and events_by_id[event_id].lifecycle_operation in {"capture", "remember", "correct"}
            ]
            if not sources or any(
                identity(source, fact.scope) != identity(target, fact.scope) for source in sources
            ):
                raise ValueError(f"answerable target for {fact_id} is outside source scope")

    for event in session.events:
        if event.lifecycle_operation != "remember":
            continue
        if any((event.target_user_profile_id, event.target_project_id, event.target_workspace_id)):
            raise ValueError("explicit remember uses the base session context")
        for fact_id in event.evidence_fact_ids:
            fact = facts_by_id[fact_id]
            if fact.gold_status == "supported" and fact.content != event.content:
                raise ValueError(f"fact {fact_id} must equal its explicit remember body")
    for query in session.queries:
        if query.kind == "answerable":
            check_target(query, query.relevant_fact_ids)
        elif query.expected_answer is not None:
            raise ValueError(f"query {query.query_id}: no_answer cannot have expected_answer")
    if session.session_id in V4_REPAIRED_CANARY_SESSIONS:
        original_id = "query-answerable-" + session.session_id.rsplit("-", 1)[1]
        if [query.query_id for query in session.queries] != [
            original_id,
            original_id + "-in-scope-v4",
        ]:
            raise ValueError(f"session {session.session_id}: missing paired scope canary queries")
        foreign, local = session.queries
        if (
            foreign.kind != "no_answer"
            or not foreign.target_user_profile_id
            or foreign.target_user_profile_id == "t18-synthetic-profile"
            or not foreign.target_project_id
            or foreign.target_project_id == session.project_id
        ):
            raise ValueError(
                f"session {session.session_id}: foreign scope canary must stay foreign"
            )
        if local.kind != "answerable" or any(
            (local.target_user_profile_id, local.target_project_id, local.target_workspace_id)
        ):
            raise ValueError(f"session {session.session_id}: paired scope canary must stay local")
    for pair in session.state_pairs:
        check_target(pair, pair.relevant_fact_ids)
        if pair.matched_content not in {
            facts_by_id[fact_id].content for fact_id in pair.relevant_fact_ids
        }:
            raise ValueError(f"pair {pair.pair_id}: state target must match a supported fact body")


def validate_dataset(dataset: HeldoutDataset) -> HeldoutDataset:
    """Validate counts, labels, uniqueness, and development/held-out separation."""
    sessions = dataset.sessions
    heldout = [session for session in sessions if session.split == "heldout"]
    development = [session for session in sessions if session.split == "development"]
    if len(heldout) < MINIMUMS["heldout_sessions"]:
        raise ValueError(f"held-out sessions below minimum: {len(heldout)}")
    if not development:
        raise ValueError("a disjoint development set is required")
    session_ids = [session.session_id for session in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("session IDs must be unique")
    if set(dataset.development_session_ids) != {session.session_id for session in development}:
        raise ValueError("development_session_ids must exactly identify development rows")
    facts = [fact for session in sessions for fact in session.facts]
    fact_ids = [fact.fact_id for fact in facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("fact IDs must be unique")
    heldout_fact_ids = {fact.fact_id for session in heldout for fact in session.facts}
    critical = sum(fact.critical for fact in facts if fact.fact_id in heldout_fact_ids)
    if critical < MINIMUMS["critical_facts"]:
        raise ValueError(f"critical facts below minimum: {critical}")
    if any(
        fact.critical and fact.gold_status != "supported"
        for fact in facts
        if fact.fact_id in heldout_fact_ids
    ):
        raise ValueError("critical held-out facts must have supported gold status")
    queries = [query for session in heldout for query in session.queries]
    answerable = sum(query.kind == "answerable" for query in queries)
    no_answer = sum(query.kind == "no_answer" for query in queries)
    if answerable < MINIMUMS["answerable_queries"]:
        raise ValueError(f"answerable queries below minimum: {answerable}")
    if no_answer < MINIMUMS["no_answer_queries"]:
        raise ValueError(f"no-answer queries below minimum: {no_answer}")
    if {session.host for session in heldout} != HOSTS:
        raise ValueError("held-out scenarios must span Codex, Claude Code, and Hermes")
    if len({session.project_id for session in heldout}) < 2:
        raise ValueError("held-out scenarios must cover multiple projects")
    heldout_roles = {event.role for session in heldout for event in session.events}
    if heldout_roles != EVENT_ROLES:
        raise ValueError("held-out scenarios must include user, assistant, and system events")
    required_tags = {
        "conflicting",
        "hypothetical",
        "assistant_claim",
        "noise",
        "critical_explicit",
        "forgotten_replay",
        "long_session",
        "cross_profile",
        "scope_canary",
        "correction",
    }
    present_tags = {tag for session in heldout for tag in session.scenario_types}
    if required_tags - present_tags:
        raise ValueError(f"held-out scenarios missing tags: {sorted(required_tags - present_tags)}")
    for session in sessions:
        if len(set(session.scenario_types)) != len(session.scenario_types):
            raise ValueError(f"session {session.session_id} repeats scenario tags")
        if set(session.scenario_types) - SCENARIO_TAGS:
            raise ValueError(f"session {session.session_id} has an unknown scenario tag")
        event_ids = [event.event_id for event in session.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"session {session.session_id} repeats event IDs")
        parsed_times = [
            datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))
            for event in session.events
        ]
        if parsed_times != sorted(parsed_times) or len(set(parsed_times)) != len(parsed_times):
            raise ValueError(f"session {session.session_id} events must be strictly ordered")
        fact_ids_for_session = {fact.fact_id for fact in session.facts}
        event_ids_for_session = set(event_ids)
        events_by_id = {event.event_id: event for event in session.events}
        facts_by_id = {fact.fact_id: fact for fact in session.facts}
        for fact in session.facts:
            if set(fact.evidence_event_ids) - event_ids_for_session:
                raise ValueError(f"fact {fact.fact_id} references an unknown evidence event")
            if fact.gold_status == "supported" and not fact.evidence_event_ids:
                raise ValueError(f"supported fact {fact.fact_id} needs evidence")
            if fact.gold_status == "supported" and not any(
                events_by_id[event_id].role == "user"
                and events_by_id[event_id].write_mode in {"automatic", "explicit"}
                for event_id in fact.evidence_event_ids
            ):
                raise ValueError(f"supported fact {fact.fact_id} needs user evidence")
        for event in session.events:
            if set(event.evidence_fact_ids) - fact_ids_for_session:
                raise ValueError(f"event {event.event_id} references an unknown fact")
        if dataset.version in LIFECYCLE_TARGET_VERSIONS:
            _validate_v3_lifecycle_targets(session, events_by_id, facts_by_id)
        if dataset.version == CONTRACT_V4_VERSION:
            _validate_v4_contract(session, events_by_id, facts_by_id)
        if not any(event.lifecycle_operation == "restart" for event in session.events):
            raise ValueError(f"session {session.session_id} needs a restart boundary")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("held-out query IDs must be unique")
    fact_id_set = heldout_fact_ids
    for query in queries:
        if set(query.relevant_fact_ids) - fact_id_set:
            raise ValueError(f"query {query.query_id} references an unknown fact")
    pairs = [pair for session in heldout for pair in session.state_pairs]
    if len(pairs) < MINIMUMS["state_pairs"]:
        raise ValueError(f"state pairs below minimum: {len(pairs)}")
    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("state pair IDs must be unique")
    if any(set(pair.relevant_fact_ids) - fact_id_set for pair in pairs):
        raise ValueError("state pair references an unknown fact")
    if any(pair.semantic_query == pair.state_query for pair in pairs):
        raise ValueError("state pair semantic and state queries must differ")
    if any(session.human_acceptance != "not_run" for session in sessions):
        raise ValueError("synthetic rows must retain human_acceptance=not_run until reviewed")
    declared_counts = {
        "heldout_session_count": len(heldout),
        "development_session_count": len(development),
        "critical_fact_count": critical,
        "answerable_query_count": answerable,
        "no_answer_query_count": no_answer,
        "state_pair_count": len(pairs),
    }
    for field, actual in declared_counts.items():
        declared = dataset.manifest.get(field)
        if declared != actual:
            raise ValueError(f"manifest {field}={declared!r} does not match parsed count {actual}")
    return dataset


def load_dataset(path: Path) -> HeldoutDataset:
    """Load and validate the versioned JSONL dataset without model or network work."""
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number}: each row must be an object")
        if item.get("record_type") == "manifest":
            if manifest is not None:
                raise ValueError("dataset may contain only one manifest")
            manifest = item
        else:
            records.append(item)
    if manifest is None:
        raise ValueError("dataset manifest row is required")
    if manifest.get("source_label") != "synthetic_seed":
        raise ValueError("manifest source_label must remain synthetic_seed")
    if manifest.get("human_acceptance") != "not_run":
        raise ValueError("manifest human_acceptance must remain not_run")
    if manifest.get("advertised_as_real") is not False:
        raise ValueError("synthetic dataset cannot be advertised as real")
    version = _required_text(manifest.get("dataset_version"), "dataset_version", "manifest")
    if version not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError(f"unsupported dataset version {version!r}")
    sessions = tuple(_parse_session(item, index) for index, item in enumerate(records))
    development_ids = tuple(str(value) for value in manifest.get("development_session_ids", []))
    digest = hashlib.sha256(_canonical_lines(path)).hexdigest()
    dataset = HeldoutDataset(
        version=version,
        sessions=sessions,
        development_session_ids=development_ids,
        digest=digest,
        manifest=manifest,
        source_path=path,
    )
    return validate_dataset(dataset)
