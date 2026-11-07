"""Declared conversion contracts, distinct from real ingestion quality."""

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from benchmarks.local_memory.accepted_adapter import adapt_terminal_gold
from benchmarks.local_memory.accepted_corpus import (
    AcceptedCorpus,
    CorpusOptions,
    run_accepted_corpus,
    validate_corpus,
)
from benchmarks.local_memory.accepted_retrieval import _digest
from benchmarks.local_memory.contracts import (
    CONTRACT_V4_VERSION,
    CONTRACT_V6_VERSION,
    FactRecord,
    HeldoutDataset,
    QueryRecord,
    SessionScenario,
    SourceEvent,
    StatePair,
    load_dataset,
)
from tests.test_accepted_corpus import Provider, corpus as v1_corpus


def development_dataset():
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def event(key, minute, content, operation="capture", evidence=(), **kwargs):
        return SourceEvent(
            event_id=key,
            occurred_at=(stamp + timedelta(minutes=minute)).isoformat(),
            role=kwargs.pop("role", "user"),
            content=content,
            source_kind="user_message",
            write_mode=kwargs.pop("write_mode", "automatic"),
            lifecycle_operation=operation,
            evidence_fact_ids=evidence,
            **kwargs,
        )

    old = "Alder production logs are retained for thirty days."
    current = "Alder production logs are retained for forty days."
    region = "Alder deploys in the east region."
    owner1 = "Mira owns the Alder deployment service."
    owner2 = "Ivo owns the Alder deployment service."
    global_fact = "The preferred release archive format is zip."
    foreign = "Birch production logs are retained for ninety days."
    events = (
        event("a-create", 0, old, "remember", ("a-old",), write_mode="explicit"),
        event(
            "a-correct",
            1,
            current,
            "correct",
            ("a-current",),
            write_mode="explicit",
            target_source_event_id="a-create",
            target_fact_id="a-old",
            replacement_content=current,
        ),
        event("a-region", 2, region, "remember", ("a-region",), write_mode="explicit"),
        event(
            "a-forget",
            3,
            "Forget the deployment region.",
            "forget",
            write_mode="explicit",
            target_source_event_id="a-region",
            target_fact_id="a-region",
        ),
        event(
            "a-replay",
            4,
            region,
            "replay",
            write_mode="explicit",
            target_source_event_id="a-region",
        ),
        event("a-conflict-one", 5, owner1, evidence=("a-conflict-one",)),
        event("a-conflict-two", 6, owner2, evidence=("a-conflict-two",)),
        event(
            "a-assistant",
            7,
            "I think the region is west.",
            "response",
            ("a-rejected",),
            role="assistant",
            write_mode="none",
        ),
        event("a-global", 8, global_fact, evidence=("a-global",)),
        event(
            "a-foreign",
            9,
            foreign,
            evidence=("a-foreign",),
            target_user_profile_id="foreign-profile",
            target_project_id="birch-other",
        ),
        event("a-restart", 10, "Restart the host.", "restart", role="system", write_mode="none"),
    )

    def fact(key, content, status, source, scope="project", critical=False):
        return FactRecord(key, content, critical, scope, "synthetic_seed", status, (source,))

    facts = (
        fact("a-old", old, "superseded", "a-create"),
        fact("a-current", current, "supported", "a-correct", critical=True),
        fact("a-region", region, "forgotten", "a-region"),
        fact("a-conflict-one", owner1, "conflict", "a-conflict-one"),
        fact("a-conflict-two", owner2, "conflict", "a-conflict-two"),
        fact("a-rejected", events[7].content, "rejected", "a-assistant"),
        fact("a-global", global_fact, "supported", "a-global", "global", True),
        fact("a-foreign", foreign, "supported", "a-foreign"),
    )
    queries = (
        QueryRecord(
            "a-current-query",
            "How long are Alder production logs retained?",
            "answerable",
            ("a-current",),
        ),
        QueryRecord("a-region-query", "Which region hosts Alder?", "no_answer", ()),
        QueryRecord("a-owner-query", "Who owns the Alder deployment service?", "no_answer", ()),
        QueryRecord(
            "a-global-query",
            "What release archive format is preferred?",
            "answerable",
            ("a-global",),
        ),
        QueryRecord(
            "a-foreign-query",
            "How long are Birch production logs retained?",
            "answerable",
            ("a-foreign",),
            target_user_profile_id="foreign-profile",
            target_project_id="birch-other",
        ),
    )
    pairs = (
        StatePair(
            "a-pair",
            "How long are Alder production logs retained?",
            "This is urgent. How long are Alder production logs retained?",
            ("a-current",),
            "focused",
            current,
        ),
    )
    first = SessionScenario(
        "dev-a",
        "development",
        "codex",
        "alder",
        facts,
        queries,
        pairs,
        "synthetic_seed",
        "not_run",
        events,
        ("correction", "forgotten_replay", "conflicting", "cross_profile"),
    )
    owner = "Tess owns the Cypress API."
    workspace = "The shared release checklist is in team/release.md."
    second = SessionScenario(
        "dev-b",
        "development",
        "claude-code",
        "cypress",
        (
            fact("b-owner", owner, "supported", "b-owner", critical=True),
            fact("b-workspace", workspace, "supported", "b-workspace", "workspace"),
        ),
        (
            QueryRecord("b-owner-query", "Who owns the Cypress API?", "answerable", ("b-owner",)),
            QueryRecord(
                "b-region-query", "Where is the Cypress deployment region?", "no_answer", ()
            ),
        ),
        (),
        "synthetic_seed",
        "not_run",
        (
            event("b-owner", 0, owner, "remember", ("b-owner",), write_mode="explicit"),
            event(
                "b-workspace",
                1,
                workspace,
                evidence=("b-workspace",),
                target_workspace_id="shared-workspace",
            ),
            event("b-restart", 2, "Restart the host.", "restart", role="system", write_mode="none"),
        ),
        ("critical_explicit",),
    )
    return HeldoutDataset(
        CONTRACT_V6_VERSION,
        (first, second),
        ("dev-a", "dev-b"),
        _digest([asdict(first), asdict(second)]),
        {
            "source_label": "synthetic_seed",
            "human_acceptance": "not_run",
            "advertised_as_real": False,
        },
    )


def adapt(data=None, **kwargs):
    return adapt_terminal_gold(
        data or development_dataset(), split="development", query_phase="after_all_events", **kwargs
    )


def test_terminal_gold_preserves_ids_sources_scopes_and_conflicts():
    source = development_dataset()
    before = asdict(source)
    result = adapt(source)
    first = result.corpus.cases[0]
    snapshot = first.snapshots[0]
    facts = {f.id: f for f in snapshot.facts}
    assert set(facts) == {"a-current", "a-conflict-one", "a-conflict-two", "a-global", "a-foreign"}
    assert facts["a-current"].content == source.sessions[0].facts[1].content
    assert facts["a-current"].version == 1
    assert facts["a-conflict-one"].metadata["assertion_status"] == "conflict"
    assert facts["a-foreign"].metadata["user_profile_id"] == "foreign-profile"
    assert (
        facts["a-global"].metadata["scope"] == "global"
        and facts["a-global"].metadata["scope_id"] is None
    )
    events = {e.event_id: e for e in source.sessions[0].events}
    for event in snapshot.sources:
        assert event.content == event.normalized_content == events[event.id].content
    for link in snapshot.evidence:
        assert link.excerpt == events[link.source_event_id].content
    last = datetime.fromisoformat(source.sessions[0].events[-1].occurred_at)
    assert snapshot.as_of == last
    assert all(q.observed_at == last + timedelta(microseconds=1) for q in first.queries)
    assert result.provenance["gold_dispositions"] == {
        "superseded": 1,
        "supported": 5,
        "forgotten": 1,
        "conflict": 2,
        "rejected": 1,
    }
    assert (
        result.provenance["source_events"] == 14
        and result.provenance["retained_source_events"] == 7
    )
    assert (
        not result.provenance["lifecycle_executed"]
        and not result.provenance["quality_gate_claimed"]
    )
    assert asdict(source) == before
    assert all(
        set(f.metadata)
        == {
            "user_profile_id",
            "project_id",
            "workspace_id",
            "session_id",
            "host",
            "scope",
            "scope_id",
            "assertion_status",
        }
        for f in snapshot.facts
    )


def test_wording_pair_is_preserved_and_round_trips():
    result = adapt()
    queries = [q for q in result.corpus.cases[0].queries if q.pair_id]
    assert result.corpus.version == "accepted-retrieval-corpus-v2"
    pair = development_dataset().sessions[0].state_pairs[0]
    assert [q.request.query for q in queries] == [pair.semantic_query, pair.state_query]
    assert all(q.pair_mode == "source_wordings" for q in queries)
    restored = AcceptedCorpus.model_validate_json(result.corpus.model_dump_json())
    assert restored == result.corpus
    assert validate_corpus(restored, CorpusOptions())["counts"]["state_pairs"] == 1


def test_v1_canonical_digest_stays_unchanged():
    data = v1_corpus()
    assert (
        _digest(data.model_dump(mode="json"))
        == "93adf570572021e3823815f98c7bb03a0604aad362ab0af9a6758a61293acefd"
    )
    assert all("pair_mode" not in q for c in data.model_dump()["cases"] for q in c["queries"])


@pytest.mark.parametrize(
    "mutation", ["v1_wording", "same_wording", "mixed_mode", "wrong_scope", "wrong_label"]
)
def test_wording_pairs_require_explicit_v2_mode_and_shared_conditions(mutation):
    data = adapt().corpus
    queries = [q for q in data.cases[0].queries if q.pair_id]
    if mutation == "v1_wording":
        data.version = "accepted-retrieval-corpus-v1"
    elif mutation == "same_wording":
        queries[1].request.query = queries[0].request.query
    elif mutation == "mixed_mode":
        queries[1].pair_mode = "fixed_query"
    elif mutation == "wrong_scope":
        queries[1].request.allowed_scopes = ["project"]
    elif mutation == "wrong_label":
        queries[1].expected_fact_ids = ("a-global",)
    with pytest.raises(ValueError):
        validate_corpus(data, CorpusOptions())


@pytest.mark.parametrize(
    "mutation",
    [
        "source_missing",
        "naive",
        "reordered",
        "duplicate_event",
        "scope_ambiguous",
        "correction_target",
        "forgotten_query",
        "conflict_query",
        "explicit_body",
        "profile_conflict",
        "pair_collision",
        "equal_wordings",
        "unknown_status",
        "wrong_kind",
        "old_version",
        "wrong_membership",
        "human_label",
        "duplicate_session",
        "not_synthetic",
        "missing_restart",
        "unknown_tag",
        "one_way_evidence",
    ],
)
def test_adapter_rejects_ambiguous_or_changed_source_contract(mutation):
    data = development_dataset()
    case = data.sessions[0]
    facts = list(case.facts)
    events = list(case.events)
    queries = list(case.queries)
    pairs = case.state_pairs
    if mutation == "source_missing":
        facts[1] = replace(facts[1], evidence_event_ids=("missing",))
    elif mutation == "naive":
        events[0] = replace(events[0], occurred_at="2026-01-01T00:00:00")
    elif mutation == "reordered":
        events[0], events[1] = events[1], events[0]
    elif mutation == "duplicate_event":
        events[1] = replace(events[1], event_id=events[0].event_id)
    elif mutation == "scope_ambiguous":
        facts[1] = replace(facts[1], evidence_event_ids=("a-correct", "a-foreign"))
    elif mutation == "correction_target":
        events[1] = replace(events[1], target_fact_id="missing")
    elif mutation == "forgotten_query":
        queries[1] = replace(queries[1], kind="answerable", relevant_fact_ids=("a-region",))
    elif mutation == "conflict_query":
        queries[2] = replace(queries[2], kind="answerable", relevant_fact_ids=("a-conflict-one",))
    elif mutation == "explicit_body":
        events[8] = replace(events[8], lifecycle_operation="remember", write_mode="explicit")
        facts[6] = replace(facts[6], content="changed explicit remember body")
    elif mutation == "profile_conflict":
        events[9] = replace(events[9], target_project_id="alder")
    elif mutation == "pair_collision":
        queries[0] = replace(queries[0], query_id="a-pair:semantic")
    elif mutation == "equal_wordings":
        pairs = (replace(pairs[0], state_query=pairs[0].semantic_query),)
    elif mutation == "unknown_status":
        facts[1] = replace(facts[1], gold_status="maybe")
    elif mutation == "wrong_kind":
        queries[1] = replace(queries[1], relevant_fact_ids=("a-current",))
    elif mutation == "old_version":
        data = replace(data, version="local-memory-heldout-v2")
    elif mutation == "wrong_membership":
        data = replace(data, development_session_ids=("dev-b",))
    elif mutation == "human_label":
        case = replace(case, human_acceptance="accepted")
    elif mutation == "duplicate_session":
        data = replace(data, sessions=(case, case))
    elif mutation == "not_synthetic":
        data = replace(data, manifest={**data.manifest, "advertised_as_real": True})
    elif mutation == "missing_restart":
        events[-1] = replace(events[-1], lifecycle_operation="noise")
    elif mutation == "unknown_tag":
        case = replace(case, scenario_types=("unknown-tag",))
    elif mutation == "one_way_evidence":
        events[8] = replace(events[8], evidence_fact_ids=())
    if mutation != "duplicate_session":
        case = replace(
            case,
            facts=tuple(facts),
            events=tuple(events),
            queries=tuple(queries),
            state_pairs=pairs,
        )
        data = replace(data, sessions=(case, data.sessions[1]))
    with pytest.raises(ValueError):
        adapt(data)


def test_workspace_answer_requires_explicit_workspace_permission():
    data = development_dataset()
    case = data.sessions[1]
    query = QueryRecord(
        "workspace-query",
        "Where is the shared release checklist?",
        "answerable",
        ("b-workspace",),
        target_workspace_id="shared-workspace",
    )
    data = replace(data, sessions=(data.sessions[0], replace(case, queries=(*case.queries, query))))
    with pytest.raises(ValueError, match="visible"):
        adapt(data)
    result = adapt(data, allowed_scopes=("project", "workspace", "global"))
    assert result.corpus.cases[1].queries[-1].request.allowed_scopes == [
        "project",
        "workspace",
        "global",
    ]
    assert result.provenance["allowed_scopes"] == ["project", "workspace", "global"]


def test_unsupported_query_phase_and_fileless_heldout_are_refused():
    with pytest.raises(ValueError, match="query phase"):
        adapt_terminal_gold(development_dataset(), split="development", query_phase="before_events")
    with pytest.raises(ValueError, match="source file"):
        adapt_terminal_gold(development_dataset(), split="heldout", query_phase="after_all_events")


def test_bound_existing_v4_development_conversion_is_deterministic():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl"
    )
    source = load_dataset(path)
    assert source.version == CONTRACT_V4_VERSION
    result = adapt(source)
    assert result.provenance["source_file_sha256"] is not None
    assert {c.case_id for c in result.corpus.cases} == set(source.development_session_ids)
    assert result == adapt(source)
    assert all(c.case_id in source.development_session_ids for c in result.corpus.cases)


def test_bound_file_rejects_in_memory_source_changes():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl"
    )
    data = load_dataset(path)
    changed = replace(data.sessions[0], source_label="modified")
    with pytest.raises(ValueError, match="bound file"):
        adapt(replace(data, sessions=(changed, *data.sessions[1:])))


@pytest.mark.asyncio
async def test_converted_corpus_runs_without_ingestion_and_preserves_wording_inputs(tmp_path):
    result = adapt()
    provider = Provider()
    report = await run_accepted_corpus(
        result.corpus,
        CorpusOptions(repetitions=1, bootstrap_repetitions=10),
        tmp_path / "run",
        lambda: provider,
        dimension=2,
        binding_files=[Path(__file__)],
        provider_identity={"kind": "fixture"},
    )
    assert report["observations"] == 27
    assert len([call for call in provider.calls if call[0] == "document"]) == 7
    words = [text for role, text in provider.calls if role == "query"]
    assert len(words) == 9
    original = development_dataset().sessions[0].state_pairs[0]
    assert original.semantic_query in words and original.state_query in words
    assert not report["quality_gate_claimed"]
