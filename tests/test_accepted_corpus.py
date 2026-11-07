import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from benchmarks.local_memory import accepted_corpus as runner
from benchmarks.local_memory.accepted_retrieval import _digest
from mnemos.utils.embeddings import EmbeddingProvider

STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/local_memory/fixtures/accepted-retrieval-development-v1.json"
)


def corpus():
    data = json.loads(FIXTURE.read_bytes())
    old = data["snapshot"]
    new = deepcopy(old)
    new["snapshot_id"] = "after-correction"
    new["as_of"] = (STAMP + timedelta(days=1)).isoformat()
    fact = next(f for f in new["facts"] if f["id"] == "prod-retention")
    fact.update(
        content=fact["content"].replace("thirty", "forty"),
        version=2,
        updated_at=new["as_of"],
        last_observed_at=new["as_of"],
    )
    event = next(s for s in new["sources"] if s["id"] == "prod-retention-source")
    event.update(
        id="corrected-source",
        event_fingerprint="corrected-fingerprint",
        content=fact["content"],
        normalized_content=fact["content"],
        occurred_at=new["as_of"],
        created_at=new["as_of"],
    )
    evidence = next(e for e in new["evidence"] if e["fact_id"] == "prod-retention")
    evidence.update(
        evidence_id="corrected-evidence",
        source_event_id="corrected-source",
        excerpt=fact["content"],
        created_at=new["as_of"],
    )
    source_query = deepcopy(data["queries"][0])
    source_query["request"]["context"]["host"] = "codex"
    queries = []
    for name, state, day, labels in [
        ("before", old, 0.5, ["prod-retention"]),
        ("after", new, 1.5, ["prod-retention"]),
        ("absent", new, 1.5, []),
    ]:
        queries.append(
            dict(
                query_id=name,
                snapshot_id=state["snapshot_id"],
                observed_at=STAMP + timedelta(days=day),
                request=deepcopy(source_query["request"]),
                expected_fact_ids=labels,
            )
        )
        if not labels:
            queries[-1]["request"]["query"] = "Which region hosts Pine production?"
    return runner.AcceptedCorpus.model_validate(
        dict(
            dataset_id="time-state-development",
            stage="development",
            source_label="authored-synthetic-development",
            cases=[
                dict(
                    case_id="lifecycle",
                    session_id="declared-snapshot",
                    snapshots=[old, new],
                    queries=queries,
                    critical_fact_ids=["prod-retention"],
                )
            ],
        )
    )


def options():
    return runner.CorpusOptions(bootstrap_repetitions=100)


class Provider(EmbeddingProvider):
    dim = 2

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def embed(self, text):
        raise AssertionError("role required")

    def embed_batch(self, texts):
        raise AssertionError("unexpected batch")

    def call(self, role, text):
        self.calls.append((role, text))
        if self.failure == role:
            raise RuntimeError("injected provider failure")
        return [1.0, 0.0]

    def embed_document(self, text):
        return self.call("document", text)

    def embed_query(self, text):
        return self.call("query", text)


def raw_rows(data=None, config=None):
    data, config = data or corpus(), config or options()
    rows = []
    for case in data.cases:
        for query in case.queries:
            snapshot = next(s for s in case.snapshots if s.snapshot_id == query.snapshot_id)
            facts = {f.id: f for f in snapshot.facts}
            selected = list(query.expected_fact_ids)
            result = {
                "facts": [
                    dict(
                        id=i,
                        version=facts[i].version,
                        content=facts[i].content,
                        scope=facts[i].metadata["scope"],
                        scope_id=facts[i].metadata["scope_id"],
                        assertion_status=facts[i].metadata["assertion_status"],
                    )
                    for i in selected
                ]
            }
            for rep in range(1, config.repetitions + 1):
                for arm in runner.STRATEGIES:
                    rows.append(
                        dict(
                            case_id=case.case_id,
                            session_id=case.session_id,
                            query_id=query.query_id,
                            snapshot_id=query.snapshot_id,
                            observed_at=query.observed_at.isoformat(),
                            repetition=rep,
                            strategy=arm,
                            request=query.request.model_dump(mode="json"),
                            expected_fact_ids=list(query.expected_fact_ids),
                            selected_ids=selected.copy(),
                            result=deepcopy(result),
                            error_type=None,
                            query_kind=query.kind,
                            corpus_sha256=_digest(data.model_dump(mode="json")),
                            fixture_sha256=_digest(snapshot.model_dump(mode="json")),
                            state_sha256="state:" + query.snapshot_id,
                            document_vectors_sha256="docs:" + query.snapshot_id,
                            sqlite_snapshot_sha256="sqlite:" + query.snapshot_id,
                            query_vector_sha256=None if arm == "lexical" else "query",
                        )
                    )
    return rows


def test_declared_times_and_small_development_coverage():
    coverage = runner.validate_corpus(corpus(), options())
    assert coverage["counts"] == dict(
        heldout_sessions=1,
        critical_facts=1,
        answerable_queries=2,
        no_answer_queries=1,
        state_pairs=0,
    )
    assert not coverage["minimums_met"] and not coverage["freshness_and_semantic_review_verified"]


@pytest.mark.parametrize(
    "mutation",
    [
        "future",
        "old",
        "naive",
        "duplicate_query",
        "duplicate_snapshot",
        "foreign_label",
        "conflict_label",
        "missing_label",
        "budget",
        "profile",
        "cluster",
        "critical",
        "version",
        "pair",
        "duplicate_scope",
    ],
)
def test_invalid_corpus_fails_before_provider_or_output(tmp_path, mutation):
    data = corpus()
    case = data.cases[0]
    query = case.queries[0]
    if mutation == "future":
        query.snapshot_id = case.snapshots[1].snapshot_id
    elif mutation == "old":
        case.queries[1].snapshot_id = case.snapshots[0].snapshot_id
    elif mutation == "naive":
        query.observed_at = query.observed_at.replace(tzinfo=None)
    elif mutation == "duplicate_query":
        case.queries[1].query_id = query.query_id
    elif mutation == "duplicate_snapshot":
        case.snapshots[1].snapshot_id = case.snapshots[0].snapshot_id
    elif mutation == "foreign_label":
        query.expected_fact_ids = ("birch-retention",)
    elif mutation == "conflict_label":
        query.expected_fact_ids = ("review-two",)
    elif mutation == "missing_label":
        query.expected_fact_ids = ("missing",)
    elif mutation == "budget":
        query.request.max_context_tokens = 100
    elif mutation == "profile":
        query.request.profile = "biomimetic"
    elif mutation == "cluster":
        query.request.context.session_id = "different"
    elif mutation == "critical":
        case.critical_fact_ids = ("absent",)
    elif mutation == "version":
        next(f for f in case.snapshots[1].facts if f.id == "prod-retention").version = 1
    elif mutation == "pair":
        query.pair_id = "unmatched"
    elif mutation == "duplicate_scope":
        query.request.allowed_scopes = ["project", "project"]
    called = []
    with pytest.raises(ValueError):
        asyncio.run(
            runner.run_accepted_corpus(
                data,
                options(),
                tmp_path / "run",
                lambda: called.append(True),
                dimension=2,
                binding_files=[FIXTURE],
                provider_identity={"kind": "fixture"},
            )
        )
    assert not called and not (tmp_path / "run").exists()


def test_heldout_minimums_cannot_be_bypassed_by_a_small_fixture():
    data = corpus()
    data.stage = "heldout"
    with pytest.raises(ValueError, match="minimums"):
        runner.validate_corpus(data, options())


def test_full_coverage_validates_offline_without_claiming_semantic_review():
    data = corpus()
    base = data.cases[0]
    cases = []
    for number in range(200):
        case = base.model_copy(deep=True)
        case.case_id = case.session_id = f"s{number}"
        for query in case.queries:
            query.request.context.session_id = case.session_id
            query.request.context.host = ["codex", "claude-code", "hermes"][number % 3]
            if number % 2:
                query.request.context.project_id = "birch"
                if query.expected_fact_ids:
                    query.expected_fact_ids = ("birch-retention",)
        if number < 50:
            variants = []
            for variant, state in [("semantic", "neutral"), ("state", "focused")]:
                query = case.queries[0].model_copy(deep=True)
                query.query_id = variant
                query.pair_id = "pair"
                query.pair_variant = variant
                query.declared_state = state
                variants.append(query)
            case.queries = (*case.queries, *variants)
        cases.append(case)
    data.cases = tuple(cases)
    data.stage = "heldout"
    result = runner.validate_corpus(data, options())
    assert result["minimums_met"] and result["counts"]["state_pairs"] == 50
    assert result["counts"]["heldout_sessions"] == 200 and result["project_count"] == 2
    assert (
        not result["state_conditions_used_by_baseline"]
        and not result["freshness_and_semantic_review_verified"]
    )


def test_intervals_do_not_treat_repetitions_as_independent_questions():
    report = runner.summarize_corpus(corpus(), options(), raw_rows())
    answers = report["quality_kinds"]["answerable"]
    vector = answers["strategies"]["vector"]
    assert vector["attempted"] == 6 and vector["unique_queries"] == 2 and vector["successes"] == 6
    assert [v["interval"]["sample_count"] for v in vector["wilson_by_repetition"]] == [2, 2, 2]
    assert vector["wilson_by_repetition"][0]["interval"]["lower"] == pytest.approx(0.3423802275)
    paired = answers["paired_to_vector"]["hybrid"]
    assert paired["sample_count"] == 6 and paired["cluster_count"] == 1 and paired["estimate"] == 0
    assert not report["quality_gate_claimed"] and not report["full_module_ablation_run"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unplanned",
        "label",
        "request",
        "snapshot",
        "vector",
        "cross_query_state",
    ],
)
def test_summary_rejects_incomplete_or_unpaired_evidence(mutation):
    rows = raw_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "unplanned":
        rows[0]["query_id"] = "other"
    elif mutation == "label":
        rows[0]["expected_fact_ids"] = []
    elif mutation == "request":
        rows[0]["request"]["top_k"] = 1
    elif mutation == "snapshot":
        rows[0]["sqlite_snapshot_sha256"] = "different"
    elif mutation == "vector":
        rows[0]["query_vector_sha256"] = "different"
    elif mutation == "cross_query_state":
        for row in rows:
            if row["query_id"] == "absent":
                row["document_vectors_sha256"] = "changed"
    with pytest.raises(ValueError):
        runner.summarize_corpus(corpus(), options(), rows)


def test_errors_remain_in_abstention_and_paired_denominators():
    rows = raw_rows()
    for row in rows:
        if row["strategy"] == "hybrid" and row["query_id"] == "absent":
            row.update(result=None, error_type="TimeoutError", selected_ids=[])
    report = runner.summarize_corpus(corpus(), options(), rows)["quality_kinds"]["no_answer"]
    assert report["strategies"]["hybrid"]["attempted"] == 3
    assert report["strategies"]["hybrid"]["successes"] == 0
    assert report["strategies"]["hybrid"]["operational_failure"] == 3
    assert report["paired_to_vector"]["hybrid"]["estimate"] == -1


def test_summary_recomputes_scores_from_raw_content_and_detects_stale_versions():
    rows = raw_rows()
    rows[0]["grade"] = {"answerable_hit": True}
    rows[0]["result"]["facts"][0]["content"] = "Wrong content with the expected ID"
    report = runner.summarize_corpus(corpus(), options(), rows)["quality_kinds"]["answerable"][
        "strategies"
    ]["vector"]
    assert report["successes"] == 5 and report["stale_count"] == 1


@pytest.mark.asyncio
async def test_runner_uses_query_time_state_shares_vectors_and_preserves_artifacts(tmp_path):
    provider = Provider()
    output = tmp_path / "run"
    summary = await runner.run_accepted_corpus(
        corpus(),
        options(),
        output,
        lambda: provider,
        dimension=2,
        binding_files=[FIXTURE],
        provider_identity={"kind": "fixture"},
    )
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 27 and summary["status"] == "complete"
    for row in rows:
        facts = {f["id"]: f for f in row["result"]["facts"]}
        if "prod-retention" in facts:
            assert ("thirty" if row["query_id"] == "before" else "forty") in facts[
                "prod-retention"
            ]["content"]
    assert len([c for c in provider.calls if c[0] == "document"]) == 12
    assert len([c for c in provider.calls if c[0] == "query"]) == 9
    assert json.loads((output / "control.json").read_bytes())["status"] == "completed"
    before = (output / "observations.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        await runner.run_accepted_corpus(
            corpus(),
            options(),
            output,
            lambda: pytest.fail("must not create provider"),
            dimension=2,
            binding_files=[FIXTURE],
            provider_identity={"kind": "fixture"},
        )
    assert (output / "observations.jsonl").read_bytes() == before


@pytest.mark.asyncio
async def test_document_provider_failure_records_full_failure_matrix(tmp_path):
    summary = await runner.run_accepted_corpus(
        corpus(),
        options(),
        tmp_path / "run",
        lambda: Provider("document"),
        dimension=2,
        binding_files=[FIXTURE],
        provider_identity={"kind": "fixture"},
    )
    for kind in ["answerable", "no_answer"]:
        for arm in runner.STRATEGIES:
            metric = summary["quality_kinds"][kind]["strategies"][arm]
            assert metric["successes"] == 0 and metric["operational_failure"] == metric["attempted"]


@pytest.mark.asyncio
async def test_query_provider_failure_does_not_earn_empty_result_credit(tmp_path):
    summary = await runner.run_accepted_corpus(
        corpus(),
        options(),
        tmp_path / "run",
        lambda: Provider("query"),
        dimension=2,
        binding_files=[FIXTURE],
        provider_identity={"kind": "fixture"},
    )
    for arm in ["vector", "hybrid"]:
        metric = summary["quality_kinds"]["no_answer"]["strategies"][arm]
        assert metric["successes"] == 0 and metric["operational_failure"] == 3
    assert (
        summary["quality_kinds"]["answerable"]["strategies"]["lexical"]["operational_failure"] == 0
    )


@pytest.mark.asyncio
async def test_file_drift_aborts_and_retains_failure_manifest(tmp_path):
    bound = tmp_path / "bound.txt"
    bound.write_text("original")

    def factory():
        bound.write_text("changed")
        return Provider()

    output = tmp_path / "run"
    with pytest.raises(ValueError, match="binding changed"):
        await runner.run_accepted_corpus(
            corpus(),
            options(),
            output,
            factory,
            dimension=2,
            binding_files=[bound],
            provider_identity={"kind": "fixture"},
        )
    assert json.loads((output / "control.json").read_bytes())["status"] == "failed"
    assert not (output / "summary.json").exists()


@pytest.mark.asyncio
async def test_aborted_query_closes_owned_store(tmp_path, monkeypatch):
    owned = []
    original = runner.build_accepted_snapshot

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        owned.append(result.service.engine.store)
        return result

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(runner, "build_accepted_snapshot", build)
    monkeypatch.setattr(runner, "compare_accepted_query", cancel)
    with pytest.raises(asyncio.CancelledError):
        await runner.run_accepted_corpus(
            corpus(),
            options(),
            tmp_path / "run",
            Provider,
            dimension=2,
            binding_files=[FIXTURE],
            provider_identity={"kind": "fixture"},
        )
    assert len(owned) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        owned[0]._conn.execute("SELECT 1")


def test_paired_fractional_recall_detects_missing_facts_when_any_hit_is_unchanged():
    data = corpus()
    for query in data.cases[0].queries[:2]:
        query.expected_fact_ids = ("prod-retention", "stage-retention")
        query.request.query = "How long are production and staging request logs kept?"
    rows = raw_rows(data)
    for row in rows:
        if row["strategy"] == "hybrid" and row["query_kind"] == "answerable":
            row["selected_ids"] = ["prod-retention"]
            row["result"]["facts"] = row["result"]["facts"][:1]
    report = runner.summarize_corpus(data, options(), rows)["quality_kinds"]["answerable"]
    assert report["paired_to_vector"]["hybrid"]["estimate"] == 0
    assert report["paired_fractional_recall_to_vector"]["hybrid"]["estimate"] == -0.5
    assert report["strategies"]["hybrid"]["mean_fractional_recall"] == 0.5


def test_whole_session_resampling_preserves_unequal_query_cluster_sizes():
    data = corpus()
    second = data.cases[0].model_copy(deep=True)
    second.case_id = "second"
    second.session_id = "second-session"
    second.queries = (
        *second.queries,
        second.queries[0].model_copy(deep=True),
        second.queries[1].model_copy(deep=True),
    )
    second.queries[-2].query_id = "extra-before"
    second.queries[-1].query_id = "extra-after"
    for query in second.queries:
        query.request.context.session_id = second.session_id
    data.cases = (*data.cases, second)
    rows = raw_rows(data)
    for row in rows:
        if row["case_id"] == "lifecycle" and row["strategy"] == "hybrid":
            row["selected_ids"] = []
            row["result"]["facts"] = []
    paired = runner.summarize_corpus(data, options(), rows)["quality_kinds"]["answerable"][
        "paired_to_vector"
    ]["hybrid"]
    assert paired["sample_count"] == 18 and paired["cluster_count"] == 2
    assert paired["estimate"] == pytest.approx(-1 / 3)
    assert paired["lower"] == -1 and paired["upper"] == 0


def test_scope_leak_cannot_be_hidden_by_expected_id_grade():
    data = corpus()
    rows = raw_rows(data)
    foreign = next(f for f in data.cases[0].snapshots[0].facts if f.id == "birch-retention")
    rows[0]["selected_ids"].append(foreign.id)
    rows[0]["result"]["facts"].append(
        dict(
            id=foreign.id,
            version=foreign.version,
            content=foreign.content,
            scope="project",
            scope_id="birch",
            assertion_status="asserted",
        )
    )
    metric = runner.summarize_corpus(data, options(), rows)["quality_kinds"]["answerable"][
        "strategies"
    ]["vector"]
    assert metric["scope_leak_count"] == 1 and metric["clean_hits"] == 5


@pytest.mark.asyncio
async def test_provider_startup_failure_is_terminal_and_preserved(tmp_path):
    def factory():
        raise RuntimeError("startup failed")

    with pytest.raises(RuntimeError):
        await runner.run_accepted_corpus(
            corpus(),
            options(),
            tmp_path / "run",
            factory,
            dimension=2,
            binding_files=[FIXTURE],
            provider_identity={"kind": "fixture"},
        )
    control = json.loads((tmp_path / "run/control.json").read_bytes())
    assert control["status"] == "failed" and control["observations"] == 0
    assert not (tmp_path / "run/summary.json").exists()
