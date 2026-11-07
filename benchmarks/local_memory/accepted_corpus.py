"""Run paired retrieval on declared query-time state, without ingestion.

Accepted state and labels are supplied inputs. This runner cannot establish
extraction quality, lifecycle correctness, semantic label quality or freshness.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mnemos.contracts import RetrievalRequest
from mnemos.scopes import scope_allows
from mnemos.utils.embeddings import EmbeddingProvider
from mnemos.utils.storage import SQLiteStore

from .accepted_retrieval import (
    STRATEGIES,
    AcceptedSnapshot,
    _digest,
    _validate,
    build_accepted_snapshot,
    compare_accepted_query,
)
from .contracts import HOSTS, MINIMUMS
from .scoring import paired_bootstrap_interval, wilson_interval

KINDS = ("answerable", "no_answer", "state_pair_semantic", "state_pair_state")


class AcceptedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str
    snapshot_id: str
    observed_at: datetime
    request: RetrievalRequest
    expected_fact_ids: tuple[str, ...]
    pair_id: str | None = None
    pair_variant: Literal["semantic", "state"] | None = None
    declared_state: str | None = None

    @property
    def kind(self) -> str:
        if self.pair_variant is not None:
            return "state_pair_" + self.pair_variant
        return "answerable" if self.expected_fact_ids else "no_answer"


class AcceptedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    session_id: str
    snapshots: tuple[AcceptedSnapshot, ...]
    queries: tuple[AcceptedQuery, ...]
    critical_fact_ids: tuple[str, ...] = ()


class AcceptedCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["accepted-retrieval-corpus-v1"] = "accepted-retrieval-corpus-v1"
    dataset_id: str
    stage: Literal["development", "heldout"]
    source_label: str
    human_acceptance: Literal["not_run", "accepted"] = "not_run"
    cases: tuple[AcceptedCase, ...]


class CorpusOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repetitions: int = Field(default=3, ge=1, strict=True)
    top_k: Literal[5] = 5
    max_context_tokens: int = Field(default=1200, gt=0, le=32_000, strict=True)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_repetitions: int = Field(default=10_000, ge=1, strict=True)
    bootstrap_seed: int = Field(default=20261107, strict=True)


def _unique(values: Sequence[str], label: str) -> None:
    if any(not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError("Blank or duplicate " + label)


def validate_corpus(corpus: AcceptedCorpus, options: CorpusOptions) -> dict[str, Any]:
    """Validate every snapshot and label before a provider is constructed."""
    corpus = AcceptedCorpus.model_validate(corpus.model_dump(mode="python"))
    options = CorpusOptions.model_validate(options.model_dump(mode="python"))
    if not corpus.dataset_id.strip() or not corpus.source_label.strip() or not corpus.cases:
        raise ValueError("Corpus identity, provenance and cases are required")
    _unique([case.case_id for case in corpus.cases], "case IDs")
    sessions: set[str] = set()
    critical: set[tuple[str, str]] = set()
    projects: set[tuple[str, str | None]] = set()
    hosts: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    query_counts: dict[str, int] = defaultdict(int)
    for case in corpus.cases:
        if not case.session_id.strip() or not case.snapshots or not case.queries:
            raise ValueError("Each case needs a session, snapshots and queries")
        sessions.add(case.session_id)
        _unique([snapshot.snapshot_id for snapshot in case.snapshots], "snapshot IDs")
        _unique([query.query_id for query in case.queries], "query IDs")
        _unique(case.critical_fact_ids, "critical fact IDs")
        snapshots = {snapshot.snapshot_id: _validate(snapshot) for snapshot in case.snapshots}
        if len({snapshot.as_of for snapshot in snapshots.values()}) != len(snapshots):
            raise ValueError("Snapshot cutoffs must be distinct within a case")
        all_facts = {fact.id for snapshot in snapshots.values() for fact in snapshot.facts}
        if not set(case.critical_fact_ids) <= all_facts:
            raise ValueError("Critical labels must identify declared facts")
        critical.update((case.session_id, fact_id) for fact_id in case.critical_fact_ids)
        versions: dict[str, tuple[int, str]] = {}
        for snapshot in sorted(snapshots.values(), key=lambda item: item.as_of):
            for fact in snapshot.facts:
                previous = versions.get(fact.id)
                if previous is not None and (
                    fact.version < previous[0]
                    or (fact.content != previous[1] and fact.version <= previous[0])
                ):
                    raise ValueError("A changed fact needs a newer declared version")
                versions[fact.id] = (fact.version, fact.content)
        paired: dict[str, list[AcceptedQuery]] = defaultdict(list)
        used = set()
        for query in case.queries:
            if query.observed_at.tzinfo is None or query.observed_at.utcoffset() is None:
                raise ValueError("Query time must be timezone-aware")
            available = [s for s in snapshots.values() if s.as_of <= query.observed_at]
            if (
                not available
                or max(available, key=lambda s: s.as_of).snapshot_id != query.snapshot_id
            ):
                raise ValueError("Query must bind the latest declared state available at its time")
            snapshot = snapshots[query.snapshot_id]
            used.add(query.snapshot_id)
            request = query.request
            snapshot.registry.validate_context(request.context)
            if request.context.session_id != case.session_id:
                raise ValueError("Query context must use the declared session cluster")
            if (
                request.top_k != options.top_k
                or request.max_context_tokens != options.max_context_tokens
                or request.view != "memory"
                or request.response_format != "full"
                or request.profile not in {None, "baseline"}
                or not request.query.strip()
                or not request.allowed_scopes
                or len(set(request.allowed_scopes)) != len(request.allowed_scopes)
            ):
                raise ValueError("Queries must share the declared baseline retrieval contract")
            _unique(query.expected_fact_ids, "answer labels")
            visible = {
                fact.id
                for fact in snapshot.facts
                if fact.metadata.get("assertion_status") == "asserted"
                and scope_allows(fact.metadata, request.context, request.allowed_scopes)
            }
            if not set(query.expected_fact_ids) <= visible:
                raise ValueError("Answers must be visible accepted facts at query time")
            hosts.add(request.context.host)
            projects.add((request.context.user_profile_id, request.context.project_id))
            query_counts[query.kind] += 1
            if query.pair_id is None:
                if query.pair_variant is not None or query.declared_state is not None:
                    raise ValueError("State fields require a matched pair identity")
            else:
                if (
                    not query.pair_id.strip()
                    or query.pair_variant is None
                    or not query.expected_fact_ids
                ):
                    raise ValueError("A matched pair needs identity, variant and answer labels")
                if not query.declared_state or not query.declared_state.strip():
                    raise ValueError("A matched pair must declare its state condition")
                paired[query.pair_id].append(query)
        if used != set(snapshots):
            raise ValueError("Every declared snapshot needs a query")
        for pair_id, group in paired.items():
            if len(group) != 2 or {q.pair_variant for q in group} != {"semantic", "state"}:
                raise ValueError("State pairs require exactly two matched variants")
            left, right = group
            if (
                left.snapshot_id != right.snapshot_id
                or left.observed_at != right.observed_at
                or left.request != right.request
                or left.expected_fact_ids != right.expected_fact_ids
                or left.declared_state == right.declared_state
            ):
                raise ValueError("State pairs must differ only in their declared state condition")
            pairs.add((case.session_id, pair_id))
    counts = {
        "heldout_sessions": len(sessions),
        "critical_facts": len(critical),
        "answerable_queries": query_counts["answerable"],
        "no_answer_queries": query_counts["no_answer"],
        "state_pairs": len(pairs),
    }
    deficits = {key: limit - counts[key] for key, limit in MINIMUMS.items() if counts[key] < limit}
    if not HOSTS <= hosts:
        deficits["missing_hosts"] = len(HOSTS - hosts)
    if len({project for project in projects if project[1] is not None}) < 2:
        deficits["multiple_projects"] = 1
    if options.repetitions < 3:
        deficits["repetitions"] = 3 - options.repetitions
    if corpus.stage == "heldout" and deficits:
        raise ValueError(
            "Held-out coverage minimums not met: " + json.dumps(deficits, sort_keys=True)
        )
    return {
        "counts": counts,
        "query_kinds": dict(query_counts),
        "hosts": sorted(hosts),
        "project_count": len({project for project in projects if project[1] is not None}),
        "minimum_deficits": deficits,
        "minimums_met": not deficits,
        "state_conditions_used_by_baseline": False,
        "freshness_and_semantic_review_verified": False,
    }


def _index(
    corpus: AcceptedCorpus,
) -> dict[tuple[str, str], tuple[AcceptedCase, AcceptedQuery, AcceptedSnapshot]]:
    return {
        (case.case_id, query.query_id): (
            case,
            query,
            next(s for s in case.snapshots if s.snapshot_id == query.snapshot_id),
        )
        for case in corpus.cases
        for query in case.queries
    }


def _score(row: dict[str, Any], query: AcceptedQuery, snapshot: AcceptedSnapshot) -> dict[str, Any]:
    """Grade raw returned identities/content; do not trust stored grade fields."""
    selected = row["selected_ids"]
    _unique(selected, "returned IDs")
    result = row["result"]
    if result is None:
        if selected or not row["error_type"]:
            raise ValueError("Missing results require an explicit failure and empty selection")
    elif [fact["id"] for fact in result["facts"]] != selected:
        raise ValueError("Raw results and selected IDs disagree")
    if len(selected) > query.request.top_k:
        raise ValueError("Result exceeds the frozen top-k")
    success = result is not None and row["error_type"] is None and not result.get("degraded")
    declared = {fact.id: fact for fact in snapshot.facts}
    visible = {
        fact.id
        for fact in snapshot.facts
        if scope_allows(fact.metadata, query.request.context, query.request.allowed_scopes)
    }
    current = set()
    stale = set()
    for fact in result["facts"] if result is not None else []:
        expected = declared.get(fact["id"])
        if expected is not None and (
            fact["version"] == expected.version
            and fact["content"] == expected.content
            and fact["scope"] == expected.metadata["scope"]
            and fact["scope_id"] == expected.metadata.get("scope_id")
            and fact["assertion_status"] == expected.metadata["assertion_status"]
        ):
            current.add(fact["id"])
        else:
            stale.add(fact["id"])
    labels = set(query.expected_fact_ids)
    overlap = labels & current if success else set()
    unsupported = set(selected) - labels | stale
    return {
        "hit": bool(overlap),
        "clean_hit": bool(overlap) and not unsupported,
        "recall": len(overlap) / len(labels) if labels else None,
        "abstention": bool(success and not labels and not selected),
        "operational_failure": not success,
        "unsupported_count": len(unsupported),
        "scope_leak_count": len(set(selected) - visible),
        "stale_count": len(stale),
    }


def summarize_corpus(
    corpus: AcceptedCorpus, options: CorpusOptions, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Require the full paired matrix; failures stay in each denominator."""
    coverage = validate_corpus(corpus, options)
    index = _index(corpus)
    expected = {
        (case, query, rep, arm)
        for case, query in index
        for rep in range(1, options.repetitions + 1)
        for arm in STRATEGIES
    }
    observed = {}
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    states: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
    corpus_digest = _digest(corpus.model_dump(mode="json"))
    for row in rows:
        key = (row["case_id"], row["query_id"], row["repetition"], row["strategy"])
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate observation identity")
        case, query, snapshot = index[(key[0], key[1])]
        if (
            row["session_id"] != case.session_id
            or row["snapshot_id"] != query.snapshot_id
            or row["observed_at"] != query.observed_at.isoformat()
            or row["request"] != query.request.model_dump(mode="json")
            or row["expected_fact_ids"] != list(query.expected_fact_ids)
            or row["query_kind"] != query.kind
            or row["corpus_sha256"] != corpus_digest
            or row["fixture_sha256"] != _digest(snapshot.model_dump(mode="json"))
        ):
            raise ValueError("Observation differs from the frozen corpus")
        observed[key] = _score(row, query, snapshot)
        grouped[key[:3]].append(row)
        states[(case.case_id, snapshot.snapshot_id)].add(
            tuple(
                row[name]
                for name in (
                    "fixture_sha256",
                    "state_sha256",
                    "document_vectors_sha256",
                    "sqlite_snapshot_sha256",
                )
            )
        )
    if set(observed) != expected:
        raise ValueError(
            "Missing observations; incomplete matrices cannot produce a complete summary"
        )
    if any(len(values) != 1 for values in states.values()):
        raise ValueError(
            "Accepted state or document vectors changed between queries or repetitions"
        )
    for group in grouped.values():
        for field in (
            "fixture_sha256",
            "state_sha256",
            "document_vectors_sha256",
            "sqlite_snapshot_sha256",
        ):
            if len({row[field] for row in group}) != 1:
                raise ValueError("Strategies did not share identical snapshot bindings")
        vector_hashes = {
            row["query_vector_sha256"] for row in group if row["strategy"] != "lexical"
        }
        if len(vector_hashes) != 1:
            raise ValueError("Vector and hybrid did not share the query vector")
        for row in group:
            if row["result"] is not None and any(
                row[f] is None
                for f in ("state_sha256", "document_vectors_sha256", "sqlite_snapshot_sha256")
            ):
                raise ValueError("A measured result needs snapshot provenance")
            if (
                row["strategy"] != "lexical"
                and row["result"] is not None
                and row["error_type"] is None
                and row["query_vector_sha256"] is None
            ):
                raise ValueError("Measured semantic retrieval needs a query vector binding")
    kinds = {}
    for kind in KINDS:
        keys = [
            (case, query, rep)
            for (case, query), (_, q, _) in index.items()
            if q.kind == kind
            for rep in range(1, options.repetitions + 1)
        ]
        arms, paired, paired_recall = {}, {}, {}
        for arm in STRATEGIES:
            grades = [observed[key + (arm,)] for key in keys]
            metric = "abstention" if kind == "no_answer" else "hit"
            by_rep = []
            for rep in range(1, options.repetitions + 1):
                values = [observed[key + (arm,)][metric] for key in keys if key[2] == rep]
                by_rep.append(
                    {
                        "repetition": rep,
                        "interval": (
                            wilson_interval(sum(values), len(values), options.confidence).as_dict()
                            if values
                            else None
                        ),
                    }
                )
            arms[arm] = {
                "attempted": len(grades),
                "unique_queries": len(keys) // options.repetitions,
                "successes": sum(g[metric] for g in grades),
                "estimate": sum(g[metric] for g in grades) / len(grades) if grades else None,
                "clean_hits": sum(g["clean_hit"] for g in grades),
                "mean_fractional_recall": (
                    sum(g["recall"] for g in grades) / len(grades)
                    if grades and kind != "no_answer"
                    else None
                ),
                **{
                    name: sum(g[name] for g in grades)
                    for name in (
                        "operational_failure",
                        "unsupported_count",
                        "scope_leak_count",
                        "stale_count",
                    )
                },
                "wilson_by_repetition": by_rep,
            }
            if arm != "vector":
                paired[arm] = (
                    paired_bootstrap_interval(
                        [float(observed[key + ("vector",)][metric]) for key in keys],
                        [float(observed[key + (arm,)][metric]) for key in keys],
                        confidence=options.confidence,
                        repetitions=options.bootstrap_repetitions,
                        seed=options.bootstrap_seed,
                        cluster_ids=[index[(key[0], key[1])][0].session_id for key in keys],
                    ).as_dict()
                    if keys
                    else None
                )
                paired_recall[arm] = (
                    paired_bootstrap_interval(
                        [float(observed[key + ("vector",)]["recall"]) for key in keys],
                        [float(observed[key + (arm,)]["recall"]) for key in keys],
                        confidence=options.confidence,
                        repetitions=options.bootstrap_repetitions,
                        seed=options.bootstrap_seed,
                        cluster_ids=[index[(key[0], key[1])][0].session_id for key in keys],
                    ).as_dict()
                    if keys and kind != "no_answer"
                    else None
                )
        kinds[kind] = {
            "strategies": arms,
            "paired_to_vector": paired,
            "paired_fractional_recall_to_vector": paired_recall,
        }
    return {
        "status": "complete",
        "stage": corpus.stage,
        "corpus_sha256": corpus_digest,
        "observations": len(rows),
        "coverage": coverage,
        "quality_kinds": kinds,
        "quality_gate_claimed": False,
        "full_module_ablation_run": False,
        "interval_note": "Wilson intervals describe unique questions within each repetition. Queries within a session can still be dependent. Paired differences resample whole sessions, retaining all queries and repetitions.",
        "state_pair_note": "Declared state variants are recorded controls. All three baseline strategies ignore state; no state-module effectiveness claim follows.",
    }


async def run_accepted_corpus(
    corpus: AcceptedCorpus,
    options: CorpusOptions,
    output: Path,
    provider_factory: Callable[[], EmbeddingProvider],
    *,
    dimension: int,
    binding_files: Sequence[Path],
    provider_identity: dict[str, Any],
) -> dict[str, Any]:
    """Execute a new owned attempt. Refuse reuse; preserve terminal failures."""
    corpus = AcceptedCorpus.model_validate(corpus.model_dump(mode="python"))
    options = CorpusOptions.model_validate(options.model_dump(mode="python"))
    coverage = validate_corpus(corpus, options)
    if type(dimension) is not int or dimension <= 0 or not provider_identity or not binding_files:
        raise ValueError("Explicit provider, dimension and frozen file bindings are required")
    provider_identity = json.loads(json.dumps(provider_identity, allow_nan=False))
    output = output.resolve()
    bindings = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in binding_files
    }
    if any(output == Path(name) or output in Path(name).parents for name in bindings):
        raise ValueError("Output cannot contain a frozen input")
    output.mkdir(exist_ok=False)
    corpus_digest = _digest(corpus.model_dump(mode="json"))
    control: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": corpus_digest,
        "options": options.model_dump(mode="json"),
        "coverage": coverage,
        "bindings": bindings,
        "provider_identity": provider_identity,
        "dimension": dimension,
        "observations": 0,
        "snapshot_failures": [],
    }

    def save(name: str, value: Any) -> None:
        (output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def verify() -> None:
        if any(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
            for name, digest in bindings.items()
        ):
            raise ValueError("Frozen input or source binding changed")

    save("corpus.json", corpus.model_dump(mode="json"))
    save("control.json", control)
    rows: list[dict[str, Any]] = []
    try:
        verify()
        provider = provider_factory()
        for case_number, case in enumerate(corpus.cases):
            for snapshot_number, snapshot in enumerate(
                sorted(case.snapshots, key=lambda s: s.as_of)
            ):
                verify()
                prepared = None
                error_type = None
                try:
                    prepared = build_accepted_snapshot(
                        output / f"accepted-{case_number:04d}-{snapshot_number:04d}.sqlite",
                        snapshot,
                        provider,
                        dimension=dimension,
                    )
                except Exception as error:
                    error_type = type(error).__name__
                    control["snapshot_failures"].append(
                        {
                            "case_id": case.case_id,
                            "snapshot_id": snapshot.snapshot_id,
                            "error_type": error_type,
                        }
                    )
                try:
                    for repetition in range(1, options.repetitions + 1):
                        for query in case.queries:
                            if query.snapshot_id != snapshot.snapshot_id:
                                continue
                            verify()
                            if prepared is None:
                                results = [
                                    {
                                        "query_id": query.query_id,
                                        "repetition": repetition,
                                        "strategy": arm,
                                        "fixture_sha256": _digest(snapshot.model_dump(mode="json")),
                                        "state_sha256": None,
                                        "document_vectors_sha256": None,
                                        "sqlite_snapshot_sha256": None,
                                        "query_vector_sha256": None,
                                        "request": query.request.model_dump(mode="json"),
                                        "expected_fact_ids": list(query.expected_fact_ids),
                                        "selected_ids": [],
                                        "result": None,
                                        "error_type": error_type,
                                        "query_embedding_ms": None,
                                        "retrieval_ms": None,
                                    }
                                    for arm in STRATEGIES
                                ]
                            else:
                                results = await compare_accepted_query(
                                    prepared,
                                    query.request,
                                    query_id=query.query_id,
                                    expected_fact_ids=list(query.expected_fact_ids),
                                    repetition=repetition,
                                )
                            for row in results:
                                row.update(
                                    case_id=case.case_id,
                                    session_id=case.session_id,
                                    snapshot_id=snapshot.snapshot_id,
                                    observed_at=query.observed_at.isoformat(),
                                    query_kind=query.kind,
                                    corpus_sha256=corpus_digest,
                                )
                                with (output / "observations.jsonl").open(
                                    "a", encoding="utf-8"
                                ) as stream:
                                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                                rows.append(row)
                                control["observations"] += 1
                            save("control.json", control)
                finally:
                    if prepared is not None:
                        store = prepared.service.engine.store
                        assert isinstance(store, SQLiteStore)
                        store.close()
        verify()
        summary = summarize_corpus(corpus, options, rows)
        save("summary.json", summary)
        control["status"] = "completed"
        return summary
    except BaseException as error:
        control.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        control["finished_at"] = datetime.now(timezone.utc).isoformat()
        save("control.json", control)
