"""Development comparison of retrieval with and without cited answer support.

Facts are explicitly seeded; no extraction runs here. This is not the fresh
held-out evaluation or a replacement for any historical score.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.config import MnemosConfig
from mnemos.contracts import RetrievalRequest
from mnemos.engine import MnemosEngine
from mnemos.judgment import Judge
from mnemos.memory_safety import source_firewall
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext, ScopeRegistry, context_metadata
from mnemos.types import MemoryChunk
from mnemos.utils.embeddings import EmbeddingProvider, FastembedProvider
from mnemos.utils.llm import OllamaProvider, UnavailableLLMProvider
from mnemos.utils.ollama_judge import OllamaJudge
from mnemos.utils.storage import SQLiteStore

from .retrieval_comparison import STRATEGIES, comparison_snapshots


def grade_selection(expected: list[str], selected: list[str], status: str) -> dict[str, Any]:
    expected_set, selected_set = set(expected), set(selected)
    return {
        "answerable": bool(expected_set),
        "answerable_hit": bool(expected_set & selected_set) and status == "assessed",
        "correct_abstention": not expected_set and not selected_set and status == "assessed",
        "supported_count": len(expected_set & selected_set),
        "unsupported_count": len(selected_set - expected_set),
        "expected_count": len(expected_set),
        "operational_success": status == "assessed",
    }


async def compare_query(
    service: LocalMemoryService,
    context: ScopeContext,
    query: dict[str, Any],
    *,
    judge: Judge,
    support_config: AnswerSupportConfig,
    top_k: int = 5,
    max_context_tokens: int = 1200,
    order_seed: str = "development",
) -> list[dict[str, Any]]:
    rows = []
    with comparison_snapshots(service) as (digest, variants):
        order = list(STRATEGIES)
        random.Random(f"{order_seed}:{query['id']}").shuffle(order)
        for strategy in order:
            variant = variants[strategy]
            started = time.perf_counter()
            baseline = await variant.retrieve(
                RetrievalRequest(
                    query=query["query"],
                    context=context,
                    top_k=top_k,
                    max_context_tokens=max_context_tokens,
                )
            )
            retrieval_ms = (time.perf_counter() - started) * 1000
            ids = [item["id"] for item in baseline["facts"]]
            started = time.perf_counter()
            support = await assess_answer_support(
                query["query"],
                ids,
                store=variant.engine.store,
                registry=variant.registry,
                context=context,
                judge=judge,
                config=support_config,
            )
            selected = [item.fact_id for item in support.supported]
            baseline_status = "unavailable" if baseline.get("degraded") else "assessed"
            status = support.status if baseline_status == "assessed" else "unavailable"
            rows.append(
                {
                    "query_id": query["id"],
                    "query": query["query"],
                    "category": query.get("category"),
                    "strategy": strategy,
                    "snapshot_sha256": digest,
                    "expected_fact_ids": query["expected_fact_ids"],
                    "baseline_ids": ids,
                    "supported_ids": selected,
                    "baseline_facts": baseline["facts"],
                    "baseline_grade": grade_selection(
                        query["expected_fact_ids"], ids, baseline_status
                    ),
                    "support_grade": grade_selection(query["expected_fact_ids"], selected, status),
                    "support_status": status,
                    "support_error_type": support.error_type,
                    "support_decisions": [asdict(item) for item in support.decisions],
                    "support_trace": support.trace.calls,
                    "retrieval_ms": retrieval_ms,
                    "support_ms": (time.perf_counter() - started) * 1000,
                    "token_count_method": baseline["token_count_method"],
                    "baseline_context_upper_bound": baseline["context_token_upper_bound"],
                }
            )
    return rows


def _seed_case(
    root: Path, case: dict[str, Any], embedder: EmbeddingProvider
) -> tuple[LocalMemoryService, ScopeContext]:
    project = root / "project"
    foreign = root / "foreign"
    project.mkdir(parents=True)
    foreign.mkdir()
    # Output commonly lives inside the source checkout. Plain subdirectories
    # would inherit that repository's identity and collapse the scope control.
    for directory in (project, foreign):
        subprocess.run(["git", "init", "-q", str(directory)], check=True, capture_output=True)
    registry = ScopeRegistry()
    registry.register_project(project, user_profile_id="development", enabled=True)
    registry.register_project(foreign, user_profile_id="development", enabled=True)
    context = registry.resolve(
        project, user_profile_id="development", session_id=case["id"], host="manual"
    )
    foreign_context = registry.resolve(
        foreign, user_profile_id="development", session_id=case["id"], host="manual"
    )
    if context.project_id == foreign_context.project_id:
        raise ValueError("Development project identities must be distinct")
    config = MnemosConfig()
    config.modules.surprisal = False
    config.modules.sleep = False
    config.modules.mutable_rag = False
    config.modules.affective = False
    config.retrieval.query_state_source = "none"
    store = SQLiteStore(str(root / "accepted.sqlite"))
    try:
        engine = MnemosEngine(
            config=config, embedder=embedder, llm=UnavailableLLMProvider(), store=store
        )
        for item in case["facts"]:
            fact_context = foreign_context if item.get("scope") == "foreign" else context
            chunk = MemoryChunk(
                id=item["id"],
                content=item["content"],
                embedding=embedder.embed(item["content"]),
                metadata={
                    **context_metadata(fact_context, "project"),
                    "assertion_status": item.get("assertion_status", "asserted"),
                    "source": "accepted_development_fixture",
                },
            )
            store.store(chunk)
            store._conn.execute(
                "INSERT INTO fact_evidence "
                "(evidence_id,fact_id,source_event_id,excerpt,source_ref,created_at) VALUES (?,?,?,?,?,?)",
                (
                    item["id"] + "-evidence",
                    item["id"],
                    None,
                    item.get("evidence", item["content"]),
                    "fixture:" + case["id"] + ":" + item["id"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            store._conn.commit()
            engine.refresh_graph_chunk(chunk)
        while engine.rebuild_graph_batch(limit=100):
            pass
        return LocalMemoryService(engine, registry), context
    except BaseException:
        store.close()
        raise


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for strategy in STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        arms = {}
        for arm in ("baseline", "support"):
            grades = [row[arm + "_grade"] for row in group]
            answerable = [grade for grade in grades if grade["answerable"]]
            absent = [grade for grade in grades if not grade["answerable"]]
            arms[arm] = {
                "attempted": len(grades),
                "answerable_hits": sum(grade["answerable_hit"] for grade in answerable),
                "answerable_queries": len(answerable),
                "correct_abstentions": sum(grade["correct_abstention"] for grade in absent),
                "no_answer_queries": len(absent),
                "unsupported_results": sum(grade["unsupported_count"] for grade in grades),
                "operational_failures": sum(not grade["operational_success"] for grade in grades),
            }
        summary[strategy] = arms
    return summary


def _load_dataset(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Development dataset must be an object")
    data: dict[str, Any] = value
    if data.get("stage") != "development" or not data.get("cases"):
        raise ValueError("This runner requires an explicitly labeled development dataset")
    case_ids = [case["id"] for case in data["cases"]]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate development case IDs")
    for case in data["cases"]:
        if not case["id"].replace("-", "").isalnum():
            raise ValueError("Case IDs must be plain alphanumeric names")
        ids = [fact["id"] for fact in case["facts"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate fact IDs")
        accepted = {
            fact["id"]
            for fact in case["facts"]
            if fact.get("scope", "project") == "project"
            and fact.get("assertion_status", "asserted") == "asserted"
        }
        query_ids = [query["id"] for query in case["queries"]]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Duplicate query IDs")
        for query in case["queries"]:
            if not set(query["expected_fact_ids"]) <= accepted:
                raise ValueError("Expected answer must identify an accepted local fact")
    return data


async def run_development(args: argparse.Namespace) -> dict[str, Any]:
    if urlparse(args.endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("This development runner requires an explicit loopback model endpoint")
    dataset_path = Path(args.dataset)
    dataset = _load_dataset(dataset_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[2]
    sources = sorted((project_root / "mnemos").rglob("*.py")) + sorted(
        Path(__file__).parent.glob("*.py")
    )
    manifest: dict[str, Any] = {
        "namespace": "answer-support-development-v1",
        "status": "prepared",
        "stage": "development",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": args.model,
        "endpoint": args.endpoint,
        "embedding_model": args.embedding_model,
        "temperature": 0,
        "think": False,
        "max_output_tokens": 2048,
        "repetitions": args.repetitions,
        "top_k": 5,
        "context_budget": 1200,
        "support_config": AnswerSupportConfig().model_dump(),
        "extraction": "disabled; accepted fixture facts",
        "full_profile_limit": "spreading enabled; affective disabled; no inference about affective utility",
    }
    manifest_path = output / "manifest.json"

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    rows: list[dict[str, Any]] = []
    embedder: EmbeddingProvider | None = None
    llm: OllamaProvider | None = None
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(args.endpoint.rstrip("/") + "/api/tags")
            response.raise_for_status()
            models = response.json()["models"]
            manifest["model_identity"] = next(item for item in models if item["name"] == args.model)
            response = await client.get(args.endpoint.rstrip("/") + "/api/version")
            response.raise_for_status()
            manifest["server_version"] = response.json()

        def observe(payload: dict[str, Any]) -> None:
            with (output / "provider-responses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=True) + "\n")

        llm = OllamaProvider(
            model=args.model,
            base_url=args.endpoint,
            think=False,
            temperature=0,
            max_tokens=2048,
            response_observer=observe,
        )
        judge = OllamaJudge(llm, firewall=source_firewall())
        embedder = FastembedProvider(model=args.embedding_model, cache_dir=args.embedding_cache)
        manifest["status"] = "running"
        save_manifest()
        with (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
            for repetition in range(args.repetitions):
                for case in dataset["cases"]:
                    service, context = _seed_case(
                        output / f"r{repetition}" / case["id"], case, embedder
                    )
                    try:
                        for query in case["queries"]:
                            observations = await compare_query(
                                service,
                                context,
                                query,
                                judge=judge,
                                support_config=AnswerSupportConfig(),
                                order_seed=f"{case['id']}:{repetition}",
                            )
                            for row in observations:
                                row.update(case_id=case["id"], repetition=repetition)
                                rows.append(row)
                                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                            stream.flush()
                            print(
                                json.dumps(
                                    {
                                        "case": case["id"],
                                        "query": query["id"],
                                        "observations": len(rows),
                                    }
                                ),
                                flush=True,
                            )
                    finally:
                        await service.engine.aclose()
        report = {"stage": "development", "observations": len(rows), "results": summarize(rows)}
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest["status"] = "completed"
        manifest["observations_sha256"] = hashlib.sha256(
            (output / "observations.jsonl").read_bytes()
        ).hexdigest()
        save_manifest()
        return report
    except BaseException as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, completed_observations=len(rows)
        )
        save_manifest()
        raise
    finally:
        if embedder is not None:
            embedder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    print(json.dumps(asyncio.run(run_development(args)), indent=2))


if __name__ == "__main__":
    main()
