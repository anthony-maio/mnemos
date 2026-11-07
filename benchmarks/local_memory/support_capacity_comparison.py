"""Development-only model comparison over one frozen set of retrieved inputs."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .answer_support_comparison import (
    ObservedJudge,
    SupportTreatment,
    _load_dataset,
    _seed_case,
    grade_selection,
    summarize,
)
from .retrieval_comparison import STRATEGIES, comparison_snapshots
from mnemos.answer_support import AnswerSupportConfig, assess_answer_support
from mnemos.contracts import RetrievalRequest
from mnemos.memory_safety import source_firewall
from mnemos.memory_service import LocalMemoryService
from mnemos.scopes import ScopeContext
from mnemos.utils.embeddings import FastembedProvider
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.ollama_judge import OllamaJudge

MODEL_DIGESTS = {
    "qwen3:4b": "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
    "qwen3:14b": "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
}


@dataclass(frozen=True)
class FrozenInput:
    """Immutable JSON plus the original accepted store used for support checks."""

    service: LocalMemoryService
    context: ScopeContext
    record_json: str

    @property
    def record(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.record_json)
        return value


async def freeze_inputs(
    service: LocalMemoryService,
    context: ScopeContext,
    case: dict[str, Any],
    *,
    reference_date: date | None = None,
) -> list[FrozenInput]:
    inputs = []
    for query in case["queries"]:
        with comparison_snapshots(service) as (digest, variants):
            order = list(STRATEGIES)
            random.Random(f"capacity-v3:{case['id']}:{query['id']}").shuffle(order)
            for strategy in order:
                started = time.perf_counter()
                baseline = await variants[strategy].retrieve(
                    RetrievalRequest(
                        query=query["query"],
                        context=context,
                        top_k=5,
                        max_context_tokens=1200,
                    )
                )
                ids = [fact["id"] for fact in baseline["facts"]]
                record = {
                    "case_id": case["id"],
                    "query_id": query["id"],
                    "query": query["query"],
                    "category": query.get("category"),
                    "expected_fact_ids": query["expected_fact_ids"],
                    "strategy": strategy,
                    "snapshot_sha256": digest,
                    "baseline_ids": ids,
                    "baseline_facts": baseline["facts"],
                    "baseline_status": "unavailable" if baseline.get("degraded") else "assessed",
                    "retrieval_ms": (time.perf_counter() - started) * 1000,
                    "token_count_method": baseline["token_count_method"],
                    "baseline_context_upper_bound": baseline["context_token_upper_bound"],
                }
                record["baseline_grade"] = grade_selection(
                    query["expected_fact_ids"], ids, record["baseline_status"]
                )
                if reference_date is not None:
                    record["reference_date"] = reference_date.isoformat()
                inputs.append(
                    FrozenInput(
                        service, context.model_copy(deep=True), json.dumps(record, sort_keys=True)
                    )
                )
    return inputs


async def evaluate_input(
    entry: FrozenInput, policy: SupportTreatment, model: str
) -> dict[str, Any]:
    record = entry.record
    previous = policy.judge.count if isinstance(policy.judge, ObservedJudge) else 0
    started = time.perf_counter()
    support = await assess_answer_support(
        record["query"],
        record["baseline_ids"],
        store=entry.service.engine.store,
        registry=entry.service.registry,
        context=entry.context,
        judge=policy.judge,
        config=policy.config,
        reference_date=(
            date.fromisoformat(record["reference_date"]) if "reference_date" in record else None
        ),
    )
    selected = [candidate.fact_id for candidate in support.supported]
    status = support.status if record["baseline_status"] == "assessed" else "unavailable"
    return {
        **record,
        "model": model,
        "treatment": policy.name,
        "input_sha256": hashlib.sha256(entry.record_json.encode()).hexdigest(),
        "judge_request_ids": (
            [f"{policy.name}:{i}" for i in range(previous + 1, policy.judge.count + 1)]
            if isinstance(policy.judge, ObservedJudge)
            else []
        ),
        "supported_ids": selected,
        "support_grade": grade_selection(record["expected_fact_ids"], selected, status),
        "support_status": status,
        "support_error_type": support.error_type,
        "support_decisions": [asdict(item) for item in support.decisions],
        "support_trace": support.trace.calls,
        "support_ms": (time.perf_counter() - started) * 1000,
    }


def verify_pairing(rows: list[dict[str, Any]], treatments: set[str]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["query_id"], row["strategy"])].append(row)
    if not groups:
        raise ValueError("No comparison observations")
    for group in groups.values():
        if len(group) != len(treatments) or {r["treatment"] for r in group} != treatments:
            raise ValueError("Incomplete or duplicated treatment group")
        if len({r["input_sha256"] for r in group}) != 1:
            raise ValueError("Models received different frozen inputs")


def verify_stores(inputs: list[FrozenInput]) -> None:
    seen = set()
    for entry in inputs:
        identity = id(entry.service)
        if identity in seen:
            continue
        seen.add(identity)
        with comparison_snapshots(entry.service) as (digest, _):
            if digest != entry.record["snapshot_sha256"]:
                raise ValueError("Accepted store changed after input freeze")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if urlparse(args.endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Capacity comparison requires an explicit loopback endpoint")
    root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_path = Path(args.dataset).resolve()
    dataset = _load_dataset(dataset_path)
    sources = sorted((root / "mnemos").rglob("*.py")) + sorted(
        (root / "benchmarks/local_memory").glob("*.py")
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    bound_paths = [*sources, dataset_path, Path(args.protocol).resolve()]
    for path in bound_paths:
        relative = path.relative_to(root).as_posix()
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        if committed.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Comparison inputs must be committed before inference: " + relative)
    order = list(MODEL_DIGESTS)
    random.Random(20261106).shuffle(order)
    manifest: dict[str, Any] = {
        "stage": "development",
        "study": "model-capacity-support-v3",
        "status": "prepared",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        "model_order": order,
        "model_digests": MODEL_DIGESTS,
        "context_length": 8192,
        "max_prompt_bytes": 4096,
        "max_output_tokens": 2048,
        "temperature": 0,
        "think": False,
        "repetitions": 1,
        "support_config": AnswerSupportConfig(batching="candidate").model_dump(),
        "extraction": "disabled; one accepted-fact store per scenario",
        "model_runs": [],
    }

    def save() -> None:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    services = []
    inputs: list[FrozenInput] = []
    rows: list[dict[str, Any]] = []
    active: ContextVar[dict[str, Any]] = ContextVar("capacity_support_request", default={})
    embedder = FastembedProvider(model="BAAI/bge-small-en-v1.5", cache_dir=args.embedding_cache)
    loaded_model: str | None = None
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
            response = await client.get(args.endpoint + "/api/version")
            response.raise_for_status()
            if response.json() != {"version": "0.40.9"}:
                raise ValueError("Unexpected Ollama version")
            response = await client.get(args.endpoint + "/api/tags")
            response.raise_for_status()
            models = {m["name"]: m for m in response.json()["models"]}
            if any(models.get(m, {}).get("digest") != h for m, h in MODEL_DIGESTS.items()):
                raise ValueError("Model digest differs from prospective protocol")
            manifest["model_metadata"] = {m: models[m] for m in order}
            for case in dataset["cases"]:
                service, context = _seed_case(output / "accepted" / case["id"], case, embedder)
                services.append(service)
                service.registry.save(output / "accepted" / case["id"] / "registry.json")
                inputs.extend(await freeze_inputs(service, context, case))
            (output / "frozen-inputs.jsonl").write_text(
                "".join(i.record_json + "\n" for i in inputs), encoding="utf-8"
            )
            verify_stores(inputs)
            manifest.update(
                status="running",
                input_count=len(inputs),
                frozen_inputs_sha256=hashlib.sha256(
                    (output / "frozen-inputs.jsonl").read_bytes()
                ).hexdigest(),
            )
            save()
            all_treatments: set[str] = set()
            with (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
                for model in order:
                    response = await client.get(args.endpoint + "/api/ps")
                    response.raise_for_status()
                    if response.json().get("models"):
                        raise ValueError(
                            "Expected private runtime to be unloaded before model block"
                        )
                    loaded_model = model
                    response = await client.post(
                        args.endpoint + "/api/generate",
                        json={"model": model, "prompt": "", "stream": False, "keep_alive": "1h"},
                    )
                    response.raise_for_status()
                    response = await client.get(args.endpoint + "/api/ps")
                    response.raise_for_status()
                    loaded = response.json()["models"]
                    if (
                        len(loaded) != 1
                        or loaded[0]["digest"] != MODEL_DIGESTS[model]
                        or loaded[0]["context_length"] != 8192
                    ):
                        raise ValueError("Loaded model or context differs from protocol")
                    model_run = {"model": model, "active_before": loaded, "status": "running"}
                    manifest["model_runs"].append(model_run)
                    save()

                    def observe(payload: dict[str, Any]) -> None:
                        with (output / "provider-responses.jsonl").open(
                            "a", encoding="utf-8"
                        ) as raw:
                            raw.write(
                                json.dumps(
                                    {**payload, "support_request": active.get()}, ensure_ascii=True
                                )
                                + "\n"
                            )

                    llm = OllamaProvider(
                        model=model,
                        base_url=args.endpoint,
                        think=False,
                        temperature=0,
                        max_tokens=2048,
                        response_observer=observe,
                    )
                    protocols: list[tuple[str, Literal["joint_v1", "separate_v2"]]] = [
                        ("joint", "joint_v1"),
                        ("separate", "separate_v2"),
                    ]
                    policies = [
                        SupportTreatment(
                            model.replace(":", "-") + "-" + name,
                            ObservedJudge(
                                model.replace(":", "-") + "-" + name,
                                OllamaJudge(
                                    llm,
                                    firewall=source_firewall(),
                                    require_choice_probabilities=True,
                                    max_prompt_bytes=4096,
                                ),
                                output / "judge-requests.jsonl",
                                active,
                            ),
                            AnswerSupportConfig(protocol=protocol, batching="candidate"),
                        )
                        for name, protocol in protocols
                    ]
                    all_treatments.update(p.name for p in policies)
                    for i, entry in enumerate(inputs):
                        policy_order = policies.copy()
                        random.Random(f"capacity-v3:{i}:policies").shuffle(policy_order)
                        for policy in policy_order:
                            row = await evaluate_input(entry, policy, model)
                            rows.append(row)
                            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                        stream.flush()
                        print(
                            json.dumps(
                                {
                                    "model": model,
                                    "inputs_completed": i + 1,
                                    "observations": len(rows),
                                }
                            ),
                            flush=True,
                        )
                    verify_stores(inputs)
                    response = await client.get(args.endpoint + "/api/ps")
                    response.raise_for_status()
                    model_run.update(status="completed", active_after=response.json())
                    response = await client.post(
                        args.endpoint + "/api/generate",
                        json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                    )
                    response.raise_for_status()
                    loaded_model = None
                    save()
            verify_pairing(rows, all_treatments)
            for name, digest in manifest["source_sha256"].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Comparison source changed during execution")
            report = {
                "stage": "development",
                "observations": len(rows),
                "distinct_questions": sum(len(c["queries"]) for c in dataset["cases"]),
                "results": {
                    name: summarize([r for r in rows if r["treatment"] == name])
                    for name in sorted(all_treatments)
                },
            }
            (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            manifest.update(
                status="completed",
                observations_sha256=hashlib.sha256(
                    (output / "observations.jsonl").read_bytes()
                ).hexdigest(),
            )
            save()
            return report
    except BaseException as exc:
        manifest.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            completed_observations=len(rows),
        )
        save()
        raise
    finally:
        cleanup_errors = []
        try:
            if loaded_model is not None:
                async with httpx.AsyncClient(trust_env=False, timeout=30) as cleanup:
                    response = await cleanup.post(
                        args.endpoint + "/api/generate",
                        json={
                            "model": loaded_model,
                            "prompt": "",
                            "stream": False,
                            "keep_alive": 0,
                        },
                    )
                    response.raise_for_status()
                    loaded_model = None
        except BaseException as exc:
            cleanup_errors.append("model_unload:" + type(exc).__name__)
        for service in services:
            try:
                await service.engine.aclose()
            except BaseException as exc:
                cleanup_errors.append("service_close:" + type(exc).__name__)
        try:
            embedder.close()
        except BaseException as exc:
            cleanup_errors.append("embedding_close:" + type(exc).__name__)
        manifest["cleanup_errors"] = cleanup_errors
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if cleanup_errors:
            manifest["status"] = "failed"
        save()
        if cleanup_errors:
            raise RuntimeError("Capacity comparison cleanup failed: " + ", ".join(cleanup_errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "output", "endpoint", "source-root", "protocol", "embedding-cache"]:
        parser.add_argument("--" + name, required=True)
    print(json.dumps(asyncio.run(run(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
