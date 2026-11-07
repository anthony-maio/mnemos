"""Development-only temporal support comparison over identical frozen inputs."""

from __future__ import annotations

import argparse
import asyncio
from contextvars import ContextVar
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .answer_support_comparison import (
    ObservedJudge,
    SupportTreatment,
    _load_dataset,
    _seed_case,
    summarize,
)
from .support_capacity_comparison import (
    FrozenInput,
    evaluate_input,
    freeze_inputs,
    verify_pairing,
    verify_stores,
)
from mnemos.answer_support import AnswerSupportConfig
from mnemos.memory_safety import source_firewall
from mnemos.utils.embeddings import FastembedProvider
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.ollama_judge import OllamaJudge

MODEL_DIGESTS = {
    "qwen3:14b": "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
}
DATASET_DIGESTS = {
    "answer-support-development-v1": "2c388cb8a603d4a1463496e0a8243a0a22496026c2a0e1d42dca640ad05d9aef",
    "answer-support-temporal-development-v1": "9adc7a464e4387802e2a76e2f63e4d8572ba398da7969f2cc84b15855bc8a807",
}

SupportProtocol = Literal["joint_v1", "qualified_v3", "checked_v4", "contrast_v5"]


def support_configuration(protocol: SupportProtocol) -> AnswerSupportConfig:
    return AnswerSupportConfig(
        protocol=protocol, batching="all" if protocol == "contrast_v5" else "candidate"
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if urlparse(args.endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Temporal comparison requires an explicit loopback endpoint")
    candidate: SupportProtocol
    requested_candidate = getattr(args, "candidate_protocol", "qualified_v3")
    if requested_candidate == "qualified_v3":
        candidate = "qualified_v3"
    elif requested_candidate == "checked_v4":
        candidate = "checked_v4"
    elif requested_candidate == "contrast_v5":
        candidate = "contrast_v5"
    else:
        raise ValueError("Unsupported candidate support protocol")
    candidate_name = {
        "qualified_v3": "qualified",
        "checked_v4": "checked",
        "contrast_v5": "contrast",
    }[candidate]
    protocols: list[tuple[str, SupportProtocol]] = [
        ("joint", "joint_v1"),
        (candidate_name, candidate),
    ]
    root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_paths = [Path(item).resolve() for item in args.dataset]
    datasets = [_load_dataset(path) for path in dataset_paths]
    if len(datasets) != 2 or {data["version"] for data in datasets} != set(DATASET_DIGESTS):
        raise ValueError("The temporal study requires its two declared datasets")
    for path, data in zip(dataset_paths, datasets):
        if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_DIGESTS[data["version"]]:
            raise ValueError("Dataset digest differs from the prospective protocol")
    dataset = {
        "cases": [
            {**case, "dataset_version": data["version"]}
            for data in datasets
            for case in data["cases"]
        ]
    }
    if len({case["id"] for case in dataset["cases"]}) != len(dataset["cases"]):
        raise ValueError("Scenario identities must be distinct across datasets")
    sources = sorted((root / "mnemos").rglob("*.py")) + sorted(
        (root / "benchmarks/local_memory").glob("*.py")
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    bound_paths = [*sources, *dataset_paths, Path(args.protocol).resolve()]
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
        "study": {
            "qualified_v3": "temporal-support-v4",
            "checked_v4": "checked-support-v5",
            "contrast_v5": "contrast-support-v6",
        }[candidate],
        "status": "prepared",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "dataset_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dataset_paths
        },
        "reference_date": "2026-11-06",
        "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        "model_order": order,
        "model_digests": MODEL_DIGESTS,
        "context_length": 8192,
        "max_prompt_bytes": 5120,
        "max_output_tokens": 1024,
        "temperature": 0,
        "think": False,
        "repetitions": 1,
        "support_config": AnswerSupportConfig(batching="candidate").model_dump(),
        "treatments": [support_configuration(protocol).model_dump() for _, protocol in protocols],
        "extraction": "disabled; one accepted-fact store per scenario",
        "model_runs": [],
    }

    def save() -> None:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    services = []
    inputs: list[FrozenInput] = []
    rows: list[dict[str, Any]] = []
    active: ContextVar[dict[str, Any]] = ContextVar("temporal_support_request", default={})
    embedder: FastembedProvider | None = None
    loaded_model: str | None = None
    try:
        embedder = FastembedProvider(model="BAAI/bge-small-en-v1.5", cache_dir=args.embedding_cache)
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
                entries = await freeze_inputs(
                    service, context, case, reference_date=date(2026, 11, 6)
                )
                for entry in entries:
                    record = {**entry.record, "dataset_version": case["dataset_version"]}
                    inputs.append(
                        FrozenInput(
                            entry.service, entry.context, json.dumps(record, sort_keys=True)
                        )
                    )
            if len(inputs) != 176:
                raise ValueError("Unexpected number of frozen temporal-study inputs")
            random.Random(20261106).shuffle(inputs)
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
                        max_tokens=1024,
                        response_observer=observe,
                    )
                    policies = [
                        SupportTreatment(
                            model.replace(":", "-") + "-" + name,
                            ObservedJudge(
                                model.replace(":", "-") + "-" + name,
                                OllamaJudge(
                                    llm,
                                    firewall=source_firewall(),
                                    require_choice_probabilities=True,
                                    max_prompt_bytes=5120,
                                ),
                                output / "judge-requests.jsonl",
                                active,
                            ),
                            support_configuration(protocol),
                        )
                        for name, protocol in protocols
                    ]
                    all_treatments.update(p.name for p in policies)
                    for i, entry in enumerate(inputs):
                        policy_order = policies.copy()
                        identity = "|".join(
                            entry.record[key] for key in ("case_id", "query_id", "strategy")
                        )
                        random.Random("temporal-v4:" + identity).shuffle(policy_order)
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
            if len(rows) != len(inputs) * 2:
                raise ValueError("Temporal-study observations are incomplete")
            verify_pairing(rows, all_treatments)
            for name, digest in manifest["source_sha256"].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Comparison source changed during execution")
            for name, digest in manifest["dataset_sha256"].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Comparison dataset changed during execution")
            if (
                hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
                != manifest["protocol_sha256"]
            ):
                raise ValueError("Comparison protocol changed during execution")
            report = {
                "stage": "development",
                "observations": len(rows),
                "distinct_questions": sum(len(c["queries"]) for c in dataset["cases"]),
                "datasets": {
                    data["version"]: {
                        "distinct_questions": sum(len(c["queries"]) for c in data["cases"]),
                        "results": {
                            name: summarize(
                                [
                                    r
                                    for r in rows
                                    if r["treatment"] == name
                                    and r["dataset_version"] == data["version"]
                                ]
                            )
                            for name in sorted(all_treatments)
                        },
                    }
                    for data in datasets
                },
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
            if embedder is not None:
                embedder.close()
        except BaseException as exc:
            cleanup_errors.append("embedding_close:" + type(exc).__name__)
        manifest["cleanup_errors"] = cleanup_errors
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if cleanup_errors:
            manifest["status"] = "failed"
        save()
        if cleanup_errors:
            raise RuntimeError("Temporal comparison cleanup failed: " + ", ".join(cleanup_errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument(
        "--candidate-protocol",
        choices=["qualified_v3", "checked_v4", "contrast_v5"],
        default="qualified_v3",
    )
    for name in ["output", "endpoint", "source-root", "protocol", "embedding-cache"]:
        parser.add_argument("--" + name, required=True)
    print(json.dumps(asyncio.run(run(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
