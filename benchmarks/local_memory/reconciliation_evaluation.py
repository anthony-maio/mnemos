"""A preclassified reconciliation study, separate from extraction quality."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .extraction_evaluation import ExtractionCase
from .extraction_evaluation_run import (
    MODEL,
    MODEL_DIGEST,
    RecordedProvider,
    append,
    load_cases,
    sha,
    write,
)
from mnemos.extraction.source_reconciliation import ReconciliationDecision, StagedReconciliation
from mnemos.extraction.source_relationship import PromptStyle, reconcile_source_relationships
from mnemos.extraction.source_selection import (
    ExtractiveMemory,
    SelectionInventory,
    render_selections,
)

ARMS: dict[str, tuple[PromptStyle, bool]] = {
    "legacy_full": ("legacy", False),
    "legacy_relevant": ("legacy", True),
    "direct_full": ("direct", False),
    "direct_relevant": ("direct", True),
}


@dataclass(frozen=True)
class ReconciliationCase:
    source: ExtractionCase
    candidate: ExtractiveMemory
    alternatives: tuple[ReconciliationDecision, ...]


def load_reconciliation_cases(path: Path) -> tuple[dict[str, Any], list[ReconciliationCase]]:
    data, sources = load_cases(path)
    if data.get("evaluation") != "preclassified_reconciliation":
        raise ValueError("Expected a preclassified reconciliation corpus")
    cases = []
    for raw, source in zip(data["cases"], sources):
        if source.expected or source.expected_preflight_error:
            raise ValueError("This study does not score extraction or preflight guards")
        inventory = SelectionInventory.model_validate({"selections": raw["selections"]})
        candidates = render_selections(inventory, list(source.events))
        if len(candidates) != 1:
            raise ValueError("This study requires one preclassified candidate per case")
        alternatives = tuple(
            ReconciliationDecision.model_validate(value) for value in raw["expected_alternatives"]
        )
        if not alternatives or any(value.candidate_index != 0 for value in alternatives):
            raise ValueError("Expected alternatives must address the single candidate")
        targets = {target.id: target for target in source.targets}
        for value in alternatives:
            if value.target_fact_id is not None:
                target = targets.get(value.target_fact_id)
                if target is None or target.version != value.expected_version:
                    raise ValueError("Expected alternative cites an unobserved target version")
        cases.append(ReconciliationCase(source, candidates[0], alternatives))
    return data, cases


def grade_reconciliation(
    case: ReconciliationCase,
    result: list[StagedReconciliation] | None,
    *,
    error: str | None,
) -> dict[str, bool]:
    if (result is None) == (error is None):
        raise ValueError("Supply exactly one result or error")
    valid = False
    exact = False
    if result is not None and len(result) == 1:
        item, candidate = result[0], case.candidate
        valid = (
            item.candidate_index == 0
            and item.source_id == candidate.source_id
            and item.source_role == candidate.source_role
            and item.source_kind == candidate.kind
            and item.content == candidate.content
            and item.revises_existing_requirement == candidate.revises_existing_requirement
        )
        actual = (item.operation, item.target_fact_id, item.expected_version)
        exact = valid and any(
            actual == (expected.operation, expected.target_fact_id, expected.expected_version)
            for expected in case.alternatives
        )
    return {
        "operational_success": error is None,
        "provenance_valid": valid,
        "case_exact": exact,
        "correct_abstention": bool(exact and result and result[0].operation == "unresolved"),
        "unsafe_target_choice": bool(
            result
            and not exact
            and any(item.operation in {"correct", "support"} for item in result)
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    for endpoint in (args.endpoint, args.tokenizer_url):
        url = urlsplit(endpoint)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.username
            or url.password
            or url.path
            or url.query
            or url.fragment
        ):
            raise ValueError("Explicit HTTP IPv4 loopback ports are required")
    root = Path(args.source_root).resolve()
    dataset, protocol = Path(args.dataset).resolve(), Path(args.protocol).resolve()
    if sha(dataset) != args.dataset_sha256:
        raise ValueError("Dataset differs from the frozen digest")
    data, cases = load_reconciliation_cases(dataset)
    sources = sorted((root / "mnemos").rglob("*.py")) + sorted(
        (root / "benchmarks/local_memory").glob("*.py")
    )
    bound = [*sources, dataset, protocol]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(p.relative_to(root)) for p in bound]],
        cwd=root,
        text=True,
    ).strip():
        raise ValueError("Commit all bound source, data and protocol before inference")
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        response = await client.get(args.endpoint + "/api/tags")
        response.raise_for_status()
        model = next(item for item in response.json()["models"] if item["name"] == MODEL)
        if model["digest"] != MODEL_DIGEST:
            raise ValueError("Installed model digest differs from protocol")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "stage": data["stage"],
        "evaluation": data["evaluation"],
        "dataset_version": data["version"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {p.relative_to(root).as_posix(): sha(p) for p in sources},
        "dataset_sha256": sha(dataset),
        "protocol_sha256": sha(protocol),
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "temperature": 0,
        "think": False,
        "context_length": 8192,
        "max_output_tokens": 2048,
        "max_counted_input_tokens": 5632,
        "case_timeout_seconds": 60,
        "repetitions": args.repetitions,
        "arms": ARMS,
        "storage_writes": False,
        "extraction_calls": 0,
        "retrieval_calls": 0,
        "embedding_calls": 0,
        "cases": len(cases),
        "clusters": len({case.source.cluster_id for case in cases}),
        "shuffle_seed": 2026110701,
    }
    write(output / "manifest.json", manifest)
    provider = RecordedProvider(args.endpoint, args.tokenizer_url, output)
    observations = []
    try:
        order = [
            (repetition, case, arm)
            for repetition in range(args.repetitions)
            for case in cases
            for arm in ARMS
        ]
        random.Random(manifest["shuffle_seed"]).shuffle(order)
        write(
            output / "order.json",
            [(repetition, case.source.case_id, arm) for repetition, case, arm in order],
        )
        for repetition, case, arm in order:
            source = case.source
            provider.case_id, provider.repetition = source.case_id + "/" + arm, repetition
            before_calls = provider.call_id
            snapshot = {
                "events": [event.model_dump(mode="json") for event in source.events],
                "targets": [target.model_dump(mode="json") for target in source.targets],
                "candidates": [asdict(case.candidate)],
            }
            result = None
            error = None
            started = time.perf_counter()
            try:
                style, relevant = ARMS[arm]
                result = await asyncio.wait_for(
                    reconcile_source_relationships(
                        [case.candidate],
                        [event.model_copy(deep=True) for event in source.events],
                        [target.model_copy(deep=True) for target in source.targets],
                        provider,
                        prompt_style=style,
                        relevant_revision_flag=relevant,
                    ),
                    timeout=60,
                )
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
            observation = {
                "case_id": source.case_id,
                "cluster_id": source.cluster_id,
                "arm": arm,
                "repetition": repetition,
                "input": snapshot,
                "input_sha256": hashlib.sha256(
                    json.dumps(snapshot, sort_keys=True).encode()
                ).hexdigest(),
                "expected_alternatives": [value.model_dump() for value in case.alternatives],
                "provider_call_ids": list(range(before_calls + 1, provider.call_id + 1)),
                "error": error,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "grade": grade_reconciliation(case, result, error=error),
                "result": [value.model_dump() for value in result] if result is not None else None,
            }
            append(output / "observations.jsonl", observation)
            observations.append(observation)
            print(json.dumps({"completed": len(observations), "planned": len(order)}), flush=True)
        for name, digest in manifest["source_sha256"].items():
            if sha(root / name) != digest:
                raise RuntimeError("Bound source changed during evaluation")
        if (
            sha(dataset) != manifest["dataset_sha256"]
            or sha(protocol) != manifest["protocol_sha256"]
        ):
            raise RuntimeError("Data or protocol changed during evaluation")
        manifest.update(
            status="completed", observations=len(observations), provider_calls=provider.call_id
        )
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = {p.name: sha(p) for p in output.glob("*.jsonl")}
        write(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "dataset-sha256", "protocol", "output", "endpoint", "tokenizer-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=3)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
