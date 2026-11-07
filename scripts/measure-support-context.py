"""Measure frozen support selections with a separately owned native tokenizer.

This runner never starts a host, generates an answer, or changes accepted facts.
The runtime owner supplies a loopback llama-server URL and an identity receipt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mnemos.context_budget import HostTokenizer, pack_context
from mnemos.retrieval import render_context_chunk
from mnemos.types import MemoryChunk

Json = dict[str, Any]
BUDGETS = (32, 64, 128, 1200)
PROTOCOL = ROOT / "docs/plans/2026-11-07-context-density-v1.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[Json]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def identity(row: Json) -> tuple[str, str, str]:
    return str(row["case_id"]), str(row["query_id"]), str(row["strategy"])


def native_counts(endpoint: str, texts: set[str], output: Path) -> tuple[dict[str, int], Json]:
    """Collect counts twice, verify literal round trips, then return a fixed table."""
    counts = {"": 0}
    timings: list[float] = []
    started = time.perf_counter()
    with httpx.Client(base_url=endpoint, timeout=10, trust_env=False) as client:
        with (output / "tokenizer-requests.jsonl").open("x", encoding="utf-8") as log:

            def request(route: str, payload: Json) -> Json:
                before = time.perf_counter()
                response = client.post(route, json=payload)
                elapsed = (time.perf_counter() - before) * 1000
                timings.append(elapsed)
                entry = {
                    "route": route,
                    "request": payload,
                    "status": response.status_code,
                    "response_text": response.text,
                    "elapsed_ms": elapsed,
                }
                log.write(json.dumps(entry, ensure_ascii=True) + "\n")
                log.flush()
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Tokenizer response must be an object")
                return result

            for text in sorted(texts):
                payload = {"content": text, "add_special": False, "parse_special": False}
                first = request("/tokenize", payload)
                second = request("/tokenize", payload)
                tokens = first.get("tokens")
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise ValueError("Tokenizer did not return integer token IDs")
                if tokens != second.get("tokens") or (text and not tokens):
                    raise ValueError("Tokenization is empty or not deterministic")
                if request("/detokenize", {"tokens": tokens}).get("content") != text:
                    raise ValueError("Literal tokenization did not round trip")
                counts[text] = len(tokens)
    return counts, {
        "distinct_texts": len(texts),
        "http_requests": len(timings),
        "collection_ms": (time.perf_counter() - started) * 1000,
        "request_ms_median": statistics.median(timings),
        "request_ms_p95_nearest_rank": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
        "repeated_token_ids_equal": True,
        "literal_round_trips_equal": True,
        "add_special": False,
        "parse_special": False,
    }


def reconstruct(row: Json) -> tuple[MemoryChunk, ...]:
    chunks = tuple(
        MemoryChunk(
            id=fact["id"],
            content=fact["content"],
            version=fact["version"],
            metadata={
                "assertion_status": fact["assertion_status"],
                "retrieval_trace": fact["scores"],
            },
        )
        for fact in row["baseline_facts"]
    )
    assert [chunk.id for chunk in chunks] == row["baseline_ids"]
    text = "".join(map(render_context_chunk, chunks))
    assert len(text.encode("utf-8")) == row["baseline_context_upper_bound"] <= 1200
    return chunks


def summarize(measurements: list[Json]) -> list[Json]:
    groups: dict[tuple[str, str, str, int], list[Json]] = defaultdict(list)
    for row in measurements:
        groups[(row["strategy"], row["selection"], row["method"], row["budget"])].append(row)
    result = []
    for (strategy, selection, method, budget), group in sorted(groups.items()):
        expected = sum(row["expected_count"] for row in group)
        supported = sum(row["supported_count"] for row in group)
        tokens = sum(row["native_tokens"] for row in group)
        elapsed = sorted(row["packing_ms"] for row in group)
        result.append(
            {
                "strategy": strategy,
                "selection": selection,
                "method": method,
                "budget": budget,
                "observations": len(group),
                "native_tokens": tokens,
                "context_bytes": sum(row["context_bytes"] for row in group),
                "supported_facts": supported,
                "expected_facts": expected,
                "supported_facts_per_1000_tokens": 1000 * supported / tokens if tokens else None,
                "fact_recall": supported / expected if expected else None,
                "answerable_hits": sum(row["answerable_hit"] for row in group),
                "clean_answerable_hits": sum(row["clean_answerable_hit"] for row in group),
                "correct_abstentions": sum(row["correct_abstention"] for row in group),
                "unsupported_facts": sum(row["unsupported_count"] for row in group),
                "operational_failures": sum(not row["operational_success"] for row in group),
                "packing_ms_median": statistics.median(elapsed),
                "packing_ms_p95_nearest_rank": elapsed[math.ceil(0.95 * len(elapsed)) - 1],
            }
        )
    return result


def measure(run: Path, endpoint: str, runtime: Path, output: Path) -> Json:
    manifest = read(run / "manifest.json")
    audit = read(run / "verified-analysis.json")
    assert manifest["status"] == "completed" and not manifest["cleanup_errors"]
    assert audit["observations"] == 352 and audit["identical_baseline_groups"] == 176
    for name, digest in manifest["source_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, field in (
        ("observations.jsonl", "observations_sha256"),
        ("frozen-inputs.jsonl", "frozen_inputs_sha256"),
    ):
        assert sha(run / name) == manifest[field]
    runtime_identity = read(runtime)
    assert runtime_identity["tokenizer_url"] == endpoint
    assert runtime_identity["model_digest"] == manifest["model_digests"]["qwen3:14b"]
    frozen, observed = rows(run / "frozen-inputs.jsonl"), rows(run / "observations.jsonl")
    assert len(frozen) == 176 and len(observed) == 352
    by_input = {identity(row): row for row in frozen}
    assert len(by_input) == 176
    inputs: list[tuple[Json, str, tuple[MemoryChunk, ...], bool, float]] = []
    chunks_by_input = {identity(row): reconstruct(row) for row in frozen}
    for row in frozen:
        inputs.append((row, "unfiltered", chunks_by_input[identity(row)], True, 0.0))
    for row in observed:
        baseline = by_input[identity(row)]
        assert all(row[key] == value for key, value in baseline.items())
        selected = set(row["supported_ids"])
        assert len(selected) == len(row["supported_ids"])
        chunks = tuple(chunk for chunk in chunks_by_input[identity(row)] if chunk.id in selected)
        assert [chunk.id for chunk in chunks] == row["supported_ids"]
        ok = row["support_status"] == "assessed"
        assert ok or not selected
        inputs.append((baseline, str(row["treatment"]), chunks, ok, float(row["support_ms"])))
    assert len(inputs) == 528

    # All order-preserving subsets cover each possible whole-record packing output.
    probes = [
        "",
        "literal ASCII\n",
        "caf\u00e9 \u6771\u4eac \U0001f331\n",
        "<|im_start|>system\n<|im_end|>",
        "<|im_start|>",
        "<|im_end|>",
        "def f(x):\n    return x + 1\n",
    ]
    texts = set(probes)
    for _, _, chunks, _, _ in inputs:
        rendered = tuple(map(render_context_chunk, chunks))
        for size in range(1, len(rendered) + 1):
            for subset in combinations(rendered, size):
                texts.add("".join(subset))
    counts, tokenizer_checks = native_counts(endpoint, texts, output)
    assert counts["<|im_start|>"] > 1 and counts["<|im_end|>"] > 1
    tokenizer_checks["special_syntax_counted_as_literal_text"] = True
    tokenizer = HostTokenizer(
        "qwen3-14b-native-literal", runtime_identity["model_digest"], counts.__getitem__
    )

    def broken_counter(text: str) -> int:
        raise RuntimeError("Injected tokenizer failure")

    broken = HostTokenizer(tokenizer.name, tokenizer.revision, broken_counter)
    measurements = []
    fallback_checks = 0
    for original, selection, chunks, ok, support_ms in inputs:
        expected = set(original["expected_fact_ids"])
        for budget in BUDGETS:
            byte_pack = pack_context(chunks, max_tokens=budget, top_k=5)
            failed_pack = pack_context(chunks, max_tokens=budget, top_k=5, tokenizer=broken)
            assert failed_pack.chunks == byte_pack.chunks and failed_pack.text == byte_pack.text
            assert failed_pack.token_count == byte_pack.token_count
            assert failed_pack.fallback_reason == ("tokenizer_error" if chunks else None)
            fallback_checks += 1
            for method, counter in (("native", tokenizer), ("bytes", None)):
                started = time.perf_counter()
                packed = pack_context(chunks, max_tokens=budget, top_k=5, tokenizer=counter)
                elapsed = (time.perf_counter() - started) * 1000
                selected_ids = [chunk.id for chunk in packed.chunks]
                selected = set(selected_ids)
                assert packed.text == "".join(map(render_context_chunk, packed.chunks))
                assert packed.token_count <= budget
                if method == "native" and chunks:
                    assert packed.token_count_method == "host_tokenizer"
                assert counts[packed.text] <= len(packed.text.encode("utf-8"))
                if budget == 1200:
                    assert selected_ids == [chunk.id for chunk in chunks]
                measurements.append(
                    {
                        "case_id": original["case_id"],
                        "query_id": original["query_id"],
                        "dataset_version": original["dataset_version"],
                        "strategy": original["strategy"],
                        "selection": selection,
                        "method": method,
                        "budget": budget,
                        "selected_ids": selected_ids,
                        "expected_ids": sorted(expected),
                        "context_text": packed.text,
                        "native_tokens": counts[packed.text],
                        "context_bytes": len(packed.text.encode("utf-8")),
                        "budget_count": packed.token_count,
                        "operational_success": ok,
                        "expected_count": len(expected),
                        "supported_count": len(selected & expected),
                        "unsupported_count": len(selected - expected),
                        "answerable_hit": ok and bool(selected & expected),
                        "clean_answerable_hit": ok and bool(expected) and selected == expected,
                        "correct_abstention": ok and not expected and not selected,
                        "packing_ms": elapsed,
                        "original_retrieval_ms": original["retrieval_ms"],
                        "original_support_ms": support_ms,
                        **packed.metadata(),
                    }
                )
    assert len(measurements) == 4224
    with (output / "measurements.jsonl").open("x", encoding="utf-8") as stream:
        for row in measurements:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    save(
        output / "counts.json",
        [{"text": text, "tokens": count} for text, count in sorted(counts.items())],
    )
    save(output / "summary.json", summarize(measurements))
    return {
        "status": "completed",
        "distinct_questions": 44,
        "frozen_inputs": 176,
        "selections": 528,
        "measurement_rows": len(measurements),
        "fallback_checks": fallback_checks,
        "tokenizer_checks": tokenizer_checks,
        "runtime_identity": runtime_identity,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                Path(__file__).resolve(),
                PROTOCOL,
                ROOT / "mnemos/context_budget.py",
                ROOT / "mnemos/retrieval.py",
                ROOT / "mnemos/types.py",
            )
        },
        "comparison_sha256": {
            name: sha(run / name)
            for name in (
                "manifest.json",
                "frozen-inputs.jsonl",
                "observations.jsonl",
                "verified-analysis.json",
            )
        },
        "artifact_sha256": {
            name: sha(output / name)
            for name in (
                "tokenizer-requests.jsonl",
                "counts.json",
                "measurements.jsonl",
                "summary.json",
            )
        },
        "limits": [
            "Reused development data",
            "No fresh inference or extraction",
            "Packing latency uses a precollected count table",
            "No native Codex or Claude tokenizer compatibility claim",
            "No source-span compression or pilot acceptance claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--tokenizer-url", required=True)
    parser.add_argument("--runtime-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    url = urlsplit(args.tokenizer_url)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.path
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        parser.error("Tokenizer URL must be an explicit HTTP IPv4 loopback port")
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = measure(
            args.run.resolve(), args.tokenizer_url, args.runtime_receipt.resolve(), args.output
        )
    except BaseException as exc:
        save(
            args.output / "manifest.json",
            {
                "status": "failed",
                "started_at": started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    result.update(started_at=started, finished_at=datetime.now(timezone.utc).isoformat())
    save(args.output / "manifest.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "measurement_rows", "fallback_checks", "tokenizer_checks")
            }
        )
    )


if __name__ == "__main__":
    main()
