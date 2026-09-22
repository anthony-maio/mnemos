"""Phase 0 spike: judged source classification on the 46-case development fixture.

Runs ``classify_source_memories_with_judge`` through one or two backends
(TypeSafe Jev, local Ollama) over the same cases the prompt-driven classifier
was measured on, and reports mechanical agreement proxies, latency, tokens,
and cost next to that earlier run. It claims no semantic score: the rubrics
give expected meanings, and ``spike-review.json`` lays retained slices beside
them for a person to read.

Go/no-go for the judged pipeline: TypeSafe presence agreement at or above the
earlier local run, zero contract errors, and p95 latency per call under 2 s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemos.config import JudgmentConfig, MemorySafetyConfig
from mnemos.extraction.judged_classification import classify_source_memories_with_judge
from mnemos.judgment import Judge, JudgmentContractError
from mnemos.memory_safety import MemoryWriteFirewall
from mnemos.types import EpisodicEvent
from mnemos.utils.llm import OllamaProvider
from mnemos.utils.ollama_judge import OllamaJudge
from mnemos.utils.typesafe import USD_PER_MILLION_INPUT_TOKENS, TypeSafeJudge

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO / "docs/validation/2026-09-21/source-classification-terminal-v12"


def _load_fixture(fixture: Path) -> dict[str, Any]:
    inputs = json.loads((fixture / "inputs.json").read_text(encoding="utf-8"))
    durability = json.loads((fixture / "durability-rubric.json").read_text(encoding="utf-8"))
    history = json.loads((fixture / "history-rubric.json").read_text(encoding="utf-8"))
    baseline: dict[str, dict[str, Any]] = {}
    observations = fixture / "observations.jsonl"
    if observations.exists():
        for line in observations.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            baseline[row["case_id"]] = row
    return {
        "cases": inputs["cases"],
        "baseline_model": inputs.get("model"),
        "durability": {case["case_id"]: case for case in durability["cases"]},
        "history": {case["id"]: case for case in history["cases"]},
        "baseline": baseline,
    }


def expected_presence(case_id: str, fixture: dict[str, Any]) -> bool | None:
    """Whether the rubric expects something durable to be retained, if it says."""
    durable = fixture["durability"].get(case_id)
    if durable is not None:
        return not str(durable["expected_durable_meaning"]).strip().lower().startswith("none")
    history = fixture["history"].get(case_id)
    if history is not None:
        return bool(history["expected_decision"] == "facts")
    return None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def run_backend(
    name: str,
    judge: Judge,
    fixture: dict[str, Any],
    *,
    config: JudgmentConfig,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in fixture["cases"][:limit]:
        event = EpisodicEvent.model_validate(case["source"])
        row: dict[str, Any] = {"backend": name, "case_id": case["id"]}
        started = time.monotonic()
        try:
            result = await classify_source_memories_with_judge(
                [event], judge, config=config, include_affective=True
            )
        except JudgmentContractError as exc:
            row.update(status="contract_error", error=str(exc)[:500])
        except Exception as exc:  # provider outages, blocked content, render rejections
            row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
        else:
            row.update(
                status="ok",
                retained=[
                    {"kind": memory.kind, "content": memory.content.strip()}
                    for memory in result.memories
                ],
                decisions=[asdict(decision) for decision in result.decisions],
                calls=result.trace.calls,
                cognitive_state=(
                    result.cognitive_state.model_dump() if result.cognitive_state else None
                ),
            )
        row["wall_ms"] = (time.monotonic() - started) * 1000.0
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("backend", "case_id", "status")}), flush=True)
    return rows, summarize(name, rows, fixture)


def summarize(name: str, rows: list[dict[str, Any]], fixture: dict[str, Any]) -> dict[str, Any]:
    calls = [call for row in rows for call in row.get("calls", [])]
    latencies = [float(call["latency_ms"]) for call in calls]
    input_tokens = sum(int(call["input_tokens"]) for call in calls)
    presence_hits = presence_total = 0
    twin_hits = twin_total = 0
    twins: dict[str, dict[str, bool]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        retained = bool(row["retained"])
        expected = expected_presence(row["case_id"], fixture)
        if expected is not None:
            presence_total += 1
            presence_hits += int(retained == expected)
        durable = fixture["durability"].get(row["case_id"])
        if durable is not None:
            twins.setdefault(durable["pair_id"], {})[row["case_id"]] = retained
    for pair_id, members in twins.items():
        one_turn = [v for k, v in members.items() if k.endswith("-one-turn")]
        policy = [v for k, v in members.items() if k.endswith("-policy")]
        if one_turn and policy:
            twin_total += 1
            twin_hits += int(not one_turn[0] and policy[0])
    return {
        "backend": name,
        "cases": len(rows),
        "ok": sum(row["status"] == "ok" for row in rows),
        "contract_errors": sum(row["status"] == "contract_error" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "presence_agreement": (presence_hits / presence_total) if presence_total else None,
        "presence_total": presence_total,
        "twin_pair_agreement": (twin_hits / twin_total) if twin_total else None,
        "twin_pairs": twin_total,
        "calls": len(calls),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_max": max(latencies) if latencies else None,
        "input_tokens": input_tokens,
        "cost_usd": (
            input_tokens * USD_PER_MILLION_INPUT_TOKENS / 1e6 if name == "typesafe" else None
        ),
        "models": sorted({str(call["model"]) for call in calls}),
    }


def baseline_summary(fixture: dict[str, Any]) -> dict[str, Any]:
    hits = total = 0
    for case_id, row in fixture["baseline"].items():
        if row.get("status") != "runtime_valid":
            continue
        expected = expected_presence(case_id, fixture)
        if expected is None:
            continue
        total += 1
        hits += int(bool(row.get("retained")) == expected)
    return {
        "backend": f"baseline:{fixture['baseline_model']}",
        "ok": sum(row.get("status") == "runtime_valid" for row in fixture["baseline"].values()),
        "cases": len(fixture["baseline"]),
        "presence_agreement": (hits / total) if total else None,
        "presence_total": total,
    }


def review_rows(rows: list[dict[str, Any]], fixture: dict[str, Any]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_case.setdefault(row["case_id"], {"case_id": row["case_id"]})
        entry[row["backend"]] = row.get("retained", row.get("error"))
    for case_id, entry in by_case.items():
        durable = fixture["durability"].get(case_id)
        history = fixture["history"].get(case_id)
        rubric = durable or history or {}
        entry["expected"] = rubric.get("expected_durable_meaning") or rubric.get("required_meaning")
        entry["forbidden"] = rubric.get("forbidden_meaning")
        baseline = fixture["baseline"].get(case_id, {})
        entry["baseline"] = [
            {"kind": item.get("kind"), "content": str(item.get("content", "")).strip()}
            for item in baseline.get("retained", [])
        ]
    return list(by_case.values())


def build_judge(name: str, args: argparse.Namespace) -> Judge:
    firewall = MemoryWriteFirewall(MemorySafetyConfig())
    if name == "typesafe":
        key = os.environ.get("MNEMOS_TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("set MNEMOS_TYPESAFE_API_KEY to run the typesafe backend")
        return TypeSafeJudge(key, firewall=firewall, model=args.typesafe_model)
    provider = OllamaProvider(
        model=args.ollama_model,
        base_url=args.endpoint,
        timeout=args.timeout_seconds,
        temperature=0.0,
        max_tokens=args.max_tokens,
        think=False,
    )
    return OllamaJudge(provider, firewall=firewall)


async def main_async(args: argparse.Namespace) -> int:
    fixture = _load_fixture(args.fixture_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    backends = ["typesafe", "ollama"] if args.judge == "both" else [args.judge]
    config = JudgmentConfig()
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = [baseline_summary(fixture)]
    started_at = datetime.now(timezone.utc).isoformat()
    for name in backends:
        judge = build_judge(name, args)
        rows, summary = await run_backend(name, judge, fixture, config=config, limit=args.limit)
        all_rows.extend(rows)
        summaries.append(summary)
        closer = getattr(judge, "aclose", None)
        if closer is not None:
            await closer()
    (output / "observations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True, default=str) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    (output / "spike-review.json").write_text(
        json.dumps(review_rows(all_rows, fixture), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "fixture": str(args.fixture_dir),
        "judgment_config": config.model_dump(),
        "typesafe_price_usd_per_million_input_tokens": USD_PER_MILLION_INPUT_TOKENS,
        "backends": summaries,
        "note": (
            "presence_agreement and twin_pair_agreement are mechanical proxies for whether "
            "anything durable was retained; they say nothing about whether the right meaning "
            "was retained. Read spike-review.json for that."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=("typesafe", "ollama", "both"), default="typesafe")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO
        / "docs/validation"
        / datetime.now().strftime("%Y-%m-%d")
        / "typesafe-spike-v1",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--typesafe-model", default="jev-latest")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
