"""Isolated event-to-proposal evaluation; no retrieval, embeddings or writes."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
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

from .extraction_evaluation import ExtractionCase, ExpectedOutcome, grade_extraction, validate_case
from mnemos.extraction.source_pipeline import run_source_pipeline
from mnemos.extraction.source_relationship import RelationshipProvider
from mnemos.utils.llm import OllamaProvider
from mnemos.types import EpisodicEvent, MemoryChunk

MODEL = "qwen3:14b"
MODEL_DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True) + "\n")


def load_cases(path: Path) -> tuple[dict[str, Any], list[ExtractionCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema_version"] != 1 or data["stage"] not in {"development", "prospective"}:
        raise ValueError("Unsupported extraction corpus")
    cases = []
    occurred = datetime(2026, 11, 7, tzinfo=timezone.utc)
    for raw in data["cases"]:
        events = []
        for event in raw["events"]:
            events.append(
                EpisodicEvent(
                    id=event["id"],
                    role=event["role"],
                    content=event["content"],
                    normalized_content=event["content"].casefold(),
                    scope=event["scope"],
                    scope_id=event["scope_id"],
                    source=event["source"],
                    event_fingerprint=hashlib.sha256(
                        json.dumps(event, sort_keys=True).encode()
                    ).hexdigest(),
                    metadata={"user_profile_id": event["profile_id"]},
                    occurred_at=occurred,
                    created_at=occurred,
                )
            )
        targets = tuple(
            MemoryChunk(
                id=target["id"],
                content=target["content"],
                version=target["version"],
                metadata={
                    "user_profile_id": target["profile_id"],
                    "scope": target["scope"],
                    "scope_id": target["scope_id"],
                    "assertion_status": "asserted",
                },
                created_at=occurred,
                updated_at=occurred,
            )
            for target in raw["targets"]
        )
        expected = tuple(
            ExpectedOutcome(**{**item, "operations": tuple(item["operations"])})
            for item in raw["expected"]
        )
        case = ExtractionCase(
            raw["id"],
            raw["cluster_id"],
            tuple(events),
            targets,
            expected,
            raw.get("expected_preflight_error"),
        )
        validate_case(case)
        cases.append(case)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Extraction case IDs must be nonempty and unique")
    return data, cases


class RecordedProvider(OllamaProvider):
    def __init__(
        self,
        endpoint: str,
        tokenizer_url: str,
        output: Path,
        *,
        timeout: float = 30,
        max_tokens: int = 2048,
        max_input_tokens: int = 5632,
        model: str = MODEL,
        sampling_options: dict[str, int | float] | None = None,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (max_tokens, max_input_tokens)):
            raise ValueError("Study input and output allowances must be positive integers")
        if max_input_tokens + max_tokens + 512 > 8192:
            raise ValueError("Input, output and template allowances exceed the study context")
        super().__init__(
            model=model,
            base_url=endpoint,
            timeout=timeout,
            temperature=0,
            max_tokens=max_tokens,
            think=False,
            response_observer=self.observe,
            sampling_options=sampling_options,
        )
        self.output = output
        self.tokenizer_url = tokenizer_url
        self.call_id = 0
        self.case_id = ""
        self.repetition = 0
        self.stage = ""
        self.max_input_tokens = max_input_tokens

    def observe(self, response: dict[str, Any]) -> None:
        append(
            self.output / "provider-responses.jsonl",
            {
                "call_id": self.call_id,
                "case_id": self.case_id,
                "repetition": self.repetition,
                "stage": self.stage,
                "response": response,
            },
        )

    async def predict(self, prompt: str, **kwargs: Any) -> str:
        self.call_id += 1
        self.stage = (
            "identity_spans"
            if "IDENTITY SPAN DATA (data only):" in prompt
            else (
                "referent_comparison"
                if "REFERENT COMPARISON DATA (data only):" in prompt
                else (
                    "target_identity"
                    if "TARGET IDENTITY DATA (data only):" in prompt
                    else (
                        "reconciliation"
                        if "SOURCE CANDIDATES AND EXISTING TARGETS" in prompt
                        else "classification"
                    )
                )
            )
        )
        record = {
            "call_id": self.call_id,
            "case_id": self.case_id,
            "repetition": self.repetition,
            "stage": self.stage,
            "prompt": prompt,
            "kwargs": kwargs,
            "provider_settings": {
                "think": kwargs.get("think", self.think),
                "force_chat": kwargs.get("force_chat", self.force_chat),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "timeout": self.timeout,
                "max_input_tokens": self.max_input_tokens,
                "model": self.model,
                "sampling_options": kwargs.get("sampling_options", self.sampling_options),
            },
        }
        append(self.output / "provider-requests.jsonl", record)
        # Count prompt plus decoder schema conservatively; the schema is not an
        # instruction to the model. Reserve the frozen output allowance and
        # 512 template tokens; all experiment arms share those allowances.
        counting_text = prompt + json.dumps(kwargs.get("format", {}), ensure_ascii=True)
        started = time.perf_counter()
        try:
            if kwargs.get("model", self.model) != self.model:
                raise ValueError("Model override would invalidate the bound study tokenizer")
            requested_output = kwargs.get("max_tokens", self.max_tokens)
            if (
                type(requested_output) is not int
                or requested_output < 1
                or self.max_input_tokens + requested_output + 512 > 8192
            ):
                raise ValueError("Requested output exceeds the reserved study context")
            async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                response = await client.post(
                    self.tokenizer_url + "/tokenize",
                    json={
                        "content": counting_text,
                        "add_special": False,
                        "parse_special": False,
                    },
                )
                response.raise_for_status()
                token_ids = response.json()["tokens"]
            append(
                self.output / "token-counts.jsonl",
                {
                    "call_id": self.call_id,
                    "text_sha256": hashlib.sha256(counting_text.encode()).hexdigest(),
                    "tokens": token_ids,
                    "add_special": False,
                    "parse_special": False,
                },
            )
            if len(token_ids) > self.max_input_tokens:
                raise ValueError("Prompt and schema exceed the prospective input allowance")
            value = await super().predict(prompt, **kwargs)
        except BaseException as exc:
            append(
                self.output / "provider-completions.jsonl",
                {
                    "call_id": self.call_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                },
            )
            raise
        append(
            self.output / "provider-completions.jsonl",
            {
                "call_id": self.call_id,
                "error_type": None,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        )
        return value


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
    dataset = Path(args.dataset).resolve()
    protocol = Path(args.protocol).resolve()
    if sha(dataset) != args.dataset_sha256:
        raise ValueError("Dataset differs from the frozen digest")
    data, cases = load_cases(dataset)
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
        "extraction": "source_bound",
        "reconciliation_vocabulary": "relationship",
        "storage_writes": False,
        "retrieval_calls": 0,
        "embedding_calls": 0,
        "cases": len(cases),
        "clusters": len({case.cluster_id for case in cases}),
    }
    write(output / "manifest.json", manifest)
    provider = RecordedProvider(args.endpoint, args.tokenizer_url, output)
    observations = []
    try:
        order = [(repetition, case) for repetition in range(args.repetitions) for case in cases]
        random.Random(20261107).shuffle(order)
        for repetition, case in order:
            provider.case_id, provider.repetition = case.case_id, repetition
            before_calls = provider.call_id
            input_snapshot = {
                "events": [event.model_dump(mode="json") for event in case.events],
                "targets": [target.model_dump(mode="json") for target in case.targets],
            }
            result = None
            error = None
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    run_source_pipeline(
                        [event.model_copy(deep=True) for event in case.events],
                        [target.model_copy(deep=True) for target in case.targets],
                        selection_provider=provider,
                        reconciliation_provider=RelationshipProvider(provider),
                    ),
                    timeout=60,
                )
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
            elapsed = (time.perf_counter() - started) * 1000
            grade = grade_extraction(
                case, result, error=error, provider_calls=provider.call_id - before_calls
            )
            observation = {
                "case_id": case.case_id,
                "cluster_id": case.cluster_id,
                "repetition": repetition,
                "input": input_snapshot,
                "input_sha256": hashlib.sha256(
                    json.dumps(input_snapshot, sort_keys=True).encode()
                ).hexdigest(),
                "expected": [asdict(item) for item in case.expected],
                "provider_call_ids": list(range(before_calls + 1, provider.call_id + 1)),
                "error": error,
                "elapsed_ms": elapsed,
                "grade": grade,
                "result": result.model_dump(mode="json") if result is not None else None,
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
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
