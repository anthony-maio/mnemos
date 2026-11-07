"""Isolated local QA worker for the fixed support-composition experiment.

Optional model dependencies are imported only by main. Configuration and output
paths belong to the private experiment; this worker is not a product entrypoint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


def tokenize_input(tokenizer: Any, query: str, context: str) -> tuple[Any, dict[str, Any]]:
    if (
        not isinstance(query, str)
        or not query.strip()
        or not isinstance(context, str)
        or not context
    ):
        raise ValueError("QA requires nonempty question and context strings")
    question_tokens = len(tokenizer(query, add_special_tokens=False)["input_ids"])
    if not 0 < question_tokens <= 64:
        raise ValueError("Question exceeds the frozen token limit")
    encoded = tokenizer(
        query, context, truncation=False, return_offsets_mapping=True, return_tensors="pt"
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    ids = encoded["input_ids"][0].tolist()
    masks = encoded["attention_mask"][0].tolist()
    if not 0 < len(ids) <= 512:
        raise ValueError("Input exceeds the frozen token limit; truncation is disabled")
    cls = [
        i
        for i, value in enumerate(ids)
        if value == tokenizer.cls_token_id and sequence_ids[i] is None
    ]
    if len(cls) != 1:
        raise ValueError("Expected exactly one CLS token")
    return encoded, {
        "input_ids": ids,
        "offsets": offsets,
        "sequence_ids": sequence_ids,
        "attention_mask": masks,
        "cls_index": cls[0],
        "question_token_count": question_tokens,
    }


def main() -> None:
    config_path = Path(os.environ["MNEMOS_QA_WORKER_CONFIG"]).resolve()
    config = json.loads(config_path.read_bytes())
    out = Path(config["output"]).resolve()
    out.mkdir(exist_ok=False)
    model_root = Path(config["model_root"]).resolve()
    tokenize_only = config["mode"] == "tokenize_only"
    if config["mode"] not in {"tokenize_only", "inference"}:
        raise ValueError("Unsupported worker mode")
    versions = {name: importlib.metadata.version(name) for name in config["versions"]}
    if versions != config["versions"]:
        raise ValueError("QA dependency versions differ from the frozen configuration")
    for name, expected in config["model_files"].items():
        with (model_root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("QA model asset differs: " + name)
    control: dict[str, Any] = {
        "status": "loading",
        "pid": os.getpid(),
        "mode": config["mode"],
        "versions": versions,
        "inference_calls": 0,
        "requests": 0,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }

    def save() -> None:
        (out / "control.json").write_text(json.dumps(control, indent=2) + "\n", encoding="utf-8")

    def append(name: str, value: dict[str, Any]) -> None:
        with (out / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")

    save()
    try:
        load_started = time.perf_counter()
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(2026110702)
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_root), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if not tokenizer.is_fast:
            raise ValueError("QA requires the frozen fast tokenizer")
        model = None
        if not tokenize_only:
            model = (
                AutoModelForQuestionAnswering.from_pretrained(
                    str(model_root),
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                )
                .to(device="cpu", dtype=torch.float32)
                .eval()
            )
            if model.__class__.__name__ != "DebertaV2ForQuestionAnswering":
                raise ValueError("Unexpected QA architecture")
        control.update(
            status="ready",
            load_ms=(time.perf_counter() - load_started) * 1000,
            tokenizer_class=type(tokenizer).__name__,
            model_class=type(model).__name__ if model else None,
        )
        save()
        print(
            json.dumps(
                {
                    "kind": "ready",
                    "pid": os.getpid(),
                    "mode": config["mode"],
                    "load_ms": control["load_ms"],
                }
            ),
            flush=True,
        )
        seen = set()
        while True:
            wire = sys.stdin.buffer.readline(131073)
            if not wire:
                break
            if len(wire) > 131072 or not wire.endswith(b"\n"):
                raise ValueError("Worker input exceeds its frame limit")
            request = json.loads(wire)
            if request == {"kind": "stop"}:
                break
            identifier = request.get("id")
            if (
                request.get("kind") != "predict"
                or not isinstance(identifier, str)
                or not identifier
                or identifier in seen
            ):
                raise ValueError("Worker request has an invalid or repeated identity")
            seen.add(identifier)
            append("requests.jsonl", request)
            control["requests"] += 1
            started = time.perf_counter()
            response: dict[str, Any] = {
                "kind": "result",
                "id": identifier,
                "request_sha256": hashlib.sha256(wire).hexdigest(),
                "status": "unavailable",
            }
            try:
                encoded, raw = tokenize_input(
                    tokenizer, request.get("query"), request.get("context")
                )
                response["output"] = raw
                if model is not None:
                    forward_started = time.perf_counter()
                    control["inference_calls"] += 1
                    with torch.inference_mode():
                        output = model(**encoded, return_dict=True)
                    response["forward_ms"] = (time.perf_counter() - forward_started) * 1000
                    raw["start_logits"] = output.start_logits[0].tolist()
                    raw["end_logits"] = output.end_logits[0].tolist()
                response["status"] = "tokenized" if tokenize_only else "assessed"
            except Exception as exc:
                response["error_type"] = type(exc).__name__
            response["elapsed_ms"] = (time.perf_counter() - started) * 1000
            append("outputs.jsonl", response)
            save()
            print(json.dumps(response), flush=True)
        control["status"] = "stopped"
    except BaseException as exc:
        control.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
