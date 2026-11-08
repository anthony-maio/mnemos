"""Single-pass runner for the frozen discriminative feasibility protocol."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

# Run this file directly: importing the benchmark package also imports unrelated
# service adapters whose dependencies are absent from the isolated model runtime.
from discriminative_support import select_supported


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    freeze = json.loads((root / "execution-freeze.json").read_text())
    for filename, expected in freeze["files"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"artifact changed: {filename}")
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if versions != freeze["versions"]:
        raise ValueError("runtime dependency versions changed")
    os.environ["HF_HOME"] = str(root / "hf-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    rows = json.loads((root / "inputs.json").read_text())
    if len(rows) != 68 or [r["key"] for r in rows] != sorted(r["key"] for r in rows):
        raise ValueError("invalid schedule")
    # Exclusive marker prevents accidental restart after a partial measured run.
    with (root / "execution-start.json").open("x") as handle:
        json.dump(
            {"pid": os.getpid(), "freeze_sha256": digest(root / "execution-freeze.json")}, handle
        )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        root / "model", local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            root / "model",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        .to("cpu")
        .eval()
    )
    if model.config.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        raise ValueError("unexpected classifier labels")
    load_ms = (time.perf_counter() - started) * 1000

    def predict(pairs):
        if len(pairs) > 128:
            raise ValueError("pair limit exceeded")
        outputs, counts = [], []
        for offset in range(0, len(pairs), 32):
            batch = pairs[offset : offset + 32]
            encoded = tokenizer(
                [p["premise"] for p in batch],
                [p["hypothesis"] for p in batch],
                truncation=False,
                padding=True,
                return_tensors="pt",
            )
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            if not all(0 < length <= 512 for length in lengths):
                raise ValueError("token limit exceeded")
            with torch.inference_mode():
                logits = model(**encoded).logits
            if list(logits.shape) != [len(batch), 3] or not torch.isfinite(logits).all():
                raise ValueError("invalid model output")
            outputs.extend(logits.tolist())
            counts.extend(lengths)
        return outputs, counts

    started = time.perf_counter()
    warmup = predict(
        [{"premise": "A blue box is on the shelf.", "hypothesis": "There is a box on the shelf."}]
    )
    (root / "warmup.json").write_text(
        json.dumps(
            {
                "load_ms": load_ms,
                "warmup_ms": (time.perf_counter() - started) * 1000,
                "logits": warmup[0],
                "token_counts": warmup[1],
            }
        )
    )
    with (
        (root / "observation-starts.jsonl").open("x") as starts,
        (root / "observations.jsonl").open("x") as output,
    ):
        for row in rows:
            starts.write(json.dumps({"key": row["key"]}) + "\n")
            starts.flush()
            started = time.perf_counter()
            record = {"key": row["key"], "status": "assessed"}
            try:
                logits, counts = predict(row["pairs"])
                record.update(
                    logits=logits,
                    token_counts=counts,
                    decision=select_supported(row["pairs"], logits),
                )
            except Exception as exc:
                record.update(status="unavailable", error_type=type(exc).__name__, error=str(exc))
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            output.write(json.dumps(record) + "\n")
            output.flush()
    for filename, expected in freeze["files"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"artifact changed during execution: {filename}")
    (root / "execution-complete.json").write_text(
        json.dumps(
            {"observations": len(rows), "owned_pid": os.getpid(), "artifacts_unchanged": True}
        )
    )
    print("Completed 68 observations; artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
