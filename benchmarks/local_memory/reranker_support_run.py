"""One-pass execution of the frozen, isolated reranker support screen."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from reranker_support import select_witnesses


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root.resolve()
    read = lambda path: json.loads(path.read_bytes())
    freeze = read(root / "execution-freeze.json")
    for name, expected in freeze["files"].items():
        assert digest(Path(name)) == expected, name
    assert freeze["versions"] == {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == freeze["device"]
    torch.manual_seed(2026110803)
    torch.cuda.manual_seed_all(2026110803)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = read(root / "inputs.json")
    assert len(rows) == 68 and [r["key"] for r in rows] == sorted(r["key"] for r in rows)
    preflight = read(root / "tokenizer-preflight.json")
    frames = {
        (r["key"], r["fact_id"], r["evidence_id"]): r["input_ids"]
        for r in read(root / "tokenized-frames.json")
    }
    with (root / "execution-start.json").open("x") as handle:
        json.dump({"pid": os.getpid(), "freeze_sha256": digest(root / "execution-freeze.json")}, handle)
    print(json.dumps({"owned_pid": os.getpid(), "status": "loading"}), flush=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        root / "model", local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        root / "model", local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - started) * 1000
    prefix = tokenizer.encode(preflight["prefix"], add_special_tokens=False)
    suffix = tokenizer.encode(preflight["suffix"], add_special_tokens=False)
    label_ids = [preflight["label_ids"][label] for label in ["no", "yes"]]
    assert label_ids == [tokenizer.convert_tokens_to_ids(label) for label in ["no", "yes"]]

    def predict(row, check_frames=True):
        requests = row["requests"]
        assert len(requests) <= 32
        ids = []
        for request in requests:
            body = ("<Instruct>: " + request["instruction"] + "\n<Query>: "
                    + request["query"] + "\n<Document>: " + request["document"])
            tokens = prefix + tokenizer.encode(body, add_special_tokens=False, truncation=False) + suffix
            assert 0 < len(tokens) <= 4096
            if check_frames:
                assert tokens == frames[(row["key"], request["fact_id"], request["evidence_id"])]
            ids.append(tokens)
        logits = []
        if ids:
            inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                output = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, label_ids]
                assert list(output.shape) == [len(ids), 2] and torch.isfinite(output).all()
                logits = output.float().cpu().tolist()
        decision = select_witnesses(requests, logits)
        return {"logits": logits, "token_counts": [len(x) for x in ids], "decision": decision}

    warmup = {"key": "warmup", "requests": [{"fact_id": "warmup", "evidence_id": "warmup",
        "instruction": "Determine whether the document answers the query.",
        "query": "What color is the box?", "document": "The box is blue."}]}
    torch.cuda.synchronize()
    started = time.perf_counter()
    warm = predict(warmup, False)
    torch.cuda.synchronize()
    warm.update(load_ms=load_ms, elapsed_ms=(time.perf_counter() - started) * 1000)
    (root / "warmup.json").write_text(json.dumps(warm) + "\n")
    for index, row in enumerate(rows):
        with (root / "observation-starts.jsonl").open("a") as handle:
            handle.write(json.dumps({"key": row["key"]}) + "\n")
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            result = predict(row)
            torch.cuda.synchronize()
            result.update(key=row["key"], status="assessed", elapsed_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:
            with (root / "observations.jsonl").open("a") as handle:
                handle.write(json.dumps({"key": row["key"], "status": "unavailable", "error_type": type(exc).__name__}) + "\n")
            raise
        with (root / "observations.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        if (index + 1) % 10 == 0:
            print(json.dumps({"completed": index + 1}), flush=True)
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for name, expected in freeze["files"].items():
        assert digest(Path(name)) == expected, name
    (root / "execution-complete.json").write_text(json.dumps({"observations": 68, "owned_pid": os.getpid(), "artifacts_unchanged": True}) + "\n")


if __name__ == "__main__":
    main()
