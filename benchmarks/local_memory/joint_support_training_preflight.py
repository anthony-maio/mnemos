"""Tokenize the prepared bundle without loading model weights or grading labels."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path


def prepare(bundle_path, destination):
    spec = importlib.util.spec_from_file_location(
        "capacity", Path(__file__).with_name("joint_support_capacity.py")
    )
    capacity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capacity)
    bundle = json.loads(bundle_path.read_bytes())
    for path, expected in bundle["source_bindings"].items():
        if capacity.digest(path) != expected:
            raise ValueError("Original source binding changed")
    tokenizer, encode, files, binding = capacity.tokenizer_setup()
    joint = [dict(id=e["id"], frame=asdict(encode(e["request"]))) for e in bundle["examples"]]
    prefix = tokenizer.encode(bundle["baseline_frame"]["prefix"], add_special_tokens=False)
    suffix = tokenizer.encode(bundle["baseline_frame"]["suffix"], add_special_tokens=False)
    for name in ("no", "yes"):
        expected = bundle["baseline_frame"]["label_ids"][name]
        if tokenizer.encode(name, add_special_tokens=False) != [expected]:
            raise ValueError("Baseline decision token changed")
    baseline = []
    for question in bundle["baseline"]:
        rows = []
        if len(question["requests"]) > 32:
            raise ValueError("Witness count overflow")
        for request in question["requests"]:
            body = (
                "<Instruct>: "
                + request["instruction"]
                + "\n<Query>: "
                + request["query"]
                + "\n<Document>: "
                + request["document"]
            )
            ids = (
                prefix + tokenizer.encode(body, add_special_tokens=False, truncation=False) + suffix
            )
            if not 0 < len(ids) <= 4096:
                raise ValueError("Full baseline frame overflow; no truncation permitted")
            rows.append(
                dict(fact_id=request["fact_id"], evidence_id=request["evidence_id"], input_ids=ids)
            )
        baseline.append(dict(id=question["id"], witnesses=rows))
    files.update(bundle["source_bindings"])
    for path in (
        bundle_path,
        Path(__file__),
        Path(__file__).with_name("joint_support_capacity.py"),
        Path(__file__).with_name("joint_support_frame.py"),
        Path(__file__).with_name("joint_support_baseline.py"),
        Path(__file__).with_name("reranker_support.py"),
    ):
        files[str(path.resolve())] = capacity.digest(path)
    destination.mkdir(exist_ok=False)
    capacity.write(destination / "joint-frames.json", joint)
    capacity.write(destination / "baseline-frames.json", baseline)
    for name in ("joint-frames.json", "baseline-frames.json"):
        files[str((destination / name).resolve())] = capacity.digest(destination / name)
    lengths = [len(w["input_ids"]) for q in baseline for w in q["witnesses"]]
    result = dict(
        status="tokenization_preflight_only_not_execution_frozen",
        files=files,
        tokenizer_binding=binding,
        joint_examples=len(joint),
        baseline_questions=len(baseline),
        baseline_witnesses=len(lengths),
        baseline_min_tokens=min(lengths),
        baseline_max_tokens=max(lengths),
        joint_max_tokens=max(len(e["frame"]["input_ids"]) for e in joint),
    )
    for path, expected in files.items():
        if capacity.digest(path) != expected:
            raise ValueError("Binding changed during tokenization")
    capacity.write(destination / "preflight.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = prepare(args.bundle.resolve(), args.destination.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))
