"""Reconstruct frozen decisions and grade saved outputs outside the model worker."""

import argparse
import hashlib
import json
from pathlib import Path

from .joint_support_baseline import select_joint
from .joint_support_grading import grade_selections
from .reranker_support import select_witnesses


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path, allowed):
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] not in allowed or row["id"] in result:
            raise ValueError("Duplicate or unknown output identity")
        result[row["id"]] = row
    return result


def audit(destination):
    freeze = read(destination / "freeze.json")
    for path, expected in freeze["files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed before grading: " + path)
    bundle = read(freeze["bundle"])
    corpus = next(
        Path(p).parent
        for p in bundle["source_bindings"]
        if p.replace("\\", "/").endswith("joint-support-corpus-v3/labels.json")
    )
    examples, labels = read(corpus / "inputs.json"), read(corpus / "labels.json")
    joint = {
        r["id"]: r["frame"] for r in read(Path(freeze["preflight"]).with_name("joint-frames.json"))
    }
    baseline_frames = {
        r["id"]: r["witnesses"]
        for r in read(Path(freeze["preflight"]).with_name("baseline-frames.json"))
    }
    allowed = set(joint) | {
        "capacity::" + name
        for name in (
            "empty",
            "one-short",
            "five-short",
            "five-token-bound-1-sources",
            "five-token-bound-8-sources",
            "five-byte-bound",
        )
    }
    unavailable = dict(status="unavailable", selected_ids=[])
    grades = {}
    for phase in ("initial-control", "final-unmerged", "final-merged"):
        observations = rows(destination / (phase + ".jsonl"), allowed)
        decoded = {}
        for example in examples:
            identity = example["id"]
            row = observations.get(identity, unavailable)
            if row["status"] != "assessed":
                decoded[identity] = unavailable
                continue
            if row["frame_sha256"] != joint[identity]["frame_sha256"]:
                raise ValueError("Saved prediction frame changed")
            selected = select_joint(example["request"], row["target_ids"], row["logits"])
            if selected != row["selected_ids"]:
                raise ValueError("Saved selection disagrees with frozen logit decoder")
            decoded[identity] = dict(status="assessed", selected_ids=selected)
        grades[phase] = {
            split: grade_selections(
                examples,
                labels,
                {e["id"]: decoded[e["id"]] for e in examples if e["split"] == split},
                split,
            )
            for split in ("train", "tuning", "development")
        }
    baseline_inputs = {r["id"]: r["requests"] for r in bundle["baseline"]}
    observations = rows(destination / "baseline.jsonl", set(baseline_inputs))
    decoded = {}
    for identity, requests in baseline_inputs.items():
        row = observations.get(identity, unavailable)
        if row["status"] != "assessed":
            decoded[identity] = unavailable
            continue
        decision = select_witnesses(requests, row["logits"])
        if decision != row["decision"] or decision["selected_ids"] != row["selected_ids"]:
            raise ValueError("Saved baseline decision differs from frozen decoder")
        if row["token_counts"] != [len(w["input_ids"]) for w in baseline_frames[identity]]:
            raise ValueError("Baseline token accounting changed")
        decoded[identity] = dict(status="assessed", selected_ids=decision["selected_ids"])
    grades["baseline"] = grade_selections(examples, labels, decoded, "development")
    execution = read(destination / "result.json")
    result = dict(
        execution=execution,
        grades=grades,
        fresh_acceptance=False,
        complete_retrieval_latency=False,
        original_goal_complete=False,
    )
    with (destination / "diagnostic-grades.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = audit(args.destination.resolve())
    print(
        json.dumps(
            dict(
                execution_status=result["execution"]["status"],
                development={
                    phase: (grade if phase == "baseline" else grade["development"])["metrics"]
                    for phase, grade in result["grades"].items()
                },
            )
        )
    )
