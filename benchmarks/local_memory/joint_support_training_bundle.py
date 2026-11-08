"""Prepare training-only targets and matched-evidence baseline requests."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .joint_support_baseline import baseline_requests
from .joint_support_data import audit_groups
from .joint_support_loss import training_weights

CORPUS_BINDINGS = {
    "inputs.json": "3870f78c390217b3035d094ff567ab64963a6005f9080f5c792e1c9fd6edd8f9",
    "labels.json": "094ca3f92c4a0630320efe9deb3d7eba28b9b34012e6f6a7ef8f7075dcc1ed6f",
    "critical-registry.json": "6c88615c5b97893c4bb115043c029c1f7d77376a0281961488c1301702e3d282",
}
BASELINE_BINDINGS = {
    "inputs.json": "88035b398e23d12c3751331d884d9422a1222daa0b925977f6b7d35ecaf515fe",
    "tokenizer-preflight.json": "3e50d4036f020272d83341db16f70e66f3db2c3595b973589a860c43187ea2fa",
}


def build_bundle(examples, labels, instruction):
    audit = audit_groups(examples, labels)
    if audit["review_status"] != {"reviewed": len(examples)}:
        raise ValueError("Training bundle requires reviewed labels")
    weights = training_weights(examples, labels)
    training = []
    for example in examples:
        if example["split"] != "train":
            continue
        targets = {t["id"]: t for t in labels[example["id"]]["targets"]}
        training.append(
            dict(
                id=example["id"],
                targets=[
                    dict(
                        id=c["id"],
                        supported=targets[c["id"]]["supported"],
                        weight=weights[(example["id"], c["id"])],
                    )
                    for c in example["request"]["candidates"]
                ],
            )
        )
    return dict(
        status="prepared_not_execution_frozen",
        training=training,
        examples=[
            dict(id=e["id"], split=e["split"], request=deepcopy(e["request"])) for e in examples
        ],
        baseline=[
            dict(id=e["id"], requests=baseline_requests(e["request"], instruction))
            for e in examples
            if e["split"] == "development"
        ],
        recipe=dict(
            seed=2026110808,
            optimizer_updates=50,
            full_partition_accumulation=True,
            query_batch_size=1,
            max_tokens=4096,
            max_bytes=64000,
            depth=8,
            rank=8,
            alpha=16,
            dropout=0,
            adapter_lr=0.0001,
            head_lr=0.001,
            weight_decay=0.01,
            betas=[0.9, 0.999],
            eps=1e-8,
            clip_norm=1.0,
            final_checkpoint_only=True,
            threshold_logit=0.0,
            baseline_threshold=0.9,
            timeout_seconds=1800,
            dtype="float16",
            attention="sdpa",
            scaler=dict(init_scale=128, growth_factor=2, backoff_factor=0.5, growth_interval=2000),
            merge_max_absolute_logit_error=0.02,
            merge_exact_threshold_decisions=True,
        ),
    )


def prepare(corpus, baseline, destination):
    sources = {}
    values = {}
    for key, root, bindings in (
        ("corpus", corpus, CORPUS_BINDINGS),
        ("baseline", baseline, BASELINE_BINDINGS),
    ):
        values[key] = {}
        for name, expected in bindings.items():
            path = root / name
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Bound artifact changed: " + str(path))
            values[key][name] = json.loads(raw)
            sources[str(path.resolve())] = expected
    instructions = {
        request["instruction"]
        for row in values["baseline"]["inputs.json"]
        for request in row["requests"]
    }
    if len(instructions) != 1:
        raise ValueError("Baseline does not have one fixed instruction")
    bundle = build_bundle(
        values["corpus"]["inputs.json"], values["corpus"]["labels.json"], instructions.pop()
    )
    preflight = values["baseline"]["tokenizer-preflight.json"]
    bundle["baseline_frame"] = {key: preflight[key] for key in ("prefix", "suffix", "label_ids")}
    bundle["source_bindings"] = sources
    destination.mkdir(exist_ok=False)
    (destination / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    bundle = prepare(args.corpus, args.baseline, args.destination)
    print(
        json.dumps(
            dict(
                status=bundle["status"],
                training_queries=len(bundle["training"]),
                baseline_questions=len(bundle["baseline"]),
            )
        )
    )
