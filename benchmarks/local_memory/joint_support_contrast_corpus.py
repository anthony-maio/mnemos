"""Append reviewed development contrasts to immutable corpus-v2 inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from .joint_support_contrasts import build_contrasts
from .joint_support_critical import apply_critical_registry, fact_binding
from .joint_support_data import audit_groups

BASE_BINDINGS = {
    "inputs.json": "4a0cf6c88da7346533bbe8f38dd0a73156045b78dd3440c87fedeb8247bbe6f2",
    "labels.json": "404523ff3a5a33d2b21484dcd724b865f7c837ac8c42ce753cecd16cc7118ef7",
    "critical-registry.json": "ba5f238c973c469b29eae8945c61830929bc7e1b10a52120e07b6486aadb59d3",
}


def write_contrast_corpus(baseline: Path, destination: Path):
    saved = {}
    for name, expected in BASE_BINDINGS.items():
        raw = (baseline / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Original corpus binding changed: " + name)
        saved[name] = json.loads(raw)
    added, annotations = build_contrasts()
    if set(annotations) & set(saved["labels.json"]):
        raise ValueError("New labels would overwrite original examples")
    examples = saved["inputs.json"] + added
    labels = {**saved["labels.json"], **annotations}
    registry = saved["critical-registry.json"]
    rule = next(
        c for e in added for c in e["request"]["candidates"] if c["id"] == "vault-quorum-veto-rule"
    )
    registry["entries"].append(
        dict(
            target_id=rule["id"],
            fact_sha256=fact_binding(rule),
            justification="Preserves the two-of-three quorum and overriding safety veto as the complete opening authorization boundary.",
        )
    )
    labels, critical = apply_critical_registry(examples, labels, registry)
    # Registry expansion may only mark new examples. Old labels remain exact.
    if any(labels[k] != v for k, v in saved["labels.json"].items()):
        raise ValueError("Original labels changed")
    audit = audit_groups(examples, labels)
    destination.mkdir(exist_ok=False)
    for name, value in (
        ("inputs.json", examples),
        ("labels.json", labels),
        ("critical-registry.json", registry),
        ("critical-inventory.json", critical),
        ("structural-audit.json", audit),
    ):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest = dict(
        status=(
            "reviewed_draft_not_training_ready"
            if audit["review_status"] == {"reviewed": len(examples)}
            else "unreviewed_draft_not_training_ready"
        ),
        baseline_bindings=BASE_BINDINGS,
        examples=len(examples),
        targets=sum(len(v["targets"]) for v in labels.values()),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()},
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_contrast_corpus(args.baseline, args.destination)))
