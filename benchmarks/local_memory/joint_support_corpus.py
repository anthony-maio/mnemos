"""Assemble the reviewed draft without changing its historical artifacts."""

import hashlib
import json
from pathlib import Path

from .joint_support_adversarial import build_adversarial
from .joint_support_coverage import build_coverage
from .joint_support_critical import apply_critical_registry
from .joint_support_data import audit_groups
from .joint_support_expansion import build_expansion
from .joint_support_partitions import build_partitions
from .joint_support_seed import build_seed


def assemble():
    examples, labels = build_seed()
    for build in (build_expansion, build_partitions, build_coverage, build_adversarial):
        added, annotations = build()
        examples.extend(added)
        labels.update(annotations)
    return examples, labels


def write_corpus(destination: Path):
    examples, labels = assemble()
    registry_path = Path(__file__).with_name("joint_support_critical_registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    labels, critical = apply_critical_registry(examples, labels, registry)
    audit = audit_groups(examples, labels)
    destination.mkdir(exist_ok=False)
    for name, value in (
        ("inputs.json", examples),
        ("labels.json", labels),
        ("structural-audit.json", audit),
        ("critical-registry.json", registry),
        ("critical-inventory.json", critical),
    ):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest = dict(
        status=(
            "reviewed_draft_not_training_ready"
            if audit["review_status"] == {"reviewed": len(examples)}
            else "unreviewed_draft_not_training_ready"
        ),
        examples=len(examples),
        targets=sum(len(v["targets"]) for v in labels.values()),
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.iterdir()
            if p.is_file()
        },
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
