"""Bind saved role vectors and freshly embed the two missing lifecycle inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .embedding_prefix_development_v1 import MANIFEST_PATH, load_cases
from .embedding_prefix_live_v1 import RecordedAPI, RecordedProvider, utcnow
from .prepare_source_association_vectors import (
    DIGEST,
    MODEL,
    _committed_inputs,
    _digest,
    _show_signature,
    _validate_show,
    _vectors,
    sha256_bytes,
)

PAIRED_SHA256 = "598f043e266a3ee704853853e9b2b105d36266252d0d94929e48883d2eeb7ab6"
METADATA_SHA256 = "873befcfa1c0ed3c3b9ae10ac364a1dcfb9ea11fe0bcf2b8921e588b2c461d76"
PREFIXES = {
    "unprefixed": {"query_prefix": "", "document_prefix": "", "symmetric_prefix": ""},
    "nomic_task": {
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "symmetric_prefix": "",
    },
}


def _put(mapping: dict[str, list[float]], text: str, vector: list[float]) -> None:
    if text in mapping and mapping[text] != vector:
        raise ValueError("conflicting vectors for one role/text")
    mapping[text] = vector


def maps_from_paired(manifest: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    cases = manifest["graph_cases"]
    arms = paired.get("arms")
    if not isinstance(arms, list) or [arm.get("label") for arm in arms] != list(PREFIXES):
        raise ValueError("paired embedding arms changed")
    result: dict[str, Any] = {}
    for arm in arms:
        label = arm["label"]
        if arm.get("dimension") != 768 or arm.get("prefixes") != PREFIXES[label]:
            raise ValueError("paired role contract changed")
        rows = arm.get("cases", [])
        if [row.get("case_id") for row in rows] != [case["id"] for case in cases]:
            raise ValueError("paired cases changed")
        queries: dict[str, list[float]] = {}
        documents: dict[str, list[float]] = {}
        for case, row in zip(cases, rows):
            query = _vectors({"embeddings": [row["query_vector"]]}, 1)[0]
            _put(queries, case["query"], query)
            facts = [*case["facts"], *case.get("scope_controls", [])]
            raw_documents = row.get("document_vectors", {})
            if set(raw_documents) != {fact["id"] for fact in facts}:
                raise ValueError("paired document/control IDs changed")
            for fact in facts:
                document = _vectors({"embeddings": [raw_documents[fact["id"]]]}, 1)[0]
                _put(documents, fact["content"], document)
        result[label] = {"queries": queries, "documents": documents, "prefixes": PREFIXES[label]}
    return result


def corrected_contents(manifest: dict[str, Any]) -> list[str]:
    contents = [
        action["new_content"]
        for case in manifest["graph_cases"]
        for action in case.get("lifecycle_actions", [])
        if action["action"] == "correct"
    ]
    if len(contents) != 1 or not isinstance(contents[0], str) or not contents[0]:
        raise ValueError("authored lifecycle correction content changed")
    return contents


def prepare(output: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    prior = repo / "docs/validation/2026-09-20/embedding-prefix-development-v1"
    protocol = repo / "docs/plans/2026-09-20-source-association-role-protocol-v1.md"
    inputs = [
        MANIFEST_PATH,
        prior / "paired.json",
        prior / "artifact.json",
        protocol,
        Path(__file__),
        Path(__file__).with_name("embedding_prefix_live_v1.py"),
        Path(__file__).with_name("embedding_prefix_development_v1.py"),
        Path(__file__).with_name("prepare_source_association_vectors.py"),
    ]
    _, commit, hashes = _committed_inputs(repo, inputs)
    if sha256_bytes((prior / "paired.json").read_bytes()) != PAIRED_SHA256:
        raise ValueError("pinned paired vectors changed")
    if sha256_bytes((prior / "artifact.json").read_bytes()) != METADATA_SHA256:
        raise ValueError("pinned paired metadata changed")
    manifest, manifest_sha, _ = load_cases()
    paired = json.loads((prior / "paired.json").read_bytes())
    metadata = json.loads((prior / "artifact.json").read_bytes())
    if metadata["status"] != "complete" or metadata["digest"] != DIGEST:
        raise ValueError("prior embedding measurement not complete on pinned model")
    role_maps = maps_from_paired(manifest, paired)
    corrected = corrected_contents(manifest)
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("prepared vectors must be outside repository")
    output.mkdir(parents=True, exist_ok=False)
    api = RecordedAPI(output)
    report: dict[str, Any] = {
        "schema_version": "source-association-role-vectors-v1",
        "status": "running",
        "started_at": utcnow(),
        "source_commit": commit,
        "input_hashes": hashes,
        "manifest_sha256": manifest_sha,
        "paired_sha256": PAIRED_SHA256,
        "prior_metadata_sha256": METADATA_SHA256,
        "model": MODEL,
        "model_digest": DIGEST,
        "correction_contents": corrected,
        "role_vectors": role_maps,
        "new_embeddings_only": "authored corrected document, one per arm",
    }
    report_path = output / "artifact.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    providers: dict[str, RecordedProvider] = {}
    try:
        report["ollama_version"] = api.call("metadata", "/api/version")
        if report["ollama_version"] != metadata["ollama_version"]:
            raise ValueError("Ollama runtime changed since paired vectors")
        report["tags_before"] = api.call("metadata", "/api/tags")
        _digest(report["tags_before"])
        report["show_before"] = api.call("metadata", "/api/show", {"model": MODEL})
        _validate_show(report["show_before"])
        if _show_signature(report["show_before"]) != _show_signature(metadata["show_before"]):
            raise ValueError("model configuration changed since paired vectors")
        report["ps_before"] = api.call("metadata", "/api/ps")
        for label, prefixes in PREFIXES.items():
            providers[label] = RecordedProvider(api, label)
            vectors = providers[label].embed_batch(
                [prefixes["document_prefix"] + text for text in corrected]
            )
            for content, vector in zip(corrected, vectors):
                if content in role_maps[label]["documents"]:
                    raise ValueError("corrected content unexpectedly already has a vector")
                role_maps[label]["documents"][content] = vector
        report["tags_after"] = api.call("metadata", "/api/tags")
        _digest(report["tags_after"])
        report["show_after"] = api.call("metadata", "/api/show", {"model": MODEL})
        _validate_show(report["show_after"])
        if _show_signature(report["show_before"]) != _show_signature(report["show_after"]):
            raise ValueError("model configuration changed during correction embedding")
        report["ps_after"] = api.call("metadata", "/api/ps")
        _, final_commit, final_hashes = _committed_inputs(repo, inputs)
        if (final_commit, final_hashes) != (commit, hashes):
            raise ValueError("committed preparation inputs changed")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_at"] = utcnow()
        report["request_count"] = api.number
        report["loaded_profiles"] = {label: p.loaded_profiles for label, p in providers.items()}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"status": result["status"], "requests": result["request_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
