"""Offline, paired Nomic task-prefix diagnostic for the five authored cases.

This module prepares and scores development vectors only.  It does not create a
Mnemos store, run graph traversal, call Ollama, or make an acceptance claim.
Callers supply an ``EmbeddingProvider`` so live transport remains outside this
bounded harness.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mnemos.scopes import ScopeContext, scope_allows
from mnemos.utils.embeddings import EmbeddingProvider, PrefixEmbeddingProvider, cosine_similarity

MANIFEST_PATH = Path(__file__).with_name("datasets") / "source-association-development-v1.json"
SCHEMA_VERSION = "embedding-prefix-development-v1"
PROTOCOL_VERSION = "paired-role-cosine-ranking-v1"
CASE_IDS = (
    "shared_capture_triplet",
    "shared_extraction_batch",
    "semantic_only_chain",
    "scope_isolation_pair",
    "revision_forget_replay",
)
EXPECTED_DATASET_SHA256 = "809a3a27ed0052e0fdf872542168af0a38337c1dc42f3d0b2af66d769cee5741"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_cases(path: Path = MANIFEST_PATH) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != "source-association-development-v1"
    ):
        raise ValueError("unexpected source-association development manifest")
    cases = manifest.get("graph_cases")
    if not isinstance(cases, list) or [case.get("id") for case in cases] != list(CASE_IDS):
        raise ValueError("the diagnostic requires the fixed five source-association cases")
    if any(case.get("split") != "development" for case in cases):
        raise ValueError("prefix diagnostic cannot consume heldout cases")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_DATASET_SHA256:
        raise ValueError("authored source-association dataset changed; use a new protocol")
    return manifest, digest, cases


def _eligible_facts(
    case: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = ScopeContext.model_validate(case["context"])
    inventory = [*case["facts"], *case.get("scope_controls", [])]
    eligible = [
        fact
        for fact in inventory
        if scope_allows(fact["scope"], context, (fact["scope"]["scope"],))
    ]
    facts = [fact for fact in eligible if fact in case["facts"]]
    controls = [control for control in case.get("scope_controls", [])]
    if any(
        scope_allows(control["scope"], context, (control["scope"]["scope"],))
        for control in controls
    ):
        raise ValueError(f"{case['id']} contains an unexpectedly eligible scope control")
    if len(facts) != int(case["query_eligible_count"]):
        raise ValueError(f"{case['id']} does not have its authored eligible fact count")
    return facts, controls


def _validate_vector(vector: Sequence[float], dimension: int, label: str) -> list[float]:
    if len(vector) != dimension or not vector:
        raise ValueError(f"{label} has the wrong vector dimension")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise ValueError(f"{label} contains a non-numeric vector value")
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values) or all(value == 0.0 for value in values):
        raise ValueError(f"{label} must be finite and nonzero")
    return values


def _rank(query: Sequence[float], documents: Mapping[str, Sequence[float]]) -> list[str]:
    return sorted(
        documents,
        key=lambda fact_id: (
            -cosine_similarity(list(query), list(documents[fact_id])),
            fact_id,
        ),
    )


def _arm(
    provider: EmbeddingProvider,
    cases: list[dict[str, Any]],
    *,
    label: str,
    prefixes: Mapping[str, str],
) -> dict[str, Any]:
    role_provider: EmbeddingProvider = provider
    if any(prefixes.values()):
        role_provider = PrefixEmbeddingProvider(
            provider,
            query_prefix=prefixes["query_prefix"],
            document_prefix=prefixes["document_prefix"],
            symmetric_prefix=prefixes["symmetric_prefix"],
        )
    observations: list[dict[str, Any]] = []
    for case in cases:
        facts, controls = _eligible_facts(case)
        inventory = [*facts, *controls]
        texts = [str(fact["content"]) for fact in inventory]
        vectors = role_provider.embed_batch_documents(texts)
        query_vector = role_provider.embed_query(str(case["query"]))
        if len(vectors) != len(inventory):
            raise ValueError(f"{case['id']} returned invalid document vectors")
        checked_vectors = {
            str(fact["id"]): _validate_vector(vector, role_provider.dim, f"{case['id']} document")
            for fact, vector in zip(inventory, vectors)
        }
        checked_query = _validate_vector(query_vector, role_provider.dim, f"{case['id']} query")
        by_id = {str(fact["id"]): checked_vectors[str(fact["id"])] for fact in facts}
        ranked = _rank(query_vector, by_id)
        ranks = {fact_id: ranked.index(fact_id) + 1 for fact_id in ranked}
        scores = {
            fact_id: float(cosine_similarity(checked_query, vector))
            for fact_id, vector in checked_vectors.items()
        }
        scored_rows = [
            {
                "id": fact_id,
                "eligible": fact_id in ranks,
                "cosine": scores[fact_id],
                "rank": ranks.get(fact_id),
            }
            for fact_id in sorted(checked_vectors)
        ]
        observations.append(
            {
                "case_id": case["id"],
                "eligible_fact_count": len(facts),
                "ranked_fact_ids": ranked,
                "scored_rows": scored_rows,
                "query_vector": checked_query,
                "document_vectors": checked_vectors,
                "target": {"id": case["target_id"], "rank": ranks[str(case["target_id"])]},
                "seed": {"id": case["seed_id"], "rank": ranks[str(case["seed_id"])]},
                "bridge": {"id": case["bridge_id"], "rank": ranks[str(case["bridge_id"])]},
                "scope_controls": {
                    "ids": [str(control["id"]) for control in controls],
                    "present_in_ranked": [
                        str(control["id"]) for control in controls if str(control["id"]) in ranked
                    ],
                },
            }
        )
    return {
        "label": label,
        "prefixes": dict(prefixes),
        "dimension": provider.dim,
        "cases": observations,
    }


def run_paired_diagnostic(
    provider_factory: Any,
    output: Path,
    *,
    model_metadata: Mapping[str, Any],
    code_sha: str,
    manifest_path: Path = MANIFEST_PATH,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both fresh arms with one provider factory per arm.

    ``provider_factory`` receives the arm label and must return a fresh provider.
    An existing output is always rejected. Every arm must be freshly measured.
    """
    manifest, dataset_sha, cases = load_cases(manifest_path)
    arm_settings: dict[str, dict[str, str]] = {
        "unprefixed": {"query_prefix": "", "document_prefix": "", "symmetric_prefix": ""},
        "nomic_task": {
            "query_prefix": "search_query: ",
            "document_prefix": "search_document: ",
            "symmetric_prefix": "",
        },
    }
    settings = {
        "protocol": PROTOCOL_VERSION,
        "dataset_sha256": dataset_sha,
        "case_ids": list(CASE_IDS),
        "options": dict(options or {}),
        "model": dict(model_metadata),
        "code_sha": code_sha,
        "arms": arm_settings,
    }
    binding = _digest(settings)
    if output.exists():
        raise ValueError("embedding prefix diagnostic output exists; use a fresh output")
    arms = []
    for label, prefixes in arm_settings.items():
        provider = provider_factory(label)
        try:
            arms.append(_arm(provider, cases, label=label, prefixes=prefixes))
        finally:
            provider.close()
    report = {
        "schema_version": SCHEMA_VERSION,
        "binding": binding,
        "dataset": {"path": str(manifest_path), "sha256": dataset_sha, "case_ids": list(CASE_IDS)},
        "protocol": settings,
        "arms": arms,
        "scope_claim": "Only authored scope-eligible facts are ranked; controls must remain absent.",
        "interpretation": "Development vector ranking diagnostic only; no graph or general retrieval claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["CASE_IDS", "MANIFEST_PATH", "run_paired_diagnostic", "load_cases"]
