"""Render concise recall without rewriting fact content or its qualifications."""

from __future__ import annotations

from typing import Any


def compact_retrieval_response(full: dict[str, Any]) -> dict[str, Any]:
    """Keep literal context once, with scoped references for full evidence inspection.

    This changes presentation only. It does not select facts, summarize sources,
    or turn an available evidence record into a semantic support judgment.
    Context token metadata still describes the context, not the JSON envelope.
    """
    records = full["captured_evidence"] if full["view"] == "captured_evidence" else full["facts"]
    references = [
        {
            **{
                key: value
                for key, value in fact.items()
                if key not in {"content", "scores", "sources"}
            },
            "sources": [
                {key: value for key, value in source.items() if key != "excerpt"}
                for source in fact["sources"]
            ],
        }
        for fact in records
    ]
    return {
        **{key: value for key, value in full.items() if key not in {"facts", "captured_evidence"}},
        "response_format": "compact",
        "references": references,
    }
