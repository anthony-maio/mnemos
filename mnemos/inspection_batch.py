"""Lossless bounded pages of explicitly requested inspection results."""

import hashlib
from typing import Any

from .forget_review import MAX_REVIEW_WIRE_BYTES, _CURSOR, _json, _record_layout, _wire_bytes


def render_inspection_batch(results: list[dict[str, Any]], cursor: str | None = None) -> str:
    """Bind all pages to exact complete results; references never select records."""
    if not 1 <= len(results) <= 20:
        raise ValueError("Inspect between one and twenty facts per batch")
    identities = []
    for result in results:
        fact = result.get("fact")
        if result.get("schema_version") != 1 or not isinstance(fact, dict):
            raise ValueError("Every requested fact must be available for inspection")
        if (
            not isinstance(fact.get("id"), str)
            or not fact["id"]
            or type(fact.get("version")) is not int
            or fact["version"] < 1
        ):
            raise ValueError("Inspection must identify the exact fact and version")
        identities.append(fact["id"])
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate facts are not allowed in an inspection batch")
    snapshot = {"results": results}
    digest = hashlib.sha256(_json(snapshot).encode()).hexdigest()
    fragments, layout_digest = _record_layout(snapshot)
    match = None
    if cursor is not None:
        if not isinstance(cursor, str) or (match := _CURSOR.fullmatch(cursor)) is None:
            raise ValueError("Invalid inspection cursor")
        if match[1] != layout_digest:
            raise ValueError("Inspection snapshot changed; restart the batch")
    first = int(match[2]) if match else 0
    if first >= len(fragments):
        raise ValueError("Inspection cursor is outside the snapshot")

    def page(end):
        last = end == len(fragments)
        return {
            "schema_version": 1,
            "status": "inspection",
            "complete": last,
            "review": {
                "format": "json_fragments_v2",
                "snapshot_digest": digest,
                "layout_digest": layout_digest,
                "first_fragment": first,
                "total_fragments": len(fragments),
                "last_page": last,
                "next_cursor": None if last else f"{layout_digest}:{end}",
                "instructions": "Read every page in order with the same ordered fact_ids. Paths reconstruct complete results including fact IDs, versions and evidence. Changed results invalidate the cursor. This inspects only explicitly requested facts and selects nothing for deletion.",
            },
            "fragments": fragments[first:end],
        }

    end = first + 1
    rendered = _json(page(end))
    if _wire_bytes(rendered) > MAX_REVIEW_WIRE_BYTES:
        raise ValueError("Inspection metadata exceeds the output bound")
    while end < len(fragments):
        proposed = _json(page(end + 1))
        if _wire_bytes(proposed) > MAX_REVIEW_WIRE_BYTES:
            break
        rendered = proposed
        end += 1
    return rendered
