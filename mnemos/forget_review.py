"""Lossless, bounded presentation of related-forget snapshots for MCP clients.

The storage preview and its compare-and-set deletion contract remain authoritative.
Paging and short references never select records or authorize deletion themselves.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from .errors import (
    RelatedForgetPreviewIncomplete,
    RelatedForgetPreviewStale,
    RelatedForgetSelectionInvalid,
)

MAX_REVIEW_WIRE_BYTES = 12_000
_FRAGMENT_BYTES = 2_000
_CURSOR = re.compile(r"([0-9a-f]{64}):(0|[1-9][0-9]{0,9})", re.ASCII)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _wire_bytes(text: str) -> int:
    # Claude's native stream wraps the MCP string in another JSON string. Count
    # that escaping too; this bound is not a tokenizer compatibility claim.
    return len(json.dumps({"result": text}).encode("utf-8"))


def _with_references(preview: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(preview)
    source_refs = {}
    for index, source in enumerate(result.get("source_events", []), 1):
        ref = f"s{index}" if source.get("selection_allowed") is True else None
        source["selection_ref"] = ref
        source_refs[source["source_event_id"]] = ref
    for index, candidate in enumerate(result.get("candidates", []), 1):
        candidate["selection_ref"] = f"f{index}"
        candidate["required_source_refs"] = [
            source_refs.get(source_id)
            for source_id in candidate.get("required_source_event_ids", [])
        ]
    # A transport may sort keys. Keep these fields in the final fragment as
    # well as withholding the page-level copy until every fragment is returned.
    if "apply_preconditions" in result:
        preconditions = result.pop("apply_preconditions")
        result["apply_preconditions"] = preconditions
    return result


def _fragments(value: Any, path: list[str | int]) -> Iterator[dict[str, Any]]:
    if len(_json(value)) <= _FRAGMENT_BYTES:
        yield {"path": path, "value": value}
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _fragments(item, [*path, key])
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _fragments(item, [*path, index])
    elif isinstance(value, str):
        offset = 0
        while offset < len(value):
            # JSON escaping costs vary for Unicode, quotes and backslashes.
            lo, hi = 1, min(_FRAGMENT_BYTES, len(value) - offset)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(_json(value[offset : offset + mid])) <= _FRAGMENT_BYTES:
                    lo = mid
                else:
                    hi = mid - 1
            yield {
                "path": path,
                "offset": offset,
                "total_characters": len(value),
                "text": value[offset : offset + lo],
            }
            offset += lo
    else:
        raise ValueError("Forget review contains an unsupported large value")


def render_forget_review(preview: dict[str, Any], cursor: str | None = None) -> str:
    """Keep small previews intact; page larger ones without dropping any field."""
    cursor_match = None
    if cursor is not None:
        if not isinstance(cursor, str) or (cursor_match := _CURSOR.fullmatch(cursor)) is None:
            raise ValueError("Invalid forget review cursor")
    snapshot = _with_references(preview)
    serialized = _json(snapshot)
    snapshot_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if cursor_match is not None and cursor_match[1] != snapshot_digest:
        raise RelatedForgetPreviewStale()
    if cursor is None and _wire_bytes(serialized) <= MAX_REVIEW_WIRE_BYTES:
        return serialized
    fragments = list(_fragments(snapshot, []))
    first = int(cursor_match[2]) if cursor_match is not None else 0
    if first >= len(fragments):
        raise ValueError("Forget review cursor is outside the snapshot")

    def page(end: int) -> dict[str, Any]:
        last = end == len(fragments)
        return {
            "schema_version": 1,
            "status": "preview",
            "complete": preview.get("complete", False),
            "preview_digest": preview.get("preview_digest"),
            "apply_preconditions": preview.get("apply_preconditions") if last else None,
            "review": {
                "format": "json_fragments_v1",
                "snapshot_digest": snapshot_digest,
                "first_fragment": first,
                "total_fragments": len(fragments),
                "last_page": last,
                "next_cursor": None if last else f"{snapshot_digest}:{end}",
                "instructions": (
                    "Read all pages in order. Paths locate exact snapshot fields; text fragments "
                    "continue at character offsets. Nothing is summarized. Copy next_cursor into "
                    "mnemos_forget_preview with the same fact_id, expected_version and limit. "
                    "Only the last page provides apply_preconditions for a complete snapshot. "
                    "Use reviewed f/s selection_ref values to avoid retyping long IDs. "
                    "Review advisory wording and every eligible source, including deferred "
                    "sources with no derived facts."
                ),
            },
            "fragments": fragments[first:end],
        }

    end = first + 1
    rendered = _json(page(end))
    if _wire_bytes(rendered) > MAX_REVIEW_WIRE_BYTES:
        raise ValueError("Forget review metadata exceeds the output bound")
    while end < len(fragments):
        proposed = _json(page(end + 1))
        if _wire_bytes(proposed) > MAX_REVIEW_WIRE_BYTES:
            break
        end += 1
        rendered = proposed
    return rendered


def resolve_review_selections(
    preview: dict[str, Any],
    preview_digest: str,
    fact_refs: list[str],
    source_refs: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve explicit short references; storage still rechecks the digest atomically."""
    if preview.get("preview_digest") != preview_digest:
        raise RelatedForgetPreviewStale()
    if preview.get("complete") is not True:
        raise RelatedForgetPreviewIncomplete()

    def resolve(refs: list[str], prefix: str, records: list[dict[str, Any]], key: str) -> list[str]:
        if len(refs) > 500 or len(set(refs)) != len(refs):
            raise RelatedForgetSelectionInvalid()
        result = []
        for ref in refs:
            if re.fullmatch(prefix + r"[1-9][0-9]{0,2}", ref, re.ASCII) is None:
                raise RelatedForgetSelectionInvalid()
            index = int(ref[1:]) - 1
            if index >= len(records):
                raise RelatedForgetSelectionInvalid()
            record = records[index]
            if prefix == "s" and record.get("selection_allowed") is not True:
                raise RelatedForgetSelectionInvalid()
            result.append(str(record[key]))
        return result

    return (
        resolve(fact_refs, "f", preview.get("candidates", []), "fact_id"),
        resolve(source_refs, "s", preview.get("source_events", []), "source_event_id"),
    )
