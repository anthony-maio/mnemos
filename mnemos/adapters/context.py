"""Preserve service recall qualifications at the host boundary."""

from __future__ import annotations

import json
from typing import Any, Mapping


def format_retrieved_context(result: Any, *, max_chars: int) -> str:
    """Prefer the budgeted service context, including an intentional empty result.

    Older clients can return only facts. Keep their authority visible and quote
    content as data. Never cut a serialized record or a service context midway.
    """
    if not isinstance(result, Mapping):
        return ""
    context = result.get("context")
    if isinstance(context, str):
        return context if len(context) <= max_chars else ""
    facts = result.get("facts")
    if not isinstance(facts, list):
        return ""
    lines: list[str] = []
    used = 0
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        content = fact.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        status = fact.get("assertion_status")
        qualifier = (
            "asserted"
            if status == "asserted"
            else "unresolved conflict" if status == "conflict" else "unverified, may be outdated"
        )
        record: dict[str, str | int] = {"content": content}
        for key in ("id", "version", "scope", "scope_id"):
            value = fact.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                record[key] = value
        line = f"[{qualifier}] {json.dumps(record, ensure_ascii=True)}\n"
        if used + len(line) <= max_chars:
            lines.append(line)
            used += len(line)
    return "".join(lines)
