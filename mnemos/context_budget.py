"""Pack complete memory records using a supplied host tokenizer or a byte bound.

The host owns tokenizer selection and revision pinning. No model-name guessing,
network tokenizer download, or remote counting happens in this module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .retrieval import render_context_chunk
from .types import MemoryChunk


@dataclass(frozen=True)
class HostTokenizer:
    name: str
    revision: str
    count_tokens: Callable[[str], int]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.revision.strip():
            raise ValueError("Host tokenizer name and revision are required")
        if not callable(self.count_tokens):
            raise ValueError("Host token counter must be callable")


@dataclass(frozen=True)
class ContextPack:
    chunks: tuple[MemoryChunk, ...]
    text: str
    token_count: int
    token_count_method: str
    tokenizer_name: str | None
    tokenizer_revision: str | None
    fallback_reason: str | None
    omitted_ids: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "context_token_upper_bound": self.token_count,
            "context_token_count": (
                self.token_count if self.token_count_method == "host_tokenizer" else None
            ),
            "context_byte_count": len(self.text.encode("utf-8")),
            "token_count_method": self.token_count_method,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_fallback_reason": self.fallback_reason,
        }


class _CounterFailure(Exception):
    pass


def pack_context(
    chunks: Iterable[MemoryChunk],
    *,
    max_tokens: int,
    top_k: int,
    tokenizer: HostTokenizer | None = None,
) -> ContextPack:
    """Keep ranked, whole records within the final concatenated context budget.

    A failed counter restarts the entire selection with UTF-8 bytes, including
    records previously skipped by that counter. No partial mixed-unit budget
    survives. The byte fallback assumes a byte-based text tokenizer; a host
    with another tokenization contract must supply its own compatible counter.
    """
    if type(max_tokens) is not int or max_tokens < 0 or type(top_k) is not int or top_k < 0:
        raise ValueError("Context budget and top_k must be nonnegative integers")
    records = tuple(chunks)
    rendered = tuple(render_context_chunk(chunk) for chunk in records)

    def select(counter: Callable[[str], int]) -> tuple[list[int], str, int]:
        indices: list[int] = []
        text = ""
        count = 0
        if max_tokens == 0 or top_k == 0:
            return indices, text, count
        for i, record in enumerate(rendered):
            candidate = text + record
            cost = counter(candidate)
            if cost > max_tokens:
                continue
            indices.append(i)
            text, count = candidate, cost
            if len(indices) == top_k:
                break
        return indices, text, count

    def host_count(text: str) -> int:
        assert tokenizer is not None
        try:
            count = tokenizer.count_tokens(text)
        except Exception as exc:
            raise _CounterFailure("tokenizer_error") from exc
        if type(count) is not int or count <= 0:
            raise _CounterFailure("invalid_token_count")
        return count

    method = "utf8_bytes_upper_bound"
    reason = None
    if tokenizer is not None:
        try:
            indices, text, count = select(host_count)
            method = "host_tokenizer"
        except _CounterFailure as exc:
            reason = str(exc)
            indices, text, count = select(lambda value: len(value.encode("utf-8")))
    else:
        indices, text, count = select(lambda value: len(value.encode("utf-8")))
    selected = set(indices)
    return ContextPack(
        chunks=tuple(records[i] for i in indices),
        text=text,
        token_count=count,
        token_count_method=method,
        tokenizer_name=tokenizer.name if tokenizer else None,
        tokenizer_revision=tokenizer.revision if tokenizer else None,
        fallback_reason=reason,
        omitted_ids=tuple(record.id for i, record in enumerate(records) if i not in selected),
    )
