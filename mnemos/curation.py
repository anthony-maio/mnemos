"""
mnemos/curation.py — Lightweight durable-memory heuristics.

These checks intentionally stay simple and explicit. They are meant to reject
obvious transient or noisy memory candidates before they pollute long-term
storage, not to replace the main encoding pipeline.
"""

from __future__ import annotations

import re

_TIMESTAMP_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}(?::\d{2})?z?\b", re.I)
_COMMAND_ECHO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bcommand completed\b"),
    re.compile(r"(?i)\bcompleted in \d+(?:\.\d+)?s\b"),
    re.compile(r"(?i)\bstarted at\b"),
    re.compile(r"(?i)\bexit code \d+\b"),
)


def normalize_curation_text(value: str) -> str:
    return " ".join(value.split())


def _looks_repetitive(tokens: list[str]) -> bool:
    if len(tokens) < 6:
        return False
    for window_size in (1, 2, 3):
        if len(tokens) < window_size * 3 or len(tokens) % window_size != 0:
            continue
        phrase = tokens[:window_size]
        repeats = len(tokens) // window_size
        if repeats >= 3 and phrase * repeats == tokens:
            return True
    return False


def durable_memory_skip_reason(text: str) -> str | None:
    """
    Return a human-readable skip reason when content is obviously too transient
    or noisy for durable memory.
    """
    normalized = normalize_curation_text(text)
    if not normalized:
        return "empty content"

    lowered = normalized.lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if _looks_repetitive(tokens):
        return "repetitive status noise"

    has_timestamp = bool(_TIMESTAMP_PATTERN.search(lowered))
    has_command_echo = any(pattern.search(lowered) for pattern in _COMMAND_ECHO_PATTERNS)
    if has_timestamp and has_command_echo:
        return "transient command/timestamp noise"

    return None
