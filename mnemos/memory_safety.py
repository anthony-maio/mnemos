"""
mnemos/memory_safety.py — Shared memory write firewall.

Applies secret/PII policy checks and optional redaction across all long-term
memory write paths.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from .config import MemorySafetyConfig

PatternCategory = Literal["secret", "pii"]


@dataclass(frozen=True)
class MemorySafetyMatch:
    """A matched sensitive pattern found in content."""

    category: PatternCategory
    label: str
    count: int


@dataclass(frozen=True)
class MemorySafetyDecision:
    """Safety decision returned by the write firewall."""

    allowed: bool
    content: str
    reason: str
    matches: tuple[MemorySafetyMatch, ...]


class MemoryWriteFirewall:
    """Policy-driven safety checks for memory write content."""

    # UUID identity fields can contain a run of digits resembling a card number.
    # Recognize the complete identifier, never an arbitrary surrounding string.
    _UUID = re.compile(
        r"(?<![A-Za-z0-9_-])[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-"
        r"[1-8][0-9A-Fa-f]{3}-[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12}"
        r"(?![A-Za-z0-9_-])"
    )

    _PATTERNS: tuple[tuple[PatternCategory, str, re.Pattern[str]], ...] = (
        ("secret", "private_key", re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----")),
        ("secret", "aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("secret", "openai_key", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")),
        ("secret", "github_pat", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
        (
            "secret",
            "credential_assignment",
            re.compile(
                r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\b\s*[:=]\s*\S+"
            ),
        ),
        (
            "pii",
            "email",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        (
            "pii",
            "phone",
            re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
        ),
        (
            "pii",
            "ssn",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
        (
            "pii",
            "credit_card",
            re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        ),
    )

    def __init__(self, config: MemorySafetyConfig | None = None) -> None:
        self._config = config or MemorySafetyConfig()

    @property
    def config(self) -> MemorySafetyConfig:
        return self._config

    def apply(self, content: str) -> MemorySafetyDecision:
        """
        Apply safety policy to content.

        Returns a decision with allowed/block verdict and possibly redacted text.
        """
        if not self._config.enabled:
            return MemorySafetyDecision(
                allowed=True,
                content=content,
                reason="Memory safety disabled.",
                matches=(),
            )

        working = content
        matches: list[MemorySafetyMatch] = []

        for category, label, pattern in self._PATTERNS:
            found = list(pattern.finditer(working))
            if found and category == "pii" and label in {"phone", "ssn", "credit_card"}:
                # Recompute spans after earlier redactions have changed offsets.
                identifiers = [match.span() for match in self._UUID.finditer(working)]
                starts = [start for start, _ in identifiers]
                numeric_matches = []
                for match in found:
                    position = bisect_right(starts, match.start()) - 1
                    if position < 0 or match.end() > identifiers[position][1]:
                        numeric_matches.append(match)
                found = numeric_matches
            if not found:
                continue

            count = len(found)
            matches.append(
                MemorySafetyMatch(
                    category=category,
                    label=label,
                    count=count,
                )
            )
            action = self._action_for_category(category)
            if action == "block":
                return MemorySafetyDecision(
                    allowed=False,
                    content=content,
                    reason=f"Blocked {category} pattern: {label}.",
                    matches=tuple(matches),
                )
            if action == "redact":
                token = f"[REDACTED_{label.upper()}]"
                selected_starts = {match.start() for match in found}
                working = pattern.sub(
                    lambda match: token if match.start() in selected_starts else match.group(),
                    working,
                )

        if matches:
            return MemorySafetyDecision(
                allowed=True,
                content=working,
                reason="Applied safety policy.",
                matches=tuple(matches),
            )
        return MemorySafetyDecision(
            allowed=True,
            content=content,
            reason="No safety matches.",
            matches=(),
        )

    def _action_for_category(
        self, category: PatternCategory
    ) -> Literal["allow", "redact", "block"]:
        if category == "secret":
            return self._config.secret_action
        return self._config.pii_action


_SOURCE_FIREWALL: ContextVar[MemoryWriteFirewall | None] = ContextVar(
    "mnemos_source_firewall", default=None
)


def source_firewall() -> MemoryWriteFirewall:
    """Return the write policy a running source extraction must honor.

    The service binds its configured policy for one extraction, so source
    checks agree with what capture stored. Otherwise the default applies.
    """
    return _SOURCE_FIREWALL.get() or MemoryWriteFirewall()


@contextmanager
def bound_source_firewall(firewall: MemoryWriteFirewall) -> Iterator[None]:
    token = _SOURCE_FIREWALL.set(firewall)
    try:
        yield
    finally:
        _SOURCE_FIREWALL.reset(token)
