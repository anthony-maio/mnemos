"""
tests/test_memory_safety.py — Shared memory safety firewall tests.
"""

from __future__ import annotations

from mnemos.config import MemorySafetyConfig
from mnemos.memory_safety import MemoryWriteFirewall


def test_firewall_blocks_secret_by_default() -> None:
    firewall = MemoryWriteFirewall(MemorySafetyConfig())
    decision = firewall.apply("api_key=supersecretvalue")
    assert decision.allowed is False
    assert "secret" in decision.reason.lower()


def test_firewall_redacts_pii_by_default() -> None:
    firewall = MemoryWriteFirewall(MemorySafetyConfig())
    decision = firewall.apply("Contact me at jane@example.com tomorrow")
    assert decision.allowed is True
    assert "REDACTED_EMAIL" in decision.content


def test_firewall_can_allow_all() -> None:
    firewall = MemoryWriteFirewall(
        MemorySafetyConfig(
            enabled=True,
            secret_action="allow",
            pii_action="allow",
        )
    )
    content = "api_key=abc123 and jane@example.com"
    decision = firewall.apply(content)
    assert decision.allowed is True
    assert decision.content == content
