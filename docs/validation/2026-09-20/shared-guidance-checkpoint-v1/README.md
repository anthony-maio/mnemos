# Shared instructions follow the active host's tool schema

Frozen source `697b6720b201d578c03ce33d98dff3dda6d2def8` passes 1,615 source tests
with 4 skips, all six source/package checks, 652 isolated
installed cases with 4 matching skips, and seven replay assertions.
Installed validation checks case outcomes, import origins, and package bytes
against the frozen source and wheel.

Wheel: 304,209 bytes; SHA-256 `49c7d791cd6b5d7645125cec72371b453607543b371e6834c6a2e122774f42b0`.

## Observed conflict and repair

The [three-host v5 attempt](../three-host-native-v5/README.md) preserved a
conflict in generated project instructions. They required MCP scope fields
that the provider-owned Hermes tools reject. Both fields appeared in rejected
native calls, and the instructions appeared in every recorded Hermes request.
Query-only retries retrieved both taught facts. The fixed acceptance rules
still classify that attempt as failed.

Generated guidance now asks for scope fields only when the active schema
exposes them. Provider-owned tools omit them and retain their pinned scope.
Durable writes select an exposed store or remember tool, and consolidation is
requested only when supported. No tool schema, scope firewall, or lifecycle
acceptance requirement changed. No live host configuration was rewritten.

The focused canonical environment passed 90 tests with two opt-in skips.
The new regression checks generated guidance and real Hermes dispatch: invalid
scope overrides remain rejected before the client, while a query-only call
uses the provider's project context. The archived initial red/green runs used
the system interpreter; the full source and installed runs above are the
authoritative package checks.

## Evidence limits

This establishes the package repair and its dispatch contract. Native v5
remains failed. A new native Hermes prerequisite cohort is separately bound
to this package, with unchanged requirements and all three outcomes retained.
Hermes-only setup does not install the shared Codex AGENTS instructions, so
that cohort cannot measure the guidance repair's effect. A subsequent
three-host run is needed to exercise those shared instructions.

Native continuity, complete capture, real-model quality, Desktop, and the
operational pilot remain open. This README is outside the raw evidence index.
