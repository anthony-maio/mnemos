# Structured reasoning and generation settings package checkpoint

Frozen product `05c71d2b00f100ab4e7dd57426fe9d18efba9549` passes 2,005 source tests
with 4 skips, all six source/package checks, 804
isolated installed cases with 4 matching skips, and seven replay
assertions. Individual outcomes, module origins, and installed bytes match
the frozen source and wheel.

Wheel: 307,886 bytes; SHA-256 `8d75eef166c6e7f5558dd0a3b3c3562ac66fd7e89d9e1c8d7731688ef099260a`.

Structured output with explicitly enabled reasoning uses Ollama chat and
consumes final content, never the thinking channel. Managed configuration
exposes validated token budgets, timeouts, and temperature. Existing defaults
remain unchanged. Focused failures and repairs are preserved in the
[development archive](../structured-reasoning-development-v1/README.md).

This is the second full run of unchanged source. The
[first full run](../generation-settings-source-failure-v1/README.md) recorded
one Windows descriptor-replacement failure. A separate 25-cycle diagnostic
did not reproduce it. That cause remains unresolved; the original failed
result is preserved. No timeout or assertion was weakened for this rerun.

This checkpoint proves source/package and installed contract checks, not
extraction quality. The candidate and production prompt experiments remain
separate. Hermes host integration, full-module benefit, operational recovery,
and the pilot remain open. Nothing was installed into live host configuration.

The evidence index binds copied raw records and helpers. This README is
outside that index.
