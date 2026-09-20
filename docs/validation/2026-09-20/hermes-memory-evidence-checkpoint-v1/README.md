# Hermes durable evidence callback package checkpoint

Frozen product `244613aeda446f50d7cbea59a1fa80f30fab2f98` passes 1,976 source tests
with 4 skips, all six source/package checks, 775
isolated installed cases with 4 matching skips, and seven replay
assertions. Individual outcomes, module origins, and installed bytes match
the frozen source and wheel.

Wheel: 307,541 bytes; SHA-256 `c5452e7b6810da567fc9a8ae9e46af9c0d96280bc0b78653218b17d6e6a4235e`.

Compatible hosts can pass a separate durable evidence snapshot to ordinary
turn and session-end callbacks under memory_evidence_api_version 1. An empty
snapshot captures nothing and never substitutes callback or model-facing
text. Native IDs, origins, and summary classification are validated before
any enqueue. Legacy callers retain the existing contract.

The [development archive](../hermes-memory-evidence-development-v1/README.md)
preserves 13 original failures, one assertion-fixture error, the additional
malformed-classification failure, and 90 final focused passes including
14 new cases. Tests preserve original payloads through copied rows, session
switches, and provider restart; marked summaries remain capture-only.

The host must still supply the authoritative projection and preserve distinct
origins for a summary and genuine user turn in a hybrid carrier. The isolated
Hermes candidate and its actual SessionDB/MemoryManager integration remain
separate validation work. This package checkpoint does not establish native
compression, a live upgrade, complete capture, or an identity migration.
Real-model quality and the operational pilot remain open.

The evidence index binds copied raw records and helpers. This README is
outside that index.
