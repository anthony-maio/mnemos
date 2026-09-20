# Embedding-outage retrieval package checkpoint

Frozen product `4bbab4f01fd6b5a8e1e4d76504eaa8d59d349da8` passes 1,894 source tests
with 4 skips, all six source/package checks, 722
isolated installed cases with 4 matching skips, and seven replay
assertions. Installed checks include transient-provider fallback and host
status propagation, and compare individual outcomes, imported module origins,
and package bytes with the frozen source and wheel.

Wheel: 305,928 bytes; SHA-256 `9af6667d86933570beecc940e55e637f70a9258e5339e948dc48ba9692782bf6`.

The [development record](../retrieval-provider-outage-development-v1/README.md)
preserves two failing closed-endpoint cases, four failing host status cases,
85 focused retrieval passes, and 104 host passes. These suites overlap.
Existing SQLite memories now remain available through scoped lexical matches
after a transient embedding failure. Empty results retain degraded status,
semantic scores remain unmeasured, and returned context identifies the weaker
retrieval mode. Scope, evidence views, correction, forgetting, and budgets
remain enforced. Provider timeouts and retries still precede fallback;
paraphrases without matching words may return no facts.

These checks establish package behavior. Native provider-outage recall is a
separate experiment and has not run on this product. Previous native Codex
and Claude stopped-service tests remain bound to their original packages and
establish raw-capture recovery only. Full-module quality, Desktop, broader
outage recovery, and the operational pilot remain open. No live host
configuration or memory database was changed by these package checks.

The evidence index binds copied raw records and validation helpers. This
README is outside that index.
