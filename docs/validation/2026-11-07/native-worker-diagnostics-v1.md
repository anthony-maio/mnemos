# Current worker costs: Git resolution and retrieval

Fresh Git project resolution takes 61-70 ms at the median in this diagnostic,
with large outliers. Prompt retrieval takes about 94 ms inside the service.
Real query embedding takes about 5 ms. These results support investigating the
Git execution path and storage/evidence queries before changing embedding
concurrency or weakening identity checks.

This is a separate instrumented run on the current installed `3d1866d5` product
and corrected native executable. It does not replace the
[complete comparison](native-launcher-performance-v2.md), whose 100 ms hook gate
still fails. The [protocol](../../plans/2026-11-07-native-worker-diagnostics-v1.md)
was frozen at `40f41fc1a7c6f04d64277b85a28319a42a65f24a` before execution.

## Observed work

Each event has one warmup and five observed complete commands. Eight separate
recall controls include three warmups. All 20 observations pass their behavior
checks. The following values are medians in milliseconds across the five
observed commands per event. Nested spans overlap and must not be added together.
Labels with several calls report their sum within each command before taking
the median.

| Observed work | Stop | UserPromptSubmit |
| --- | ---: | ---: |
| Complete instrumented command | 134.37 | 263.40 |
| Worker dispatch | 100.36 | 221.54 |
| Effective-settings check | 2.62 | 2.87 |
| Git-based project resolution | 61.00 | 70.08 |
| Git subprocess inside resolution | 58.95 | 67.64 |
| Adapter capture | 34.32 | 35.43 |
| Durable outbox append | 8.72 | 9.06 |
| Outbox acknowledgement | 8.22 | 9.20 |
| Service capture dispatch | 2.90 | 2.76 |
| Adapter retrieval | -- | 104.87 |
| Service retrieval dispatch | -- | 93.53 |
| Real query embedding | -- | 4.56 |
| Vector storage ranking | -- | 32.04 |
| Lexical storage ranking | -- | 30.68 |
| Fact-evidence lookup | -- | 11.32 |
| Retained-source status checks | -- | 0.45 |

The largest observed Git subprocess is 171.02 ms for Stop and 513.23 ms for
prompt submission. Complete-command maxima are 250.53 and 863.96 ms. These
small, instrumented samples locate costs; they do not establish a general
operating-system bound or identify the cause of the outliers. Each hook performs
one fresh Git subprocess. No identity cache or simplified scope check is used.

The current fact-evidence query selects a fact's evidence and orders it by
creation time. Its table declaration has no secondary index for that lookup.
A read-only inspection of the frozen seed confirms a full table scan and a
temporary B-tree for sorting. A separate exact-output index probe is warranted.
The 11.32 ms measurement
covers all evidence lookups for a prompt, not one SQL call, and the index's
benefit has not yet been measured.

## Verification and preservation

The disposable runtime uses the same wheel, 96 package files, 57 dependencies,
17 BGE assets, native executable and pristine 10,000-fact seed as the complete
comparison. Wrappers call original functions without changing arguments,
results or exceptions. Eight preflight checks verify those properties and
metadata-only logging. Generated entry scripts compile, install tracing before
dispatch and preserve implicit subprocess environment inheritance.

The independent audit verifies 12 handled worker dispatches, 18 correlated
service dispatches and six real prompt embeddings, including warmups. All 20
full decoded outputs match the corresponding prior complete-run outputs.
All 12 captures survive service restart and their jobs complete. Accepted facts
and their evidence/vectors remain unchanged; database integrity and foreign-key
checks pass. Source, package, model and seed bindings remain unchanged, and
owned-process cleanup is verified. The audit makes no model calls.

The verified private archive contains 162 members, 183 logical files and
134,770,439 bytes. SHA-256:
`097bbaccc6d31b5a569976d49671eb00b9d13b3c2ff5f85ee7141539b0a1ae27`.
It retains raw observations, timing spans, disposable databases, instrumentation,
drivers, checks and the audit, with the prior comparison and seed archives
referenced by hash. See the [receipt](native-worker-diagnostics-v1-receipt.json).

Next, compare Git executable paths while retaining a fresh Git invocation and
the same environment and identity semantics. Include ordinary repositories and
linked worktrees, not just the plain-folder benchmark fixture. Separately test
the fact-evidence index on copied data, preserving every returned row and tie
order. Any resulting candidate still needs a complete installed comparison;
neither component result can replace the original 100 ms requirement.

No production code, live configuration or publication changes were made. The
native executable remains experimental and the private pilot remains inactive
with zero actual days. Semantic-quality and native-host acceptance requirements
remain open.
