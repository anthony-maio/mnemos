# Functional local memory implementation status

This record covers the unreleased `wt/functional-local-memory` worktree. It
separates implemented behavior from acceptance evidence. The original task
boundaries and thresholds remain in the [implementation plan](../../plans/2026-09-16-functional-local-memory.md).

A subsequent source repair preserves recall authority at the host boundary.
Claude's adapter had discarded the service's assertion-status and history labels;
Codex and Hermes discarded them in fact-only compatibility responses. The shared
formatter now preserves service context, labels and quotes fallback records,
respects an empty context, and omits oversized context rather than emitting a
partial record. The legacy CLI formatter also retains the shared qualifications.
Nine adapter regressions failed before the repair; 97 focused adapter/import
checks passed afterward, followed by 15 authority tests including three actual
SQLite-backed service-to-adapter checks. Black and strict mypy pass. The
[full-suite run](../2026-09-17/hook-context-authority-v1/README.md) FAILED with
981 passed, four skipped, and one failure: the captured-history restart test
found no fact after its initial remember/maintenance step. The retained fixture
also had no events or jobs. The same test passed in isolation. A separate
[UUID metadata probe](../2026-09-17/uuid-metadata-rejection-v1/README.md)
reproduced rejection of a generated UUID whose numeric tail matched the
credit-card filter. This is a confirmed write defect, but the failed full run
did not retain its generated ID, so its specific cause remains unproven.
Installed/native acceptance has not been rerun. This does not close the native
v6 failure below.

The [latest native forget/replay attempt](../2026-09-17/native-codex-claude-continuity-v6/README.md)
is FAILED overall. Codex taught the fact, Claude corrected it, and fresh Codex
retrieved version 2 across restarts. Claude then deleted it and retrieved an
unrelated retained preference. Direct service replay after restart suppressed
the original teaching event and rejected correction of the deleted target;
maintenance and another restart preserved deletion. The final Codex session
retrieved a surviving unknown captured mention, repeated its value as current,
and skipped the second requested control query. Its native post-forget control
and answer behavior therefore failed acceptance. All three Codex processes exited
zero under explicit unelevated mode; reverse and final Stop timeouts remain.

The [v5 failure](../2026-09-17/native-codex-claude-continuity-v5/README.md)
is also preserved. Claude used the wrong query for the unrelated-fact control;
the fact remained in SQLite. V6 checked the exact query rather than accepting
the model's final prose. These observations require investigation of historical
capture presentation and instruction boundaries. Current labels alone did not
prevent the observed stale answer.

The [earlier native continuity check](../2026-09-17/native-codex-claude-continuity-v4/README.md)
observed both directions on installed `ebe1b0d`: Codex stored, Claude retrieved
and corrected after restart, then a fresh Codex session retrieved version 2
after another restart. All three native queries excluded the foreign canary.
Both Codex processes exited 0xc0000005; reverse Stop timed out and Claude capture
remained degraded. Two no-integration controls exited zero. The cause remains open.
Separate [MCP-only and hooks-only controls](../2026-09-17/native-codex-isolation-controls-v1/README.md)
also exited zero. The hooks-only control retained both capture roles after
database reopen despite a Stop timeout. Sandbox mode and tool-approval
differences prevent assigning the combined-run crash to one component yet.

A [combined control](../2026-09-17/native-codex-combined-control-v1/README.md)
with all generated hooks, MCP, and explicit unelevated sandbox mode completed
native scoped retrieval and exited zero. Two hook events survived SQLite reopen.
Stop still timed out, and operator/tool-approval time prevents using the elapsed
time as a performance result. The earlier crashes remain unresolved.

The current source adds allowlisted Claude diagnostic reasons for missing
identity and ambiguous transcript association. Its [focused regression](../2026-09-17/hook-diagnostic-reasons-v1/README.md)
passed 68 tests and strict typing. It preserves separate captures and degraded
status. This change is newer than the installed ebe1b0d native controls and has
not received a new installed-package or native-host acceptance run.

The [capture import boundary](../2026-09-17/hook-import-boundary-v1/README.md)
now keeps engine, embedding, and storage imports out of client startup. All 92
focused cases passed with two platform skips, and strict product typing passed.
Public exports resolve on first access. Diagnostic shell timings remain variable;
the native Stop deadline is not yet accepted.
The [complete package checkpoint](../2026-09-17/lazy-import-package-checkpoint-v1/README.md)
at `ebe1b0d` passed 962 tests with four platform skips and all static/build
checks. Its fresh installed wheel passed 268 selected cases with two matching
skips and seven replay assertions, including subprocess import provenance.

The [capture interruption regression](../2026-09-17/capture-interruption-v1/README.md)
forces a host client process to exit before sending or after service commit.
Capture now persists sanitized evidence before startup and HTTP waits, then
removes it only on a terminal reply. Both interrupted paths replay once after
SQLite/service restart. Seven new cases and 84 related cases pass, including
forget suppression before interrupted capture is acknowledged. Native Stop
timeout and host shutdown acceptance remain open.

The [captured-assertion repair](../2026-09-17/captured-assertion-regression-v1/README.md)
preserves automatic passthrough history while labeling newly indexed copies
as unknown captured evidence. Host context now marks it unverified and possibly
outdated. Canonical dedupe preserves explicit authority and source identities,
and related-forget review still includes captured copies. All 68 focused
regressions pass. This repairs evidence labeling; it does not suppress old
mentions, migrate existing records, or prove native answer selection.
The [native qualification observation](../2026-09-17/native-codex-claude-continuity-v2/README.md)
returned the corrected target asserted at version 2 and captured history marked
unknown with the qualifier visible in Claude's actual context. That runner
failed a fixed two-old-mentions assertion after receiving one, before its second
service restart. A [revised rerun](../2026-09-17/native-codex-claude-continuity-v3/README.md)
then failed during Codex shutdown before reaching Claude. Both reports remain
failed; the earlier observation is narrower than end-to-end acceptance.

The [native Codex-to-Claude continuity check](../2026-09-17/native-codex-claude-continuity-v1/README.md)
now proves an actual Codex store/consolidate call, Claude retrieval after service
restart, and Claude correction preserving the fact ID at version 2 through
another restart. Both native Claude queries excluded the foreign canary.
The corrected query also returned two independently captured records repeating
the old value. Stable-target correction passes; stale conversation context is
an open correctness gap. The fixture disabled all modules, so investigate the
verbatim automatic-capture fallback without treating this as an extraction
quality result. Codex exited nonzero after /exit, and Claude capture remained
degraded. Reverse recall, cross-host forget/replay, and Hermes remain open.

Source `1068a62` repairs non-ASCII capture in both host adapters by decoding
hook JSON as UTF-8 rather than the Windows stdin locale. Five regressions
failed before the repair, and 75 related cases pass afterward. The
[native Codex rerun](../2026-09-17/native-codex-utf8-v1/README.md) captured the
exact Unicode assistant response and user prompt through generated hooks;
both were present after service closure and database reopen. Actual scoped
MCP retrieval passed too. Shutdown still reached its watchdog, and native
mutation, cross-host, and complete conversation coverage remain open.

The current adapter also reports degraded capture when a Codex capture hook
has no usable text. Nine regression cases failed before this change; all ten
new cases and related adapter/configuration checks now pass (50 passed, two
platform skips). This repairs a separate reporting defect; the later Unicode
failure and native rerun are recorded above. See the
[coverage regression](../2026-09-17/codex-capture-coverage-v1/README.md).

Source `d58e082` repairs an argument mismatch observed in a native Codex
interactive call: MCP retrieval now accepts structured scope lists as well as
legacy comma-separated strings. Empty or invalid lists are rejected, and the
stdio regression verifies project isolation. The [native failure](../2026-09-17/native-codex-interactive-v1/README.md)
and [regression evidence](../2026-09-17/mcp-scope-arguments-v1/README.md) are preserved.
The complete package verification below includes this repair. The subsequent
[native interactive check](../2026-09-17/native-codex-interactive-v2/README.md)
returned the scoped seed through an approved MCP call against installed
`30eaf42`. Bounded native retrieval passes. Shutdown timed out and assistant
capture was absent, so complete conversation and lifecycle acceptance remain open.

The latest package checkpoint is source `b7b4d6b`. Its frozen verification
passed 958 tests with four platform-specific skips, formatting, typing,
JavaScript, build, and Twine checks. A fresh installed wheel passed 264
contract cases with two matching platform skips and seven replay assertions. These are package and installed
runtime checks; they do not close native-host, real-model, latency, pilot, or
publication gates. A seven-case supplement explicitly binds interruption-test
children to the installed package after correcting a source import in the test.
See the [capture interruption package checkpoint](../2026-09-17/capture-durable-package-checkpoint-v1/README.md).

This checkpoint repairs retrieval across a concurrent correction or forget:
candidates are refreshed after asynchronous state classification, so scoring,
graph expansion, and budgeting use current records. All six service-level
regressions fail on the prior source and pass with the repair. See the
[concurrency reproduction](../2026-09-17/retrieval-concurrent-revision-v1/README.md).

The Claude startup default at `dedfcfd` is included in this complete source and
installed-package checkpoint. A separate isolated native `alwaysLoad` control,
using an older wheel with a manifest overlay, passed nine startup/routing
criteria. See the
[startup control](../2026-09-17/native-claude-alwaysload-v2/README.md).

Source `6cde465` adds capture recovery when the service commits an event but
loses its HTTP response. The client now handles `httpx.RemoteProtocolError`
through the existing durable outbox path. Four injected HTTP/SQLite cases
failed before the repair and passed afterward; 35 related service cases pass.
All four recovery cases also pass in the complete source and installed-package
checkpoint above. See the [lost-acknowledgment regression](../2026-09-17/service-lost-ack-regression-v1/README.md).

The first full run of `6cde465` stopped with 917 passed, four skipped, and one
failed service auto-start check. Its initial health response was `degraded`;
the preserved fixture had no runtime directory or service log at inspection.
The unchanged isolated test subsequently passed, with health returning in
952.09 ms under the original ten-second startup limit. This does not establish
the cause of the full-run failure. Both the [failed checkpoint](../2026-09-17/lost-ack-package-failure-v1/README.md)
and [isolated diagnostic](../2026-09-17/lost-ack-startup-diagnostic-v1/README.md)
remain separate evidence.

Source `a607a5a` distinguishes an exited auto-start process from a live process
that reaches its startup deadline. The client reports a numeric exit code
promptly, accepts a competing owner descriptor, and preserves queued capture.
The [before-fix installed probe](../2026-09-17/startup-exit-diagnostic-v1/README.md)
observed a child exit at 652.265 ms followed by a generic client response at
2,048.842 ms. Six deterministic startup cases and 29 related service cases
pass with the repair. See the [regression record](../2026-09-17/startup-exit-regression-v1/README.md).
The [after-fix installed probe](../2026-09-17/startup-exit-diagnostic-v2/README.md)
observed a child exit at 655.468 ms and an explicit exit-code response at
686.713 ms under the same two-second deadline. This diagnostic repair is
included in the package checkpoint above. It does not establish the cause of
the earlier full-suite startup failure.

The later [access-metadata phase probe](../2026-09-17/access-metadata-phases-v1/README.md)
times the current storage implementation on two fresh 10,000-fact fixtures.
Across 100 five-fact updates on D:, total p95 was 313.7343 ms and commit p95
was 313.5566 ms; the SQL stages stayed below 0.5 ms per call. The same helper
on C: reported total p95 2.58 ms and commit p95 2.4211 ms. Both retained
WAL/FULL and all counters after reopen. The D-then-C order and concurrent
work prevent a controlled speedup claim. This localizes the observed cost to
the durability boundary and does not close the end-to-end recall gate.

A [33-call follow-up on current source](../2026-09-17/retrieval-commit-profile-v1/README.md)
again found commit dominated the write cost: total p95 86.372 ms and commit
p95 86.213 ms on D:, with WAL/FULL retained and all counters verified after
reopen. This smaller follow-up did not reproduce the original roughly 796 ms
metadata tail. It adds an observation of timing variability; it is not an
end-to-end recall pass or a measured product speedup.

An earlier full-suite attempt at `e1fd56e` finished with 897 passed, one failed,
and four skipped. A list response in the synthetic Codex continuity fixture
lacked `facts`; its cause is under investigation. The runner stopped before
building a new wheel. The [failed checkpoint](../2026-09-17/service-runtime-checkpoint-failure-v1/README.md)
remains separate from the earlier passing package checkpoint.

The [isolated rerun](../2026-09-17/crosshost-isolated-diagnostic-v1/README.md)
passed that exact parameter with its original timeout and durability settings.
All 40 traced requests completed, including a list response with one fact.
It did not reproduce or explain the full-suite failure.

The next full run at `33c6dd9` also finished with 897 passed, one failed,
and four skipped. The continuity case passed; the generated PowerShell hook
bootstrap exceeded its existing ten-second deadline. The
[second failed checkpoint](../2026-09-17/powershell-checkpoint-failure-v1/README.md)
is preserved, and no new wheel was built from it. Two dashboard unit tests
now inject client facades to avoid contacting the default service; their
16 focused control-plane/UI cases pass. This independent isolation repair
does not establish the cause of either earlier timeout or response failure.

The [Claude capture diagnosis](../2026-09-17/claude-capture-association-diagnostic-v1/README.md)
reproduced the provisional-user association warning when the prompt hook lacks
a native prompt ID. A completed-transcript control succeeds when supplied that
ID. The [native hook-time probe](../2026-09-17/native-claude-hook-identity-v3/README.md)
then observed no prompt ID and no transcript file before or after the actual
prompt hook. The transcript appeared by Stop with native identifiers. Reading
the transcript earlier cannot resolve that observed turn's association;
capture diagnostics and complete lifecycle acceptance remain open.

| Plan item | Current implementation and evidence | Acceptance state |
| --- | --- | --- |
| T01 Runtime | Configured providers, SQLite default, explicit mock rejection, readiness checks, health reporting, and installed entry points are covered by the package and wheel checks. | Installed doctor v5 passed configuration cases at 6600dc2 without a provider reachability probe. Managed service logging, omitted-provider startup, and generated host provenance are covered by the full 8d865a5 checkpoint and its 110 passing installed contract cases with two matching skips, including actual subprocess recall. |
| T02 Scope | Profile, project, workspace, membership, enabled-project capture, foreign-project filtering, and legacy/routing review are implemented and covered by contract tests. | Local scope contract tested. Native identity coverage across every host remains open. |
| T03 Durable writes | Atomic source and job creation, leases, retries, CAS, revisions, evidence, rollback, and restart persistence are implemented and tested. | Local durability tested. Long-running operational recovery remains open. |
| T04 Local service | Authenticated loopback ownership, private descriptors, outbox handling, bounded shutdown, and HTTP/subprocess paths are implemented and tested. | Local service and installed replay checks pass. Native host lifecycle remains open. |
| T05 Maintenance | Service-owned scheduling, bounded batches, retry/failure state, retention, and visible maintenance status are implemented and tested. | Deterministic maintenance tested. A daily-use pilot has not run. |
| T06 Evidence and safety | Write firewall, source validation, role labels, governed explicit writes, private paths, retention controls, and local-original availability are implemented and tested. | Safety contracts pass. Real-model extraction precision and human acceptance remain unproved. |
| T07 Surprisal | Automatic gating, explicit-remember bypass, session-safe history, and diagnostics are implemented and covered by deterministic tests. | Mechanism contract tested. Real filtering benefit remains open. |
| T08 Correction | Stable identity, expected-version CAS, revisions, canonical updates, evidence links, and graph refresh are implemented and tested. | Native Claude corrected a Codex-stored fact after restart, preserving its ID at version 2; a fresh Codex session retrieved it after another restart. Captured old mentions remain unknown and visibly qualified as possibly outdated. General stale-context answer consistency remains open. |
| T09 Retrieval | Scoped cosine retrieval, lexical candidate union, one scoring contract, and context budgeting are implemented. | Scope and top-1 tests pass. The corrected v4 measurement has recall p95 1,088.094 ms against the 500 ms gate; capture p95 64.279 ms passes the 100 ms gate. Later variable timing comparisons do not establish a speedup. |
| T10 Graph | Persisted mutual top-k links, scope-safe traversal, bounded recovery, insertion-order controls, and capacity diagnostics are implemented and tested. | Structural controls pass. Graph utility and eligible-target coverage are not established by the available evidence. |
| T11 Forget | Exact fact deletion, suppression, replay protection, inflight handling, restore behavior, and selected related-removal preview/apply are implemented and tested. | Local and installed related-forget checks pass. Native v6 observed Claude deletion and direct service replay suppression after restart. A fresh Codex answer reused an independent captured copy, so broader native related-forget and answer acceptance remain open. Logical forgetting does not erase external transcripts or historical backups. |
| T12 Claude Code | Hook capture, transcript deltas, checkpoints, capture-before-recall, identity diagnostics, and replay repair are implemented. | Native v18 on dedfcfd passed all 15 fact-lifecycle criteria with the new startup default, including correction, forgetting, restart, and reconstructed replay. Degraded hook capture keeps the overall result partial. Native extraction and complete capture acceptance remain open. See the [native lifecycle archive](../2026-09-17/native-claude-lifecycle-v18/README.md). |
| T13 Codex | Generated hook configuration, UTF-8 payload handling, explicit MCP fallback, and bounded transport checks are implemented. | Installed ebe1b0d completed native store/consolidate and retrieved a Claude-corrected version 2 after restart. All three v6 Codex processes exited zero with explicit unelevated mode; reverse/final Stop timed out. V4 crashes remain preserved. V6's final session repeated a deleted value from an unknown captured copy and skipped a requested query. Complete capture and native answer/lifecycle acceptance remain open. |
| T14 Hermes | Installed provider integration, registry handling, Windows/WSL bridge behavior, and entry-point discovery are implemented and tested. | Installed bridge checks pass. A native model turn has not passed. |
| T15 UI | Project enable/pause, readiness, facts/history, correction/forget, captured-copy review, job state, and routing review are exposed and covered by API/browser checks. | Local UI and control-plane checks pass. Operational pilot and native-host coverage remain open. |
| T16 Recovery | Backup, integrity, new-path restore, suppression merge, staged reindex, and activation backup behavior are implemented and tested. | Local recovery contracts pass, including a backup during an active writer transaction. Live data migration remains separately authorization-bound. |
| T17 Cross-host | A shared service contract covers Claude, Codex, and Hermes contexts. Native Codex taught a fact that Claude retrieved, corrected, and deleted across service restarts. Direct service replay suppression and foreign-project exclusion were observed. | V6 is FAILED overall: fresh Codex reused an unknown captured copy after selected-fact deletion and omitted its second query. Broader native related-forget, stale-context consistency, complete capture/shutdown, Hermes, and the two-week pilot remain open. |
| T18 Evaluation | Frozen lifecycle datasets, request-bound extraction schemas, real-provider adapters, seven profiles, raw traces, checkpoints, Wilson intervals, and paired comparisons are implemented. | Power-interrupted checkpoints were preserved and the unchanged queue resumed after provenance checks. Complete real-model matrix acceptance remains open. Follow the [evaluation status and recovery record](evaluation-running.md). |
| T19 Claims and packaging | Claim ledger, compatibility matrix, release checklist, package artifacts, UI assets, and negative evidence archives are maintained. | Package verification is complete for the stated checkpoint. Publication, website, paper, and release actions remain separately authorization-bound. |

The current capture recommendation is a hybrid: automatically capture enabled
project conversations as retained evidence, extract durable facts separately,
and keep explicit remember/correct/forget available. Capture acknowledgments
must not be presented as accepted-fact acknowledgments. Automatic capture
improves continuity but needs retention, privacy controls, and extraction
evaluation; explicit-only storage is easier to inspect but misses information
an agent does not select. See the [operating guide](../../functional-local-memory.md).

Historical evidence remains available for review without changing the current
state above:

- [Claim and evidence ledger](claim-evidence-ledger.md)
- [Client compatibility matrix](../../client-compatibility-matrix.md)
- [Native plugin checkpoint](native-plugin-checkpoint-v1/README.md)
- [Atomic schema verification](atomic-schema-verification/README.md)
- [Related forget verification](related-forget-verification/README.md)
- [Corrected performance report](performance-corrected-v4/README.md)
- [Progress record, including negative measurements](progress.md)

The package and protocol checks do not promote unrun native, real-model,
latency, pilot, or human-acceptance gates. No live host configuration or live
memory database was published or migrated by this work. Publication remains a
separately authorization-bound action.
