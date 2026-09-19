# Functional local memory

This document describes the local runtime in the functional-local-memory worktree, updated September 19, 2026. It is an implementation guide and an errata record. A passing deterministic test is evidence for a mechanism. It is not evidence that a real host, real model, or long-running installation has passed.

## Start with the local service

This work is unreleased. Install this checkout with optional MCP dependencies and open the local control plane:

```bash
python -m pip install -e ".[mcp]"
mnemos ui
```

Choose the Starter SQLite profile, review the database path, and save the canonical configuration. Enable the repository explicitly with a stable local profile identity:

```bash
mnemos enable-project --path . --profile anthony-local
mnemos stats
```

Set `MNEMOS_USER_PROFILE_ID=anthony-local` in each client environment. The project registry resolves the canonical Git common directory, or the canonical non-Git root. A folder basename is not an identity. A project must be enabled for the selected user profile before scoped capture or recall is accepted.

The CLI, MCP server, and host adapters use the authenticated loopback service. The service owns one SQLite writer, one asyncio loop, model instances, and maintenance. An MCP stdio process is a client of that service. It does not open the live database as a competing writer. To run the service manually for debugging:

```bash
python -m mnemos.service
```

Most clients auto-start one bounded service process when the descriptor is absent. Set `MNEMOS_RUNTIME_DESCRIPTOR` consistently in every process when isolating a separate runtime. Automatic capture persists a sanitized event in the local outbox before waiting for startup or an HTTP response. A committed or explicitly rejected response removes it; interrupted requests retain the same event identity for replay. An outbox entry is not a committed service event. Explicit writes report failure. `mnemos service-stop` requests authenticated shutdown and waits for confirmation. Stop host clients before offline maintenance so a new request does not restart the service.

This protects capture after the outbox write completes. It cannot recover text
if the host terminates the hook before that write, and it does not establish
that native hook deadlines or clean host shutdown are satisfied.

After a transport failure, the client checks the existing owner through a
bounded authenticated health request. A reachable owner receives one retry
without another launch. Otherwise, the client waits for a replacement
descriptor while preserving the stale file. A competing launcher exit can
still yield to a recovered owner after another bounded health check.
Authorization and invalid health responses do not trigger replacement.

The client reports a service process that exits before startup with its exit
code. A process that remains alive without publishing a descriptor reports a
startup timeout. A valid existing owner takes precedence over an exited
launcher. These diagnostics preserve the same capture outbox and bounded
autostart policy; a timeout does not terminate the shared service.

The managed service writes private JSON events to `service.log` beside its
runtime descriptor. The log rotates at 1 MiB and retains two backups. It records
startup, retrieval timing, and failure categories without query text, memory
content, credentials, paths, or exception messages. A startup event means the
service is running; it does not establish that a configured model is reachable
or ready. Embedded callers with an injected dispatcher can opt in with
`service_log_path`; one file logging binding is supported per process.

Schema 8 stores access counts, retrieval timestamps, and reconsolidation due
times in `memory_chunk_access`. Retrieval updates this small table with the
existing WAL/FULL durability policy. Opening an older database first creates
a pre-migration backup, then seeds these fields in one migration transaction.
The old columns in `memory_chunks` remain compatibility seeds; use the storage
API to read current access metadata. Migration and rollback checks use
disposable databases. The separate 10,000-fact recall gate still needs a new
measurement before this change can claim a latency improvement.

## Local model configuration

Mock cognition and simple embeddings are useful for deterministic tests. They are not evidence of retrieval quality. For a local model profile, configure Ollama explicitly and select both models:

```bash
MNEMOS_LLM_PROVIDER=ollama
MNEMOS_LLM_MODEL=gemma3:4b
MNEMOS_EMBEDDING_PROVIDER=ollama
MNEMOS_EMBEDDING_MODEL=nomic-embed-text:latest
MNEMOS_OLLAMA_URL=http://localhost:11434
MNEMOS_STORE_TYPE=sqlite
MNEMOS_SQLITE_PATH=.mnemos/memory.db
```

These are environment entries; use your shell's environment syntax or save equivalent values in the canonical configuration. Prefer absolute database paths. Use the endpoint reachable from the service process. Mnemos has no cloud fallback when a local model fails. Durable capture can continue while enrichment waits. Semantic recall still requires the configured embedding provider. Saving configuration requires a service restart before the new providers apply.

Ollama embeddings reuse one HTTP client per provider. The managed service
drains pending operations before closing its engine, provider clients, and
storage. Library callers should stop their engine operations and then call
`await engine.aclose()`. Engines close providers they create themselves;
caller-supplied providers remain caller-owned unless `owns_providers=True`
is passed. A directly constructed embedding provider should be closed with
`provider.close()` after use. Closing waits for active embedding requests;
an explicitly closed provider cannot start another request. Client reuse
does not establish that the recall or capture latency targets have passed.

Ollama cognition caches its default certificate trust context after the first
request. Each request still owns and closes its HTTP client, so a provider can
be used across separate event loops without retaining an old loop's transport.
Certificate and hostname verification remain enabled. Nonempty `SSL_CERT_FILE`
or `SSL_CERT_DIR` overrides bypass this cache and retain HTTPX's per-request
loading. Recreate the provider or restart the service after changing its default
CA bundle. Proxy environment settings, request timeouts, retries, and payloads
remain unchanged. This reduces local setup work; it does not measure model
inference or establish the recall latency target. HTTPX documents these
[trust-context and environment controls](https://www.python-httpx.org/advanced/ssl/).

For a thinking-capable Ollama model, the canonical TOML configuration can also
select a thinking mode explicitly:

```toml
[providers.ollama]
think = false
```

Omitting `think` preserves the model's default. Mnemos accepts TOML booleans
and the strings `"low"`, `"medium"`, `"high"`, and `"max"`; support depends on
the selected model. There is no corresponding environment override. Ollama
returns reasoning separately from its final response. Mnemos uses only the
final response and reports a provider error when it is empty or not text.
See [Ollama thinking configuration](https://docs.ollama.com/capabilities/thinking).

In a four-case development probe, `qwen3.5:9b` with `think = false` returned
four runtime-valid extraction responses through the provider. Three matched
the predefined exact proxy; the model omitted a stated editor preference.
This establishes a working provider option, not general extraction accuracy.
[Probe and limitations](validation/2026-09-16/ollama-thinking-provider-integration/README.md).

Embedding task prefixes can be configured explicitly in canonical TOML:

```toml
[embedding]
query_prefix = "search_query: "
document_prefix = "search_document: "
symmetric_prefix = "clustering: "
```

Use the task contract documented by the selected embedding model; these values
are not automatically inferred from its name. Retrieval uses the query prefix,
stored facts and reindexing use the document prefix, and prediction/evidence
comparisons use the symmetric prefix. Prefixes default to empty strings for
compatibility and have no environment overrides. Changing any prefix changes
the persistent embedding identity and requires a new index generation before
activation. Do not point a new query configuration at an unchanged old index.
Deterministic tests cover the payloads and migration boundary; the quality
effect of prefixes on the selected local model remains unmeasured.
[Implementation and evaluation boundary](validation/2026-09-16/embedding-task-contract.md).

## Scope and host identity

The runtime resolves a `ScopeContext` from the persisted registry and process environment:

```text
MNEMOS_USER_PROFILE_ID   required registered local profile
MNEMOS_PROJECT_ROOT      project path, otherwise the process cwd
MNEMOS_SCOPE_REGISTRY    optional explicit registry path
MNEMOS_WORKSPACE_ID      optional registered workspace membership
MNEMOS_SESSION_ID        optional stable host session, otherwise a process UUID
MNEMOS_HOST              codex, claude-code, hermes, or manual; manual is the default
```

An unknown profile, disabled project, invalid workspace membership, or unknown host fails closed. An explicit MCP `scope_id` must match the resolved project or workspace. `global` is shared within the local user profile and requires an explicit scope choice. Legacy unscoped rows remain available for migration review and are excluded from automatic scoped recall.

The UI saves the selected profile in its global onboarding settings and includes the profile and registry path in generated host configuration. An explicit `MNEMOS_USER_PROFILE_ID` overrides that selection. Claude Code setup writes project `.mcp.json`; automatic conversation capture also requires the Mnemos Claude plugin. Applying MCP configuration alone does not install its capture hooks.

The Claude plugin and newly generated Claude MCP entries request
`alwaysLoad: true` for Mnemos. Claude then waits for this server's startup and
loads its tool definitions into the first prompt's context. Existing explicit
`alwaysLoad: false` choices in generated configuration are preserved. This
trades initial context space and a bounded startup wait for first-turn tool
availability. The isolated Claude Code 2.1.191
[startup control](validation/2026-09-17/native-claude-alwaysload-v2/README.md)
observed a connected catalog and successful health call with global nonblocking
startup enabled. It does not establish full capture acceptance. See the
[Claude server-loading contract](https://code.claude.com/docs/en/mcp#exempt-a-server-from-deferral).

## Capture, processing, and facts

The default combines automatic evidence capture in enabled projects with explicit remember/correct/forget. Automatic capture preserves context an agent might otherwise miss, but creates more private source data and depends on extraction quality. Explicit saves are easier to review and explain, but the host can forget to make the call and omit useful context. Keeping short-lived evidence separate from durable, source-backed facts gives automatic capture a recovery path while preserving deliberate control over accepted memory.

Automatic capture is available through Claude Code and Codex command-hook adapters and the optional Hermes memory provider. Installed-host acceptance is recorded separately in the [host matrix](client-compatibility-matrix.md). Capture accepts a sanitized, role-tagged event and durable source identity. It returns `committed` when that event and its job are durable, while `pending` or `processing` describes fact processing. The event acknowledgment does not mean a durable fact already exists.

A `queued` capture has a sanitized local outbox copy awaiting confirmation. The service may already have committed the event when its reply was lost. Replay keeps the same event identity and removes the outbox entry only after a committed response, including across a service restart. The [lost-acknowledgment regression](validation/2026-09-17/service-lost-ack-regression-v1/README.md) checks this path with ordinary and provisional captures and repeated reply loss.

When a host event cannot resolve a profile or project, the adapter records only a bounded routing diagnostic: host, hook event, fixed reason, timestamp, and hashed identity hints. It does not save conversation content, create a replay backlog, or recover the failed event. After fixing profile or project settings, start a new capture. Use `mnemos routing-review` with a configured `MNEMOS_USER_PROFILE_ID` to inspect that profile's diagnostics, then `mnemos routing-review-ack <record-id>` to dismiss one. `--unconfigured` is an explicit local-owner view of records created without a profile; it is never implied by a missing configuration. The UI exposes the same metadata-only review and requires CSRF for dismissal.

Routing diagnostics keep at most 256 records in the private runtime directory.
`MNEMOS_ROUTING_REVIEW_DIR` selects a different local directory when needed.
Oldest diagnostics may be evicted as new failures arrive; this bounded list
is not a complete capture history. Queue write failures remain nonblocking
for the host.

Explicit `remember` bypasses the surprisal decision, but it still passes the safety firewall and scope checks. Maintenance claims a leased job, runs model work outside the write transaction, and commits accepted facts, evidence, revisions, and job completion atomically. `mnemos_job_status` exposes the durable job state without exposing lease credentials or raw payloads.

When Claude supplies a prompt without a stable occurrence ID, its fallback capture is `provisional_unassociated`. The service acknowledges the sanitized raw event and a `capture_only` job atomically. Maintenance completes that job with `identity_pending`; the raw event remains `deferred` and is excluded from extraction and fact indexing. A later transcript entry with a native ID is captured separately. Mnemos does not merge these records by matching text. This preserves the raw acknowledgement without extracting the fallback as a second fact. Completed deferred events follow processed-event retention, and replay after expiration does not recreate them.

The default identity status, `normal`, preserves existing capture behavior; it is not proof of a native ID. Explicit saves remain extraction-eligible. Identical prompts without occurrence IDs still cannot be distinguished reliably from retries, so capture remains degraded in that case. The earlier native Claude test's exact-two-raw-events gate remains failed; the new deterministic checks establish extraction exclusion and retry safety, not completed native acceptance.

After resolving an outage or configuration failure, use `mnemos jobs --status exhausted` to list affected jobs, `mnemos job-status <job-id>` to inspect one, and `mnemos retry-job <job-id>` to requeue it. The UI exposes the same operation under **Recover failed processing**. Listing and retry require the current profile, project, and workspace to match. Only exhausted jobs can be reset; a running or completed job cannot be replayed with this command. Requeueing does not mean extraction succeeded. Maintenance processes the pending work and preserves the normal evidence, correction, and forget checks.

Native Claude and Codex hooks remain nonblocking on failure. They emit a sanitized diagnostic to stderr with the host, known event, status, and exception type. The diagnostic excludes captured content and exception messages; normal host context continues to use the host's stdout protocol.

Claude degraded diagnostics also include an allowlisted reason when available.
`identity_reference_unavailable` means the hook lacked a stable occurrence ID;
`ambiguous_transcript_association` means a fallback could not be linked safely
to a later transcript entry. These warnings can coexist with captured text.
Unknown error strings become `reason=unknown`, without their original content.
The warning does not acknowledge an identity association or complete capture.

Codex CLI 0.146 supports inline hooks in `config.toml` and a separate `hooks.json` file. Mnemos generates inline hooks. Its installer preserves an existing `[features].hooks = false`, which disables automatic capture even when the hook commands are present. Enable that feature intentionally in Codex before expecting automatic capture; the MCP integration remains a separate path. Avoid copying the generated hooks into both files, since Codex loads both representations. These details are specific to the [tagged 0.146 hook discovery](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/discovery.rs).

MCP tools now declare their effects so hosts can make informed approval decisions. Health, statistics, listing, inspection, job status, and forget preview are read-only. Retrieval updates durable access metadata and can call a configured provider, so it is marked as a write. Correction, forget, and maintenance retain destructive-operation hints. These hints do not override host policy or service scope checks. In Codex CLI 0.146, missing annotations can request approval and a cancelled approval returns `user cancelled MCP tool call` before server dispatch; see the [tagged approval implementation](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/core/src/mcp_tool_call.rs). A successful annotated recorder probe establishes that metadata affects dispatch, not that the complete Mnemos lifecycle has passed.

Claude scope identity upgrades preserve old forget suppressions and reject unbound legacy checkpoints. Redaction and missing stable native event IDs can also leave capture degraded. See the [scope repair and recovery limits](validation/2026-09-16/claude-scope-repair.md) before continuing an older session across an upgrade.

Retrieval returns versioned facts with their source references, support status, and score breakdown. The default retrieval profile is `baseline`. `biomimetic` enables the experimental state and graph contributions through the same scoped candidate set. The lexical search path contributes additional candidates to the union; it does not invent an FTS score that is combined with semantic similarity.

Retrieved facts also expose `assertion_status`: `asserted`, `conflict`, or
`unknown` when no recognized status was stored. A conflicting assertion is
labelled `unresolved conflict` in the context sent to hosts, and that label
counts toward the context budget. `asserted` describes the stored claim;
it is not an independent truth judgment. Source IDs establish provenance,
but a model can still attach a valid source ID to an unsupported statement.
Inspect the cited source before relying on a disputed or consequential fact.

When sleep consolidation is disabled, automatic capture remains searchable
as verbatim conversation evidence. Newly indexed copies use source
`captured_passthrough` and assertion status `unknown`. Host context labels them
`captured conversation; unverified, may be outdated`; other unknown records
are labeled `unverified`. These labels count toward the context budget. A
correction changes the selected fact, not every independent historical mention.
Explicit remember and correct retain their assertion status even when an
identical automatic capture is merged later. Captured copies remain available
for inspection and deliberate related-forget review. This does not relabel
previously indexed records or establish the truth of extracted assertions.

Corrections use a stable fact ID and an expected version:

```text
mnemos_correct(fact_id, content, expected_version)
mnemos_forget(fact_id, expected_version)
```

A stale expected version is rejected. Forget records a suppression identity so replay of an old event cannot recreate the fact. Backups can still contain historical content and are handled separately; this is logical deletion from the live store, not a claim of forensic deletion from every copy or transcript.

Single-record forget can leave an independently captured copy of the same
assertion. The [preserved reproduction](validation/2026-09-16/capture-forget-overlap/v2/README.md)
demonstrates this when sleep extraction is disabled and captured text is
promoted verbatim. The unreleased implementation now provides
`mnemos_forget_preview` and `mnemos_forget_related`, also available through
**Review captured copies** in the memory UI. Review the target, extra captured
facts, and their sources; select the extra records to remove, then apply that
preview. A changed or incomplete preview requires another review. Shared
unselected dependencies prevent deletion, including references in queued jobs
after source text expires. Matching uses the full current or historical
assertion within the same project and profile, then adds up to twenty bounded
same-scope similarity candidates from captured or sleep-consolidated facts when
the current target has a stored document vector. These candidates are advisory,
are never selected automatically, and are not an exhaustive semantic erasure
plan. Missing vectors, raw episodic paraphrases, historical-only meanings, and
external transcripts may still be missed.
The [similarity source checks](validation/2026-09-18/related-forget-similarity-v1/README.md)
cover explicit selection, source dependencies, stale previews, and replay
suppression with controlled vectors. Native acceptance remains open.
The [verification record](validation/2026-09-16/related-forget-verification/README.md)
separates passing focused/browser/package checks from pending native
acceptance.
All five module switches default to enabled; the baseline retrieval profile
is a separate ranking setting.

The [native v6 attempt](validation/2026-09-17/native-codex-claude-continuity-v6/README.md)
demonstrates the practical limit: selected-fact deletion and replay suppression
worked, but fresh Codex retrieval returned a separate captured assistant statement
and the host presented its value as current. It also skipped a requested query.
The result remains failed. The current historical/unverified labels did not
prevent that answer; native related-copy deletion and context handling still
need acceptance evidence.

The surviving v6 assistant paraphrase does not contain either the target's full
current assertion or its historical assertion after the existing normalization.
The exact-text matcher used at that revision would therefore miss that copy. This was
checked against the archived native result; it is not a new native preview run.
The new similarity suggestions address indexed paraphrase discovery, while native
review-and-delete acceptance remains open. Selected-fact deletion does not
automatically delete other semantically similar records.

The later [native v7 attempt](validation/2026-09-18/native-codex-claude-continuity-v7/README.md)
committed related deletion and retained unrelated facts, but exceeded Claude's
phase budget before replay and final recall. The
[v8 attempt](validation/2026-09-18/native-codex-claude-continuity-v8/README.md)
timed out during reverse Codex tool approval before the forget phase. These
incomplete attempts do not establish reliable post-forget answers.

The current source also checks exact forgotten assertions before automatic
extraction adds role attribution, and repeats that check at atomic commit using
the stored event role. This prevents an `Assistant reported:` prefix from
bypassing the original fact's suppression. Explicit reteaching remains allowed;
non-identical paraphrases are outside this exact-match guarantee. The
[source checkpoint](validation/2026-09-18/assistant-assertion-suppression-v1/README.md)
passes 1,030 tests with four skips and static checks. The installed package
checkpoint at `5af3d15` also passes build, Twine, 360 selected contract cases
with two matching skips, and seven synthetic replay assertions.

The [native v9 run](validation/2026-09-18/native-codex-claude-continuity-v9/README.md)
uses that installed package and passes 15 bounded lifecycle checks. Codex taught
a fact, Claude corrected it, and fresh Codex retrieved the current version after
restart. Claude reviewed and deleted the target plus five captured copies,
including similarity suggestions. Direct replay after restart did not restore
the target. Final Codex did not retrieve or repeat either convention value and
retrieved the unrelated retained preference. All three Codex processes exited
zero, but reverse and final Stop hooks timed out after three seconds. The fixture
used lexical vectors and disabled modules. Complete capture, native hook replay,
Hermes, Desktop, and real-model quality remain open.

The newer [Windows permission checkpoint](validation/2026-09-18/windows-private-acl-package-checkpoint-v1/README.md)
at `5d8cacf` passes 1,039 source tests with four skips, all package checks, 369
installed cases with two matching skips, and seven replay assertions. It repairs
permission inheritance on private runtime files and avoids rewriting an unchanged
protected permission list. Nine Windows tests verify the policy and later drift.
Its [native Stop probe](validation/2026-09-18/native-codex-private-acl-profile-v3/README.md)
completed with protected fixture permissions and persisted both conversation roles.
The instrumented shell duration was 876 ms. This single observation does not
establish repeated deadline acceptance or close the broader lifecycle gates.

The newer [competing service recovery source check](validation/2026-09-18/competing-service-recovery-v1/README.md)
fixes a startup race: a second launcher could return a dead descriptor while
the new lock owner was still initializing. Competing launchers now verify
authenticated health before adopting an owner. Five added cases cover the
race, startup timeout, authentication rejection, and invalid health data.
The full source suite passes 1,054 tests with four skips; 47 focused checks
also pass. Simultaneous native host acceptance remains open; the package
checks below validate the installed service protocol.

The current [competing recovery package checkpoint](validation/2026-09-18/competing-recovery-package-checkpoint-v1/README.md)
at `a5a9101` passes 1,054 source tests with four skips, all six source/package
checks, 384 selected installed cases with two matching skips, and seven
synthetic replay assertions. The installed selection includes the five
competing-launcher regressions. Prior native observations retain their
original wheel bindings; this package does not establish native concurrency
or real-provider recovery.

The preceding [stale descriptor checkpoint](validation/2026-09-18/stale-descriptor-package-checkpoint-v1/README.md)
at `37e446f` passes 1,049 source tests with four skips, all six source/package
checks, 379 selected installed cases with two matching skips, and seven
synthetic replay assertions. Seven new cases cover delayed replacement,
live-owner reuse, failed or stalled startup, authentication rejection, and
an owner recovering while its competing launcher exits. The client probes
authenticated health before launching a replacement and waits past a known
stale descriptor without deleting it. Earlier native evidence remains bound
to its original wheel; configured real-provider recovery remains open.

The preceding [outbox selection checkpoint](validation/2026-09-18/outbox-selection-package-checkpoint-v1/README.md)
at `c25ed98` passes 1,042 source tests with four skips, all six source/package
checks, 372 installed cases with two matching skips, and seven replay assertions.
A supplied empty outbox now keeps its selected recovery file and write policy.
Native observations above retain their original package bindings.

The [storage-worker termination checkpoint](validation/2026-09-18/storage-process-kill-v1/README.md)
adds seven cases at `76940d2`: capture after event insertion, before commit,
and after commit; forget after suppression insertion, after source deletion,
before commit, and after commit. The actual SQLite worker is terminated.
Fresh reopen, 12-table hashes, integrity checks, repeated outbox replay,
derived-state cleanup, and retained local/foreign controls pass. The clean
source suite passes 1,067 tests with four skips, and all seven new cases
pass against the existing `983920a` wheel. Runtime/build Git objects and
installed wheel bytes are verified unchanged. These fixed-provider tests
cover selected storage boundaries; native host, provider, commit-instruction
interruption, and operational-pilot acceptance remain separate.

The [capture/forget transaction checkpoint](validation/2026-09-18/capture-forget-package-checkpoint-v1/README.md)
at `983920a` prevents late capture, replay, and correction from restoring
erased raw sources after a forget commits. SQLite rechecks source IDs,
verified Claude aliases, automatic assertion suppression, and forgotten
correction targets inside its write transaction. Fresh explicit reteaching
remains supported. Six added cases pass in both source and installed runs.
The full source suite passes 1,060 tests with four skips; the installed
wheel passes 390 selected cases with two matching skips and seven replay
assertions. An earlier full run had one degraded bridge result. Its failure
and subsequent passing diagnostics are preserved; the cause remains open.
This checkpoint does not rerun native hosts or change the frozen model evaluation.

A [concurrent native Codex recovery probe](validation/2026-09-18/native-codex-concurrent-recovery-v2/README.md)
on the installed `a5a9101` wheel used two separate Codex homes and one shared
outbox. Two outage turns queued four messages. After explicit fixture-service
restoration, two fresh native processes automatically replayed that queue.
A test barrier overlapped delivery of the same event, and Windows TCP ownership
plus process ancestry tied the requests to both native processes. All eight
messages across four sessions had one event, completed job, and evidence link
each. The outbox emptied, both controls survived, and restart preserved the
database. Generated hooks were unchanged and all four native processes exited
zero. The failed v1 home-path fixture is retained. Recovery-a displayed an MCP
startup interruption, and PowerShell emitted Import-Clixml diagnostics; those
PTY observations remain explicit limitations. This closes one bounded Codex
concurrency observation, while cross-host concurrency, real-provider recovery,
complete native acceptance, and the operational pilot remain open.

A [native startup-outage probe](validation/2026-09-18/native-codex-startup-outage-v1/README.md)
on the installed `c25ed98` wheel retained the real user prompt and final answer
in its outbox while malformed fixture configuration prevented service startup.
The native process exited zero; the original runner failed because the answer
omitted a requested period. That failure is preserved. A separate recovery
helper verified the actual messages, then explicitly started the fixture service
and replayed both events. An immediate second replay had no pending events.
After service restart, replaying the same events left all four core table counts
and complete-row hashes unchanged. Each message had exactly one stored event
and one evidence row. This establishes native outage capture followed by explicit
client recovery with fixture providers. That run did not exercise automatic
native-hook replay; the later two-session observation below does.

In a [fresh two-session probe](validation/2026-09-18/native-codex-native-recovery-v1/README.md),
the first native Codex session queued both messages during startup failure.
After explicit fixture-service restoration, the second session's unchanged
generated hooks drained that queue automatically, without a direct replay call.
Both processes exited zero. All four messages retained their correct session,
turn, role, project, and profile, with one evidence link each. The outbox emptied,
and complete-row hashes stayed unchanged across maintenance and service restart.
This verifies one native automatic replay observation. The exact triggering hook
was not traced; the separate process-kill observation follows below.

A [forced service-termination probe](validation/2026-09-18/native-codex-service-kill-v2/README.md)
then killed the actual disposable service after SQLite committed a native prompt
and before its HTTP response. The event and job survived under their original IDs.
After explicit fixture-service restoration replaced the dead owner's descriptor,
a fresh native Codex session automatically drained the queue. All four native
messages had one job and one evidence link each, with unchanged table hashes
across maintenance and restart. Both native processes exited zero. The first
session's Stop timeout and the second session's MCP startup warning remain
recorded as terminal observations; this is not an all-hooks acceptance result.
Arbitrary crash timing, concurrent hosts, real-provider recovery, and complete
host acceptance remain open. The failed v1 PID preflight is preserved separately.

Three fresh [native Codex sessions](validation/2026-09-18/native-codex-stop-repeat-v1/README.md)
on the `c25ed98` wheel used unchanged generated hooks. Late native hook pairs measured
1,015 ms, 1,031 ms, and 964 ms within the three-second deadline, and both
conversation roles persisted after shutdown. The pairs are identified as Stop
by their placement between final answer and turn completion; native logs omit
the hook name and exit status. These three fixture observations do not establish
tail latency or complete native lifecycle acceptance. Original harness failures
from Codex adding hook-trust metadata are retained with the separate verifier.

## What the five modules currently mean

These are the implemented mechanism contracts. Their neuroscience analogies are design inspiration, not evidence of human-like memory.

1. Surprisal gating computes a score `s` from prediction and input embeddings. Automatic encoding accepts `s > tau`; salience is `min(max(s, 0), 1)`. Explicit `remember` skips this decision.
2. Affective routing measures state match when both states are available. Its intended score is `0.7 * semantic_similarity + 0.3 * state_match`. If state is unavailable, the response reports that status instead of fabricating a match.
3. Sleep maintenance turns captured episodes into typed fact proposals, validates evidence IDs, and commits accepted facts through durable jobs. A provider failure leaves the job visible for retry.
4. Spreading activation transfers `E_next = E_current * (1 - decay_rate) * edge_weight`. A query stores the greatest energy found for each node from a bounded seed path. Energy is query-local, cycles cannot amplify it, and node visits, neighbors, and hops are bounded. Seeds, intermediate nodes, and edges use the same profile and scope predicate.
5. Mutable RAG uses source-backed, expected-version corrections. Query-only recall updates access metadata and retrieval traces; it does not silently rewrite fact content.

Automatic memory links use persisted directed top-k preferences. Two facts get
a traversable link only when each selects the other, with stable ID tie-breaking.
The builder considers all compatible authoritative facts, even when only part
of the graph is loaded for traversal. Completed sources with no neighbors are
also persisted. Correction, forgetting, incompatible versions, and interrupted
rebuilds leave explicit pending state until the affected sources are rebuilt.
Recall falls back to semantic/lexical scoring while the graph is pending; its
trace reports that state. The dashboard displays `Memory links: rebuilding`
or `Memory links: ready` from the running service.

The standalone `connect_node` and `auto_connect` APIs use the same mutual
selection rule. Explicit `add_edge` calls remain available for manual graphs.
Calling `auto_connect` rebuilds derived adjacency and replaces manual links;
the historical `exclude_existing` argument remains accepted but no longer
preserves adjacency across a derived rebuild. Persisted builder version 4
invalidates older derived links while preserving facts and evidence.

## Backups, restore, and reindex

`mnemos backup <new-archive.zip>` takes a consistent snapshot through the owning service. It uses SQLite's backup API, includes the scope registry and a hashed manifest, and runs integrity checks. `mnemos restore <archive.zip> <new-database.sqlite>` restores to a new destination and merges current suppression records so an old snapshot does not silently resurrect a forgotten fact. Restore does not activate the new database or replace the registry. Keep the original until the restored copy passes integrity and identity checks.

Reindex requires exclusive ownership of the stopped runtime. Stop clients and run `mnemos service-stop`, then save the intended embedding configuration. `mnemos reindex --batch-size 100 --max-batches 10` stages or resumes a generation while preserving the active vectors. Once all current facts have embeddings, `mnemos reindex --activate` creates a private backup and activates the complete generation. Incomplete or stale generations cannot activate. Provider identity includes provider, model, explicit revision, dimension, normalization, and any explicit task-prefix contract. `MNEMOS_EMBEDDING_REVISION` is an operator-managed identity label; it does not verify or pin the digest served by a mutable Ollama tag. If that tag changes, update the revision and reindex explicitly. The next runtime startup rebuilds incompatible derived graph edges in bounded batches.

Legacy migrate-store and repair-db copying is restricted to legacy stores and a new target path while the owning service is stopped. Managed stores must use backup/restore so evidence, jobs, revisions, and forget suppression records survive the move.

Use `mnemos legacy-review --limit 50` or **Review legacy unassigned memory**
in the UI to inspect facts that have no profile assignment. The CLI accepts
`--after-id` for subsequent pages. This is a read-only local-owner review;
these facts stay excluded from scoped recall. Offline duplicate repair keeps
different legacy owners and scopes separate and preserves their original
metadata. It does not assign them to the current project or make them global.
Assigning legacy facts still requires a separately reviewed migration.

## Privacy and retention

The shared write firewall screens event identity, content, source references, feedback, and outbox payloads. Secret and PII handling is controlled by the `MNEMOS_MEMORY_SECRET_ACTION` and `MNEMOS_MEMORY_PII_ACTION` settings. Runtime descriptors, locks, databases, registries, checkpoints, and outboxes are intended to be user-private.

Automatic capture creates raw evidence first. Maintenance defaults to a 30-day retention window for processed raw events. Failed or pending source events remain available for recovery. Accepted facts keep source identity and a redacted supporting excerpt; they have no automatic expiration by default. Inspect and retrieve report an original source as `retained_locally` only while its episodic row remains in SQLite, `unavailable_locally` when that row has expired or is otherwise absent, and `no_source_evidence` when no source event was linked. These statuses do not inspect or make claims about external host transcripts. Forget removes live fact content and records suppression identities, including protection against extraction already in flight. Backups and external host transcripts require separate retention decisions.

## Verification boundary

The local service client uses authenticated plaintext HTTP restricted to
loopback, ignores environment proxies, and does not follow redirects. Its
transport skips certificate-bundle loading because these routes do not use
TLS; certificate verification remains required if HTTPS is introduced. If a
competing launcher exits while the original owner recovers, readiness probes
can use the remaining startup deadline without launching another process.

The dated [claim and evidence ledger](validation/2026-09-16/claim-evidence-ledger.md)
maps the original capability promises to current evidence and open acceptance
gates.

Automatic conflict proposals cannot relabel an existing non-conflict fact
through exact canonical deduplication. The storage commit rejects that
collision, preserves the fact and its existing evidence, and rolls back the
whole extraction batch. Maintenance reports `conflict_canonical_collision`;
the source remains pending while the job follows its retry policy. New conflict
content and repeats of an already-conflicted fact remain supported. This is a
conservative collision rule, not a semantic entailment check. A real
contradiction whose proposal repeats the old assertion may need correction or
review instead of automatic merging.

The implementation record in [validation progress](validation/2026-09-16/progress.md) is the source for the current test snapshot. The recorded Ollama checks used a reachable local endpoint and selected models, including `gemma3:4b` and `nomic-embed-text:latest`. They prove provider reachability and one synthetic service flow. They do not prove extraction precision, held-out retrieval quality, or daily-use reliability.

Current evidence has these boundaries:

| Evidence | Status |
| --- | --- |
| Deterministic module and storage tests | Verified source `5e4b057`: 1,195 tests passed with four skips, plus Black, configured strict mypy, JavaScript, build, and Twine checks. The fresh installed wheel passed 507 selected contract cases with two matching platform skips and seven synthetic replay assertions. Installed Hermes API v2 passes scoped lifecycle and transcript/provisional capture with fixture providers. Native deadline acceptance remains open. See the [current package checkpoint](validation/2026-09-19/hermes-lifecycle-checkpoints-v1/README.md) |
| Synthetic local service and SQLite restart flow | Observed for one provider-backed fact |
| Full real-model ablation and held-out quality evaluation | Local `fc72cc2` completed baseline and no-sleep lifecycle profiles; five profiles remain incomplete, with 50 missing correction targets after extraction returned no durable facts. Follow-on encoding/retrieval comparisons did not start. Graph utility and end-to-end affective benefit remain unproved. See the [terminal lifecycle evidence](validation/2026-09-19/lifecycle-incomplete-local-v2/README.md) and [evaluation status](validation/2026-09-16/evaluation-running.md) |
| Installed Claude Code, Codex, and Hermes host smoke | Native v9 on installed `5af3d15` passes 15 bounded Codex/Claude lifecycle checks, including reviewed deletion, restart, direct service replay, and final recall retaining the unrelated control. Lexical fixture vectors and disabled modules do not establish model quality. All Codex processes exited zero; reverse and final Stop hooks timed out, and capture remains partial. Hermes patched CLI v4 on `78c63ed` passes 16 single-turn real-model recall and raw-capture checks in an isolated host checkout. Hermes API v2 on `5e4b057` separately verifies explicit lifecycle tools with fixture providers. A later Codex-only fixture verifies automatic native-hook replay after startup failure. Desktop, complete capture, full native Hermes lifecycle, and operational readiness remain open. See the [v9 record](validation/2026-09-18/native-codex-claude-continuity-v9/README.md) and [host matrix](client-compatibility-matrix.md). |
| Two-week pilot with external users or projects | NOT RUN |
| 10,000-fact baseline latency | Three fresh SSD runs on `de6cdcf` pass the unchanged gates: warm recall p95 245.214-252.222 ms and capture p95 28.804-36.062 ms, with all 150 captures committed. Real query embeddings, synthetic stored vectors/capture provider, zero cognition calls, WAL/FULL. Cold load, semantic quality, full-module latency and native hosts are separate. See [performance evidence](validation/2026-09-19/performance-recovery-v1/README.md) |
| Published release acceptance | NOT RUN |

CLI, MCP, UI, backup, and reindex paths have isolated integration checks. Remaining acceptance work includes complete native-host continuity, held-out real-model quality and module benefits, full-module latency, and the two-week pilot. The baseline SSD latency result does not close those gates. Baseline remains the default until measured results justify a different choice.
