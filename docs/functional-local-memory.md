# Functional local memory

This document describes the local runtime in the functional-local-memory worktree, updated September 17, 2026. It is an implementation guide and an errata record. A passing deterministic test is evidence for a mechanism. It is not evidence that a real host, real model, or long-running installation has passed.

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

Most clients auto-start one bounded service process when the descriptor is absent. Set `MNEMOS_RUNTIME_DESCRIPTOR` consistently in every process when isolating a separate runtime. If startup fails, automatic capture can enter a sanitized local outbox; explicit writes report failure. An outbox entry is not a committed service event. `mnemos service-stop` requests authenticated shutdown and waits for confirmation. Stop host clients before offline maintenance so a new request does not restart the service.

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

## Capture, processing, and facts

The default combines automatic evidence capture in enabled projects with explicit remember/correct/forget. Automatic capture preserves context an agent might otherwise miss, but creates more private source data and depends on extraction quality. Explicit saves are easier to review and explain, but the host can forget to make the call and omit useful context. Keeping short-lived evidence separate from durable, source-backed facts gives automatic capture a recovery path while preserving deliberate control over accepted memory.

Automatic capture is available through Claude Code and Codex command-hook adapters and the optional Hermes memory provider. Installed-host acceptance is recorded separately in the [host matrix](client-compatibility-matrix.md). Capture accepts a sanitized, role-tagged event and durable source identity. It returns `committed` when that event and its job are durable, while `pending` or `processing` describes fact processing. The event acknowledgment does not mean a durable fact already exists.

When a host event cannot resolve a profile or project, the adapter records only a bounded routing diagnostic: host, hook event, fixed reason, timestamp, and hashed identity hints. It does not save conversation content, create a replay backlog, or recover the failed event. After fixing profile or project settings, start a new capture. Use `mnemos routing-review` with a configured `MNEMOS_USER_PROFILE_ID` to inspect that profile's diagnostics, then `mnemos routing-review-ack <record-id>` to dismiss one. `--unconfigured` is an explicit local-owner view of records created without a profile; it is never implied by a missing configuration. The UI exposes the same metadata-only review and requires CSRF for dismissal.

Routing diagnostics keep at most 256 records in the private runtime directory.
`MNEMOS_ROUTING_REVIEW_DIR` selects a different local directory when needed.
Oldest diagnostics may be evicted as new failures arrive; this bounded list
is not a complete capture history. Queue write failures remain nonblocking
for the host.

Explicit `remember` bypasses the surprisal decision, but it still passes the safety firewall and scope checks. Maintenance claims a leased job, runs model work outside the write transaction, and commits accepted facts, evidence, revisions, and job completion atomically. `mnemos_job_status` exposes the durable job state without exposing lease credentials or raw payloads.

After resolving an outage or configuration failure, use `mnemos jobs --status exhausted` to list affected jobs, `mnemos job-status <job-id>` to inspect one, and `mnemos retry-job <job-id>` to requeue it. The UI exposes the same operation under **Recover failed processing**. Listing and retry require the current profile, project, and workspace to match. Only exhausted jobs can be reset; a running or completed job cannot be replayed with this command. Requeueing does not mean extraction succeeded. Maintenance processes the pending work and preserves the normal evidence, correction, and forget checks.

Native Claude and Codex hooks remain nonblocking on failure. They emit a sanitized diagnostic to stderr with the host, known event, status, and exception type. The diagnostic excludes captured content and exception messages; normal host context continues to use the host's stdout protocol.

Codex CLI 0.146 supports inline hooks in `config.toml` and a separate `hooks.json` file. Mnemos generates inline hooks. Its installer preserves an existing `[features].hooks = false`, which disables automatic capture even when the hook commands are present. Enable that feature intentionally in Codex before expecting automatic capture; the MCP integration remains a separate path. Avoid copying the generated hooks into both files, since Codex loads both representations. These details are specific to the [tagged 0.146 hook discovery](https://raw.githubusercontent.com/openai/codex/rust-v0.146.0/codex-rs/hooks/src/engine/discovery.rs).

Claude scope identity upgrades preserve old forget suppressions and reject unbound legacy checkpoints. Redaction and missing stable native event IDs can also leave capture degraded. See the [scope repair and recovery limits](validation/2026-09-16/claude-scope-repair.md) before continuing an older session across an upgrade.

Retrieval returns versioned facts with their source references, support status, and score breakdown. The default retrieval profile is `baseline`. `biomimetic` enables the experimental state and graph contributions through the same scoped candidate set. The lexical search path contributes additional candidates to the union; it does not invent an FTS score that is combined with semantic similarity.

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
assertion within the same project and profile; it does not identify paraphrases.
The [verification record](validation/2026-09-16/related-forget-verification/README.md)
separates passing focused/browser/package checks from pending native
acceptance.
All five module switches default to enabled; the baseline retrieval profile
is a separate ranking setting.

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

Reindex requires exclusive ownership of the stopped runtime. Stop clients and run `mnemos service-stop`, then save the intended embedding configuration. `mnemos reindex --batch-size 100 --max-batches 10` stages or resumes a generation while preserving the active vectors. Once all current facts have embeddings, `mnemos reindex --activate` creates a private backup and activates the complete generation. Incomplete or stale generations cannot activate. Provider identity includes provider, model, explicit revision, dimension, and normalization; set `MNEMOS_EMBEDDING_REVISION` when pinning a mutable model tag. The next runtime startup rebuilds incompatible derived graph edges in bounded batches.

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

The implementation record in [validation progress](validation/2026-09-16/progress.md) is the source for the current test snapshot. The recorded Ollama checks used a reachable local endpoint and selected models, including `gemma3:4b` and `nomic-embed-text:latest`. They prove provider reachability and one synthetic service flow. They do not prove extraction precision, held-out retrieval quality, or daily-use reliability.

Current evidence has these boundaries:

| Evidence | Status |
| --- | --- |
| Deterministic module and storage tests | Source c805cc2: 738 tests passed with two optional skips, plus formatting, typing, JavaScript, build, and package checks. Its wheel passed Claude replay and instrumented adapter timing, Hermes entry-point discovery, and generated Codex MCP transport preflight. Native Codex calls were cancelled by the client, so full native lifecycle acceptance remains open. See [package and host verification](validation/2026-09-16/related-forget-verification/README.md) |
| Synthetic local service and SQLite restart flow | Observed for one provider-backed fact |
| Full real-model ablation and held-out quality evaluation | Failed v2 observations preserved; repaired real-model lifecycle preflight passed four cases. Frozen b7 held-out mechanism phase completed: graph correctness passed, graph coverage failed with zero eligible cases, synthetic-state affective top-1 was 0.33 versus 0.00, and classifier fidelity passed 154/300 observations. Seven-profile lifecycle evaluation is running. See [mechanism results and limits](validation/2026-09-16/real-model-v3/mechanism-heldout/REVIEW.md) |
| Installed Claude Code, Codex, and Hermes host smoke | Installed 20f Claude and Codex returned random canaries and captured both roles across restart using constant fixture vectors. Claude reported degraded transcript/identity coverage; Codex used generated TOML hooks. Actual WSL bridge and installed Hermes API observed; complete native lifecycle matrix and Desktop checks remain open |
| Two-week pilot with external users or projects | NOT RUN |
| 10,000-fact latency and growth acceptance | Frozen b7, real query embeddings with synthetic facts: recall 1,314.583 ms p95 fails the 500 ms gate; durable capture 57.410 ms p95 passes the 100 ms gate; database 324,862,864 bytes. Earlier failed and passing timings remain preserved |
| Published release acceptance | NOT RUN |

CLI, MCP, UI, backup, and reindex paths have isolated integration checks. Remaining acceptance work includes complete native-host continuity, held-out real-model quality, the failed recall latency gate, and the two-week pilot. Baseline remains the default until measured results justify a different choice.
