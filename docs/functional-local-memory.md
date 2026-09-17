# Functional local memory

This document describes the local runtime as it exists in the functional-local-memory worktree on September 16, 2026. It is an implementation guide and an errata record. A passing deterministic test is evidence for a mechanism. It is not evidence that a real host, real model, or long-running installation has passed.

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

Explicit `remember` bypasses the surprisal decision, but it still passes the safety firewall and scope checks. Maintenance claims a leased job, runs model work outside the write transaction, and commits accepted facts, evidence, revisions, and job completion atomically. `mnemos_job_status` exposes the durable job state without exposing lease credentials or raw payloads.

After resolving an outage or configuration failure, use `mnemos jobs --status exhausted` to list affected jobs, `mnemos job-status <job-id>` to inspect one, and `mnemos retry-job <job-id>` to requeue it. The UI exposes the same operation under **Recover failed processing**. Listing and retry require the current profile, project, and workspace to match. Only exhausted jobs can be reset; a running or completed job cannot be replayed with this command. Requeueing does not mean extraction succeeded. Maintenance processes the pending work and preserves the normal evidence, correction, and forget checks.

Native Claude and Codex hooks remain nonblocking on failure. They emit a sanitized diagnostic to stderr with the host, known event, status, and exception type. The diagnostic excludes captured content and exception messages; normal host context continues to use the host's stdout protocol.

Retrieval returns versioned facts with their source references, support status, and score breakdown. The default retrieval profile is `baseline`. `biomimetic` enables the experimental state and graph contributions through the same scoped candidate set. The lexical search path contributes additional candidates to the union; it does not invent an FTS score that is combined with semantic similarity.

Corrections use a stable fact ID and an expected version:

```text
mnemos_correct(fact_id, content, expected_version)
mnemos_forget(fact_id, expected_version)
```

A stale expected version is rejected. Forget records a suppression identity so replay of an old event cannot recreate the fact. Backups can still contain historical content and are handled separately; this is logical deletion from the live store, not a claim of forensic deletion from every copy or transcript.

## What the five modules currently mean

These are the implemented mechanism contracts. Their neuroscience analogies are design inspiration, not evidence of human-like memory.

1. Surprisal gating computes a score `s` from prediction and input embeddings. Automatic encoding accepts `s > tau`; salience is `min(max(s, 0), 1)`. Explicit `remember` skips this decision.
2. Affective routing measures state match when both states are available. Its intended score is `0.7 * semantic_similarity + 0.3 * state_match`. If state is unavailable, the response reports that status instead of fabricating a match.
3. Sleep maintenance turns captured episodes into typed fact proposals, validates evidence IDs, and commits accepted facts through durable jobs. A provider failure leaves the job visible for retry.
4. Spreading activation transfers `E_next = E_current * (1 - decay_rate) * edge_weight`. A query stores the greatest energy found for each node from a bounded seed path. Energy is query-local, cycles cannot amplify it, and node visits, neighbors, and hops are bounded. Seeds, intermediate nodes, and edges use the same profile and scope predicate.
5. Mutable RAG uses source-backed, expected-version corrections. Query-only recall updates access metadata and retrieval traces; it does not silently rewrite fact content.

## Backups, restore, and reindex

`mnemos backup <new-archive.zip>` takes a consistent snapshot through the owning service. It uses SQLite's backup API, includes the scope registry and a hashed manifest, and runs integrity checks. `mnemos restore <archive.zip> <new-database.sqlite>` restores to a new destination and merges current suppression records so an old snapshot does not silently resurrect a forgotten fact. Restore does not activate the new database or replace the registry. Keep the original until the restored copy passes integrity and identity checks.

Reindex requires exclusive ownership of the stopped runtime. Stop clients and run `mnemos service-stop`, then save the intended embedding configuration. `mnemos reindex --batch-size 100 --max-batches 10` stages or resumes a generation while preserving the active vectors. Once all current facts have embeddings, `mnemos reindex --activate` creates a private backup and activates the complete generation. Incomplete or stale generations cannot activate. Provider identity includes provider, model, explicit revision, dimension, and normalization; set `MNEMOS_EMBEDDING_REVISION` when pinning a mutable model tag. The next runtime startup rebuilds incompatible derived graph edges in bounded batches.

Legacy migrate-store and repair-db copying is restricted to legacy stores and a new target path while the owning service is stopped. Managed stores must use backup/restore so evidence, jobs, revisions, and forget suppression records survive the move.

## Privacy and retention

The shared write firewall screens event identity, content, source references, feedback, and outbox payloads. Secret and PII handling is controlled by the `MNEMOS_MEMORY_SECRET_ACTION` and `MNEMOS_MEMORY_PII_ACTION` settings. Runtime descriptors, locks, databases, registries, checkpoints, and outboxes are intended to be user-private.

Automatic capture creates raw evidence first. Maintenance defaults to a 30-day retention window for processed raw events. Failed or pending source events remain available for recovery. Accepted facts keep source identity and a redacted supporting excerpt; they have no automatic expiration by default. Inspect and retrieve report an original source as `retained_locally` only while its episodic row remains in SQLite, `unavailable_locally` when that row has expired or is otherwise absent, and `no_source_evidence` when no source event was linked. These statuses do not inspect or make claims about external host transcripts. Forget removes live fact content and records suppression identities, including protection against extraction already in flight. Backups and external host transcripts require separate retention decisions.

## Verification boundary

The implementation record in [validation progress](validation/2026-09-16/progress.md) is the source for the current test snapshot. The recorded Ollama checks used a reachable local endpoint and selected models, including `gemma3:4b` and `nomic-embed-text:latest`. They prove provider reachability and one synthetic service flow. They do not prove extraction precision, held-out retrieval quality, or daily-use reliability.

Current evidence has these boundaries:

| Evidence | Status |
| --- | --- |
| Deterministic module and storage tests | 508 passed, 2 optional WSL checks skipped in the broad run; both WSL checks passed when explicitly enabled; subsequent benchmark artifact regression suite: 29 passed |
| Synthetic local service and SQLite restart flow | Observed for one provider-backed fact |
| Full real-model ablation and held-out quality evaluation | RUNNING from frozen commit `5e852e7`; no completed acceptance result yet; see [run record](validation/2026-09-16/evaluation-running.md) |
| Installed Claude Code, Codex, and Hermes host smoke | Adapter/service contracts pass; actual WSL bridge and installed Hermes API observed; complete native conversation matrix NOT RUN |
| Two-week pilot with external users or projects | NOT RUN |
| 10,000-fact latency and growth acceptance | Real query embeddings with synthetic facts: recall 506.111 ms p95 fails the 500 ms gate; durable capture 68.228 ms p95 passes the 100 ms gate; database 324,862,864 bytes |
| Published release acceptance | NOT RUN |

CLI, MCP, UI, backup, and reindex paths have isolated integration checks. Remaining acceptance work includes complete native-host continuity, held-out real-model quality, the failed recall latency gate, and the two-week pilot. Baseline remains the default until measured results justify a different choice.
