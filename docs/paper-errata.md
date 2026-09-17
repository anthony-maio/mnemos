# Paper errata and reproducibility notes

This note records the differences between the March 2026 paper, the public website, and the functional-local-memory implementation record. The paper remains unchanged at [making-minds.ai/papers/mnemos.pdf](https://making-minds.ai/papers/mnemos.pdf). The website remains unchanged at [mnemos.making-minds.ai](https://mnemos.making-minds.ai).

## Version and release language

The paper describes `v0.2.0`, a production-ready MCP server, and three storage backends including Qdrant. The current worktree declares version `0.7.0`, uses SQLite as the supported local service database, and keeps embedded module APIs available for explicit experiments. The current validation record does not establish production readiness. Installed host continuity, the two-week pilot, and final release acceptance remain NOT RUN.

The website's narrower product language is the safer description: local-first scoped memory for solo coding-agent workflows. Shared MCP subprocess coverage is not native Claude Desktop or Claude Code evidence. The current host matrix separates adapter, installed API, transport, and complete native conversation observations for every host.

## Numerical and algorithmic corrections

### Surprisal angle and range

The paper says that a cosine distance threshold of `0.3` corresponds to approximately 72 degrees. For cosine distance `1 - cos(theta)`, the corresponding angle is `acos(0.7)`, approximately 45.6 degrees. Cosine distance can reach 2 for antipodal vectors, while the paper describes the score as `[0, 1]`. The implementation retains a bounded salience value after computing its distance, but the paper's geometry should not be used as a configuration target.

### Spreading activation energy

The paper gives the recurrence `E_next = E_current * (1 - delta) * w`, with `delta = 0.2`, and reports a four-node chain at energies `[1.41, 0.90, 0.41, 0.26]`. If the return threshold is `0.3`, the final value is below threshold, so that row does not demonstrate 4/4 returned nodes. The paper also reports a seed energy above 1 in a cyclic graph. The current algorithm records the greatest energy available from a seed to bounded depth in each query. It does not sum repeated paths, so cycles cannot amplify energy. Seeds, intermediate nodes, and edges are scope-filtered, and hops, neighbors, and visited nodes are bounded.

### Noise reduction

The paper's abstract claims 40 percent noise reduction at the default threshold. The reported threshold table includes an 87 percent storage rate in one regime, and the full pipeline example reports 8 of 10 interactions stored. Those observations do not reproduce a 40 percent reduction without the underlying inputs, gate decisions, and denominator. The headline should remain an unverified historical claim until the raw experiment is published and rerun.

### Compression

The paper reports 5, 10, 20, and 50 episodes compressing to 1.7x, 3.3x, 6.7x, and 16.7x. Since the reported output stays at three facts, the ratio grows roughly in proportion to episode count. That table does not establish superlinear compression. It is a small mock-provider demonstration, not a general storage-growth law.

### Affective retrieval

The paper's controlled experiment reports perfect state-congruent retrieval. That result uses `MockLLMProvider` and `SimpleEmbeddingProvider` in a synthetic setup. It is mechanism evidence. It is not a claim about real-model quality or held-out user workflows.

## Retrieval and lexical candidates

The current engine forms a bounded candidate union from semantic retrieval and lexical retrieval. The lexical path supplies candidates; it does not manufacture an FTS score and add that score to cosine similarity. Final score traces identify semantic, state, graph, and candidate-source fields separately. Baseline retrieval is the default. The biomimetic profile is an experimental profile that adds affective and bounded graph contributions.

## MCP and hooks

The paper lists eight MCP tools. The current service facade retains those names and adds `mnemos_capture`, `mnemos_correct`, and `mnemos_job_status`. Store acknowledgments distinguish a committed event from a pending processing job. Retrieval facts expose version, scope, sources, and score traces. Correct and forget require an expected version; old one-argument forget calls resolve the current version through the service.

The paper describes a `hook_autostore` path that passively ingests user prompts and tool failures. The current Claude adapter reads documented command-hook events and role-tagged transcript deltas, then uses the local service and a durable checkpoint. Codex and Hermes adapters have contract tests and fallback descriptions, but no complete real-host smoke is recorded. Do not describe all hosts as automatically capturing every conversation.

## Storage, safety, and evidence

The paper presents Qdrant as a supported backend. The local runtime acceptance path is SQLite. The offline CLI reindex workflow stages resumable embedding generations under the runtime ownership lock, preserves active vectors until explicit activation, and takes a backup before switching a complete generation.

The paper's memory safety firewall remains part of the design. The current service adds scope registry validation, private runtime descriptors, a loopback-only authenticated transport, sanitized outbox writes, durable leases, source evidence, revisions, and suppression identities for forget. A backup can preserve historical content, so live logical deletion must not be described as forensic deletion from all copies.

The paper's negative baseline cases remain part of the evaluation design. Answerable queries and no-answer queries must be scored separately. A no-answer result is correct only when the system abstains or returns no supported fact; a fabricated response is a failure. No full real-model baseline, ablation, or held-out quality result is accepted in this worktree.

See the dated [claim and evidence ledger](validation/2026-09-16/claim-evidence-ledger.md), [functional local memory](functional-local-memory.md), and [host validation](validation/local-memory-hosts.md) for current evidence boundaries and operational setup. No paper publishing source has been located in this repository; this errata does not modify the hosted paper.
