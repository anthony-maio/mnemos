# Mnemos claim and evidence ledger

Assessment date: September 17, 2026. This covers the unreleased
functional-local-memory worktree, not the installed public package or website.
The capability IDs come from the original implementation plan.

The latest complete package verification is frozen source `312b8d6`:
858 tests passed with four skips, plus Black, strict mypy, JavaScript syntax,
build, and Twine. This includes embedding roles, conflict collision protection,
concurrent backup consistency, and the expanded cross-adapter fixture. The
[fresh installed-wheel check](combined-package-checkpoint-v2/README.md) then
finished with 40 passing tests and two continuity failures. A subsequent
[installed diagnostic](installed-continuity-diagnostics-v1/README.md) reproduced
HTTP read timeouts and multi-second retrieval metadata updates; SQL execution,
locking, and commit time still need separation. The separate installed replay
probe did not run. The combined `9aba209` run's
benchmark compatibility failure and its repair remain preserved in the
[failed checkpoint](combined-package-failure-v1/README.md). Claude plugin runtime
repair `ec0f95e` passed 50 focused tests with two opt-in skips. It is not covered
by the `312b8d6` full package result; native plugin-session acceptance remains open.
An implemented capability does not close its quality, native-host, or pilot
acceptance gate.

## Claims and current boundaries

| ID | Claim | Evidence and supported scope | Acceptance still open |
| --- | --- | --- | --- |
| C01 | Persistent local memory | SQLite/service regression checks cover durable events, jobs, restart, retry, and fact persistence in the [combined source checkpoint](combined-package-checkpoint-v2/README.md). | Installed continuity failures, continuous daily operation, and the two-week pilot. |
| C02 | Project, workspace, and global isolation | Scope and lifecycle regressions are included in that package. Registry identities and membership replace folder-name identity. | Complete native-host isolation matrix. Passing fixtures do not prove every external client supplies correct identity. |
| C03 | Surprisal-gated encoding | Mechanism tests cover distance bounds, scoped prediction history, explicit-write bypass, and the later symmetric embedding role. | Real-model retention/noise benefit over baseline and simple filtering. The historical 40 percent claim remains unverified. |
| C04 | Evidence-based mutable memory | Expected-version corrections, revisions, graph refresh, and canonical handling have regression coverage. Source `46cc101` additionally rejects a conflict proposal that would relabel an existing canonical assertion. | General semantic support of extracted proposals. The collision rule is conservative and does not establish entailment. |
| C05 | Affective routing | Scoring, explicit-state handling, fallback, and top-1 execution are tested. Real-model mechanism evaluation is separate from injected-state controls. | Reliable classifier fidelity and a reproducible retrieval benefit within the declared accuracy and latency budgets. |
| C06 | Sleep consolidation and compression | Durable maintenance, evidence-linked processing, retention, and failure/retry machinery are implemented. [Development extraction evidence](preference-contract-development-v1/README.md) preserves omissions and unsupported proposals. | Held-out supported precision, critical retention, and information-preserving compression. A valid JSON response is not a supported fact. |
| C07 | Associative graph retrieval | Traversal, scope, persistence, and insertion-order controls are tested. The [capacity diagnostic](graph-capacity-diagnostic-v2/README.md) completed 15 development observations; successful recall was 0/5 at capacities 4, 8, and 16. | Associative retrieval benefit. Increasing capacity is not an accepted remedy; defaults and evaluation gates remain unchanged. |
| C08 | Independently usable, composable modules | Standalone and integrated module regression checks exercise write filtering, scope, graph state, and lifecycle behavior. | Completed real-model baseline, full-pipeline, and five module-disabled comparisons. Mechanical correctness does not establish usefulness. |
| C09 | Inspectable, correctable, forgettable memory | Facts expose status, version, evidence, and ranking traces. Correction and forget/replay suppression have persistent-service checks. | Complete native correction/forget continuity. Logical forgetting does not erase historical backups or external host transcripts. |
| C10 | Truthful local setup and readiness | Configured runtime, explicit providers, generated interpreter-bound host commands, and input-specific readiness checks are implemented. | Full native installation acceptance. Embedding revision is an operator-managed identity label; the runtime does not automatically verify a mutable Ollama tag's digest. |
| C11 | Shared memory across Codex, Claude Code, and Hermes | [Host records](../../client-compatibility-matrix.md) distinguish authenticated synthetic continuity, installed provider/bridge calls, bounded native capture, and native lifecycle attempts. | Complete native lifecycle and Codex Desktop acceptance. The latest [Codex attempt](native-codex-lifecycle-v4/README.md) ended at a cancelled store call without a successful write. |
| C12 | Proven retrieval, compression, or replacement advantage | The [corrected performance run](performance-corrected-v4/README.md) measured recall p95 1,088.094 ms against a 500 ms gate and durable capture p95 64.279 ms against 100 ms. The database after loading was 324,862,864 bytes. | Recall latency passes, completed real-model ablations, retained-information accuracy, and external pilot evidence. No general superiority or replacement claim is accepted. |

## Evidence bindings

The following indices bind raw files by byte count and SHA-256. Their READMEs
identify the source revision, configuration, and limitations. Historical results
remain attributed to the source that produced them.

- [Package verification index](access-metadata-package-verification/evidence-index.json): source `2494e31`, including schema 8. No later embedding-role or conflict-collision code is covered by this full checkpoint.
- [Combined checkpoint index](combined-package-checkpoint-v2/evidence-index.json): source `312b8d6`, passing source/package checks and the failed installed continuity check. The separate installed replay probe did not run.
- [Installed diagnostic index](installed-continuity-diagnostics-v1/evidence-index.json): valid timeout reproduction plus the excluded first diagnostic whose fixture was incorrectly placed inside Git.
- [Performance index](performance-corrected-v4/evidence-index.json): frozen d94 run with real query embeddings and synthetic load facts. It is not a semantic-quality test.
- [Metadata-write comparison index](access-metadata-comparison-v1/evidence-index.json): four fixed-order baseline/candidate arms. Baseline timing varied sharply, so the run does not establish a reliable speedup or close the full recall gate.
- [Graph diagnostic index](graph-capacity-diagnostic-v2/evidence-index.json): frozen d94 source and vectors, three capacities, development cases only, no new model calls.
- [Extraction development index](preference-contract-development-v1/evidence-index.json): development evidence, not held-out precision acceptance.
- [Native Codex index](native-codex-lifecycle-v4/evidence-index.json): installed-package preflight and the incomplete native attempt. It does not establish why the client cancelled the call.

The active v4 model queue retains its frozen d94 code, model digests, dataset,
options, and output paths. Completion and results must be checked before
promoting its running phases to accepted evidence. Later embedding-prefix
comparisons require separate outputs and bindings.

## Publication boundary

The [paper errata](../../paper-errata.md) is a local correction draft. No paper
publishing source has been located in this repository, and the hosted paper,
website, and package have not been changed by this work. Absolute competitor,
infinite-scale, production-readiness, and practical-superiority claims remain
unsupported here. Publication requires separate authorization.
