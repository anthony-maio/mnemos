# Source reconciliation integration follow-up

Status: implementation design; production integration is not complete. This supplements T17-T19 without changing their acceptance criteria.

## Evidence motivating the work

The frozen reconciliation comparisons in `docs/validation/2026-09-21/source-reconciliation-terminal-v1`, `-v2`, and `-v3` preserve the failed decisions. V3 matches eight of ten operation-level expectations, including intentionally unresolved cases. It still omits new independent facts alongside unrelated targets. Its assistant-conflict explanation incorrectly describes the source as direct user content, although the validated source role remains assistant. Model-written reasons must remain diagnostic text, never evidence or authority.

The experimental schema and runtime now reject duplicate user-fact additions, nonmatching support, and arbitrary support against duplicate identities. The combined experimental suite passed 96 tests before the instruction-placement change; all 22 reconciliation tests passed after that change. These checks do not validate production writes.

## Preserve the existing production transaction

The entry point is `LocalMemoryService.process_job` in `mnemos/memory_service.py`, after the existing encoding gate and scope checks. `SleepDaemon.propose_facts` in `mnemos/modules/sleep.py` currently returns generated-text `FactProposal` objects. A typed extraction outcome is needed to distinguish retained source candidates, discarded source material, and unresolved reconciliation. An empty proposal list must not turn unresolved work into `no_durable_facts`.

Use an explicit experimental integration seam until quality and lifecycle checks pass. Production code must not depend on benchmark modules. Move the selected implementation into a product module only when that integration is ready, with benchmark entry points delegating to it. Preserve the baseline comparison path and independently selectable modules.

Keep `SQLiteStore.commit_fact_and_job` as the atomic boundary for the lease, target versions, canonical identity, suppression, evidence, revisions, processed source events, and job result. A staged decision never authorizes a direct database write. Validate the source snapshot, target snapshot, scope, role, exact range, fingerprint and content hash again at the relevant boundaries. Keep the post-embedding version fence.

Existing support does persist evidence: the service creates a candidate using the supported content, and the store maps its temporary ID to the canonical existing ID before inserting evidence. Do not implement a repair on the assumption that support evidence is missing. Any replacement must preserve this behavior and its version fence.

## Resolve the semantic contract before adoption

1. New independent source assertions must be addable when observed targets are unrelated. Keep the unchanged failing development cases and add fresh cases before quality acceptance.
2. Exact normalized support may use only a unique observed identity. Semantic equivalence, contradiction and correction relevance remain separate judgments.
3. Bounded target retrieval is not a complete inventory. A correction with no observed target remains unresolved, not a new fact. Broader scoped target lookup may resolve it; lack of a top-five hit cannot prove absence.
4. Split the overloaded `correction_or_rejection` classification. A complete prospective requirement such as "we will not keep the side panel in this project" can carry durable negative intent without an old fact to replace. A referential correction whose subject or target is missing must stay unresolved. Preserve the exact source wording and qualifications; do not invent a rejection or replacement assertion. Establish tests for both cases before changing the contract.
5. Keep source role, kind and evidence IDs derived from validated source records. Model reasons can be incorrect even when the operation is safe. Never use reasons to infer user authority or rewrite fact content.

## Keep unresolved work durable and inspectable

The first integration contract is now implemented in `mnemos/source_review.py`.
`SourceReviewReference` retains the candidate index, source identity and hash,
exact range, partition, role/kind, revision flag and observed target/version.
It contains neither copied source text nor model-written explanations. Resolving
the excerpt rechecks source provenance, current target version and partition,
and the write firewall. The experimental pipeline now returns these references
alongside its existing diagnostic review objects. This does not persist reviews
or authorize their resolution; storage must check suppression and leases inside
the transaction. The focused suite passes 157 tests, and Python 3.10 type checks
pass for the new contract, pipeline adapter and contract tests.

Storage integration remains open. In particular, completed-job retention must
not prune a source while a review still needs it. Forgetting a target or a shared
source must invalidate its review references even when another fact preserves
that source. The existing source-deletion path clears job results, but that alone
does not cover every shared-source or target-only invalidation case.

Persistence decision after inspecting `SQLiteStore` and `backup.restore_backup`:
use a dedicated indexed review table, keyed by job ID and candidate index, with
the typed reference and lifecycle state. Keep only review identities/counts in
the job result. Existing forget queries inspect job payload dependencies, so
placing all review dependencies in opaque result JSON would leave target-only
forget and shared-source suppression uncovered.

Required lifecycle behavior for the implementation:

- Commit references in the existing fact/job transaction after validating the
  leased source, processed source IDs, current target versions and suppression.
  Unresolved-only jobs complete with an explicit review decision; mixed results
  retain both facts and review identities.
- Keep open-review sources out of `prune_processed_evidence`. Ordinary age-based
  pruning must not silently destroy unresolved work. Explicit discard or
  invalidation may release that retention pin.
- Invalidate source-linked reviews when the source is suppressed, including
  shared sources retained for another fact. Source presence alone is not enough
  to keep a review actionable. Invalidate targeted reviews when their target is
  forgotten or deleted; changed target versions require renewed reconciliation.
- Apply the same invalidation after `backup.py` merges current suppressions into
  a restored database. The restore path also directly deletes suppressed events,
  separately from `forget_fact`.
- Revalidate provenance and suppression on scoped inspection and resolution,
  even when the persisted lifecycle status still says open. Low-level chunk
  deletion is not a substitute for the explicit forget operation.
- Project configuration currently enables/disables registry entries; there is
  no project-data deletion operation. Apply registry scope checks to review
  access. Do not describe disabling a project as deleting its data.

The contract evidence is archived in
`../validation/2026-09-21/source-review-reference-v1/`. The dedicated table,
retention pins and lifecycle operations above are still unimplemented.

Persist each candidate outcome atomically with the job result, using a versioned record keyed by job and candidate index. The record needs the source event, source range/hash, validated role/kind, operation, observed target/version when applicable, and review state. Diagnostic reasons must be labeled separately from source evidence. Decide whether the durable job-result payload suffices or a dedicated indexed table is needed for scoped review before adding a migration.

Accepted facts and unresolved outcomes can coexist in one extraction. Record counts and identities explicitly. Provider, malformed-output, lease and version failures still follow durable retry behavior. A valid unresolved result is not a provider failure and must not cause endless identical retries.

Review and resolution must recheck current source availability, target version, scope and suppression. Forgetting, source edits, project deletion, backup/restore and replay must cover outcome records too; they must not become a second store from which forgotten text can be resurrected. No live migration is authorized.

## Verification order

1. Preserve the current failed comparisons and fix the remaining semantic contract in development.
2. Add the typed extraction outcome and service adapter, with exact-source proposals and explicit unresolved records.
3. Exercise add, support, correction, conflict, unresolved-only and mixed outcomes through an isolated real SQLite transaction. Check evidence attachment, duplicate identity, role attribution, lease expiry, target changes during embedding, suppression and rollback.
4. Repeat capture/restart/recovery/correction/forget/replay with all five modules enabled and a real local provider. Keep each failed phase visible.
5. Run the untouched evaluation corpus and required baseline/full/one-module-disabled comparisons. Follow with fresh native-host sessions, the daily-use pilot, final package checks and the updated writing handoff.

Publication, live host configuration and live data migration remain separate authorization boundaries.
