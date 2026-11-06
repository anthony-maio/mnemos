# Mnemos improvement work

Status: active. Started November 6, 2026 from delivery commit `9770c00` on
`wt/mnemos-improvements`. This goal resumes the work deferred on October 28.
No release, live configuration change, or pilot completion is implied.

## Scope and completion evidence

| Priority | Work | Completion evidence | Current state |
| --- | --- | --- | --- |
| 1 | Repair missing-source graph writes and transaction recovery; prepare the affected-user fix | Regression tests, separate-process client coverage, rollback checks, reviewed delivery patch | Implemented and locally tested; delivery review and affected-user release open |
| 2 | Add evidence-backed answer support and abstention | Development comparison and fresh evaluation with false-answer, supported-answer, conflict, and latency results | Open |
| 3 | Separate extraction from retrieval evaluation | Identical accepted-fact snapshots for retrieval arms; frozen prospective rubric and dataset identities; independently reported extraction and ranking outcomes | Existing snapshot harness identified; new development protocol written; fresh evaluation open |
| 4 | Complete explicit remember/inspect/restart/correct/forget workflow | Installed artifact, repeatable native-host acceptance, truthful job status, two-week daily-use pilot across two real projects | Historical evidence exists; current validation and pilot open |
| 5 | Improve context efficiency | Host tokenizer with conservative fallback, concise source-backed context, supported facts per token and recall/latency measurements | Open |
| 6 | Align code, installation and public claims | Reviewed canonical branch, verified artifact, matching README/site/package claims, separately recorded publication state | Public surfaces disagree; reconciliation open |

Keep the existing frozen scores, raw evidence and thresholds intact. Use new
development data for tuning, then new evaluation data. The recovered 600-row
run is regression evidence, not a fresh blinded evaluation. Do not substitute
synthetic sessions for actual pilot days or treat protocol checks as native
host acceptance.

## Work order

1. Storage failure and atomic graph writes, including stale clients.
2. Prospective answer-support protocol and fixed-snapshot development harness.
3. Answer-support behavior and context packing, measured independently.
4. Installed explicit-memory workflow and current native-host acceptance.
5. Prepare exact pilot configuration, backup and rollback for approval; run the
   existing two-week/two-project protocol during actual use.
6. Complete validation, reconcile claims and prepare delivery. Any needed
   publication approval follows concrete, reviewable artifacts.

The worktree is initially sparse: historical `docs/validation` archives remain
in Git and the preserved worktrees. Source, tests, benchmark code and datasets,
plans, the October recovery reports and host validation index are checked out.
Materialize any archive required by a specific check before running that check.
No historical archive is removed or rewritten.

## Execution record

- Initial assessment: 41 retrieval/view/classification/comparison tests passed
  on the delivery checkout. A disposable database reproduced a missing-source
  foreign-key error with an open transaction. These checks did not run models,
  measure retrieval quality, or modify live memories.
- Storage baseline: 13 existing storage/graph tests passed. Six new regression
  cases then failed on the unchanged implementation: stale sources from another
  process in both graph paths, partial replacement after an insert failure, and
  preservation of caller transactions on success, missing source and failure.
- Storage repair: source existence is checked while holding the SQLite write
  transaction in both graph paths. Neighbor replacement uses a savepoint so a
  failure restores prior edges without committing or discarding caller work.
- Verification: 47 tests passed across graph writes, SQLite storage, graph
  persistence/partition/recovery, durable-memory repair, touch and access
  metadata. Black and strict mypy for the changed storage module passed, as did
  `git diff --check`. Tests use disposable databases; no native host or live
  memory database was involved.
- Validation environment: isolated CPython 3.12.11, NumPy 2.2.6. The reused
  environment's newer NumPy stubs used Python 3.12-only syntax while mypy targets
  Python 3.10, so validation moved to a private environment. No repository
  dependency constraint or shared environment was changed.
- The [answer-support protocol](2026-11-06-answer-support-protocol.md) defines
  the next development experiment and evaluation boundaries. Its implementation,
  new corpus and model runs remain open.
