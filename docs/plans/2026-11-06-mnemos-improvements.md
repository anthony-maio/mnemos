# Mnemos improvement work

Status: active. Started November 6, 2026 from delivery commit `9770c00` on
`wt/mnemos-improvements`. This goal resumes the work deferred on October 28.
No release, live configuration change, or pilot completion is implied.

## Scope and completion evidence

| Priority | Work | Completion evidence | Current state |
| --- | --- | --- | --- |
| 1 | Repair missing-source graph writes and transaction recovery; prepare the affected-user fix | Regression tests, separate-process client coverage, rollback checks, reviewed delivery patch | Implemented and locally tested; public history integrated locally at 7fcca905; minimal backport d8a5e647 passes all 316 public-branch tests; review package prepared, publication pending |
| 2 | Add evidence-backed answer support and abstention | Development comparison and fresh evaluation with false-answer, supported-answer, conflict, and latency results | V4 completed 352 observations: hybrid temporal support retained 24/27 answerable hits but abstained on 11/17 versus baseline 14/17; no promotion; failure review and fresh evaluation open |
| 3 | Separate extraction from retrieval evaluation | Identical accepted-fact snapshots for retrieval arms; frozen prospective rubric and dataset identities; independently reported extraction and ranking outcomes | Accepted-fact development comparison completed; scope-fixture defect repaired; separate extraction and fresh evaluation open |
| 4 | Complete explicit remember/inspect/restart/correct/forget workflow | Installed artifact, repeatable native-host acceptance, truthful job status, two-week daily-use pilot across two real projects | Installed CLI/MCP and all six native Codex phases passed; Claude v2 passed four phases then failed during forgetting, cleanup passed; private pilot projects selected, entry gates and actual use remain open |
| 5 | Improve context efficiency | Host tokenizer with conservative fallback, concise source-backed context, supported facts per token and recall/latency measurements | Optional Python host counter and whole-context packing implemented; native-host wiring, span compression and quality measurements open |
| 6 | Align code, installation and public claims | Reviewed canonical branch, verified artifact, matching README/site/package claims, separately recorded publication state | Public state refreshed and history integrated locally; website delivery source verified against live bytes; local claims updated; final artifacts, site update and publication open |

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
  the development experiment and evaluation boundaries.
- An experimental checker now selects retained source evidence for current,
  scoped accepted facts. Provider failure is separate from abstention, and
  corrections, forgetting, scope changes or evidence changes during judgment
  invalidate the result. It is not enabled in production retrieval.
- The first local comparison and a corrected-scope run each completed 80
  observations. The original fixture folders inherited one enclosing Git
  identity; an independent-repository fixture and regression test fixed it.
  Both runs and their negative results are retained in the
  [development report](../validation/2026-11-06/answer-support-development.md).
  Corrected hybrid support retained 3/10 answerable results, abstained correctly
  on 1/10 no-answer queries and failed response validation on 11/20 queries.
  No historical scores changed. The private local model server was unloaded
  and stopped after the comparison.
- [Host tokenizer support](../context-tokenizers.md) accepts an explicitly
  identified Python counter, measures concatenated qualified context, and
  restarts packing with bytes if that counter fails. New boundary tests cover
  nonadditive counts, Unicode, oversized records and invalid counters. A
  synthetic interleaving also reproduced a correction after budget selection;
  retrieval now omits that changed revision.
- Context verification: 43 focused tests passed, followed by 136 broader tests
  across the engine, service, host context/view adapters, retrieval comparisons
  and support checker. Black, strict mypy for the four affected source modules
  and `git diff --check` passed. These checks do not establish native-host
  tokenizer compatibility or context-efficiency gains. Both experiment archives
  passed hash verification; all 20 archived databases passed integrity and fact
  count checks.
- Product `24887ef3` built wheel and source artifacts and passed Twine checks.
  Its installed smoke check passed real-embedding CLI/MCP restart recall and
  scope isolation outside the checkout. The native Claude attempt at runner
  `c3a14fd3` passed teaching and completed-job observation, then failed login in
  the first post-restart native session. Cleanup passed. The original evidence
  is preserved; see the [installed/native report](../validation/2026-11-06/installed-native-checkpoint.md).
  A separate validation login is requested for another attempt. The new guard
  refuses expired copied credentials before inference. The pilot remains
  NOT_STARTED; project selection is also pending.
- Experimental module defaults are now off across application and embedded
  configuration. Embedded ingestion and consolidation honor those switches;
  explicit memory remains available. Research fixtures opt in explicitly.
  Setup preset selection changes flags deliberately, while ordinary settings
  edits preserve inheritance. See the [defaults report](../validation/2026-11-06/experimental-defaults.md).
- Defaults verification: the initial broad run passed 2,586 tests and exposed
  58 failures plus 19 missing-fixture errors. After fixing opt-ins, materializing
  unchanged archived fixtures and updating the tokenizer response expectation,
  all 264 affected tests passed, including all 77 prior failed/error cases.
  Five artifact-harness checks and source formatting/type checks passed. A
  fresh full-suite run and the new installed artifact check are not implied.
- Product `e8a4f12e` subsequently built wheel/source artifacts and passed Twine.
  Its installed smoke check confirmed all five disabled defaults and passed
  real-embedding CLI/MCP restart recall, same-ID inspection and project isolation.
  Consumer service/workspace cleanup passed. The [new receipt](../validation/2026-11-06/installed-smoke-e8a4f12e.json)
  supplements the earlier artifact evidence. No publication, live profile
  installation, complete native lifecycle or pilot use is claimed.
- Product `6d46aa1a` adds explicit probability contracts, candidate-at-a-time
  judgment and an optional separate answer/source treatment. Existing production
  behavior stays unchanged. Seventy-six focused tests, formatting and strict
  source type checks passed before the frozen v2 development comparison.
- The [v2 comparison](../validation/2026-11-06/answer-support-development-v2.md)
  completed 320 observations over the same 20 development questions. All four
  support treatments received identical baseline facts within each retrieval
  arm. Hybrid candidate-at-a-time support retained 9/10 expected answers and
  abstained correctly on 5/10 no-answer questions; only 4/10 answerable returns
  were free of extra unsupported facts. Separating answer/source questions
  retained 10/10 but abstained on only 3/10. No treatment qualifies for promotion.
- All 444 model replies are accounted for. Source hashes, database integrity,
  reconstructed snapshot hashes and archive members verified. The private model
  server was unloaded and stopped. Further semantic development, fresh and
  extraction evaluations, native authentication/lifecycle, host tokenizer
  integration, the actual pilot and publication remain open.
- Product `6d46aa1a` built source/wheel artifacts, passed Twine and passed a fresh
  installed smoke check outside the checkout. See the
  [Codex checkpoint](../validation/2026-11-06/native-codex-checkpoint.md).
- Codex v1 at runner `5741ffb1` failed when CLI 0.146.0 was rejected as too old
  for the selected model. V2 at `28e4ec92` used the app-bundled 0.164.0 alpha
  runtime and authenticated with the pinned model, but the harness disabled
  its required Code Mode transport. Neither completed a memory tool. Both
  private archives verified and cleanup passed; normal credentials were unchanged.
- The current runtime also changed user-message provenance records. A failing
  regression and offline trace check established the parser repair; 88 focused
  checks pass. The original native failures remain failed. Native Code Mode
  transport and MCP-call provenance required new frozen acceptance protocols.
- Codex v3 reached tools but required scoped permission configuration. V4 then
  completed store, consolidate and job status, but the strict parser rejected
  an additional native dispatch correlation event. Both original failures are
  archived. The v5 parser validates that event and recovers all three saved v4
  calls with native model correlation; 118 focused tests pass. Full native
  lifecycle acceptance and the actual pilot remain open.
- Codex v5 completed store, consolidation and job observation, but its model
  script read a nonexistent job ID from consolidation. The cell failed before
  the model recovered with the store response's ID. The unchanged gate rejects
  that attempt. The failure is archived; MCP descriptions and operating guidance
  now distinguish aggregate maintenance from per-job completion. A new artifact
  and native attempt must establish any improvement from that repair.
- Product `929c44a4` built from its source distribution and passed Twine and
  a fresh installed CLI/MCP restart smoke check with real embeddings. The
  installed tool catalog contains the repaired descriptions. V6 binds that
  artifact and makes the original job-ID source explicit in the two write
  prompts, preserving the failed-cell and other lifecycle gates.
- Native Codex v6 at runner `f15bf9f5` passed all six explicit-memory phases:
  teach, restart recall/evidence inspection, same-ID correction, version-two
  restart recall, reviewed forgetting and absence after exact original-source
  replay and another restart. All 19 MCP calls had native model correlation.
  The retained, unrelated and foreign controls passed; database integrity and
  cleanup passed. The [receipt](../validation/2026-11-06/native-codex-v6-receipt.json)
  binds 776 verified private archive members. This is one synthetic scenario,
  not pilot use, a general reliability rate or a quality improvement.
- The full suite at `3446220c` passed: 2,724 passed, four environment-specific
  skips, no failures or errors. The [regression receipt](../validation/2026-11-06/full-regression-3446220c.json)
  preserves an earlier aborted run whose in-checkout temporary paths caused
  fixture scope collisions, plus the complete corrected external-directory run.
- The [capacity comparison](../validation/2026-11-06/answer-support-capacity-v3.md)
  completed 320 observations over 80 frozen inputs and twenty reused questions.
  Both 14B treatments abstained correctly on 10/10 no-answer questions in each
  arm. Hybrid returned the expected answer on 9/10 questions, with 8/10 clean
  hits, one historical-queue false support and one timeout per treatment. The
  joint treatment was faster and is the leading development candidate. The 4B
  joint treatment retained 5/10 abstention. No defaults changed.
- The capacity audit verified 128 source hashes, all paired inputs, ten original
  stores and snapshot hashes, and 567 requests against 565 retained responses.
  Two missing responses belong to the two timeouts. An analysis lookup defect
  initially hid that missing-ID list; the original output is preserved and the
  corrected accounting leaves every selection metric unchanged. Cleanup passed.
- The [delivery refresh](../validation/2026-11-06/delivery-state.md) found PR #5
  merged at public `86644437`, while PyPI remains `0.6.0`. Disposable probes
  confirmed that caller-transaction and partial-write rollback protections still
  differ from the local fix. A private-index merge preflight found one storage
  conflict; keeping the tested local implementation produces the exact local
  tree. Public integration remains uncommitted.
- The live site's three main evidence pages exactly match website delivery
  source `08d393ee`. The older outer checkout and its edits remain untouched.
  Local README, changelog and host compatibility guidance now identify the
  newer native checkpoint and distinguish it from the installable public
  preview. All 12 documentation consistency checks passed.
- Next development work is temporal applicability over explicit replacement
  evidence. Separate extraction/fresh evaluation, native tokenizer integration,
  pilot preparation/use and consistent delivery remain open.
- Product `c320c3ef` adds optional temporal applicability and scoped comparison
  evidence, with an explicit reference date shared by both comparison arms.
  Dependency changes invalidate the whole result, provider state is isolated
  between calls, and repeated state counts against the request-byte budget.
  The [preparation checkpoint](../validation/2026-11-06/answer-support-temporal-preparation.md)
  records 80 passing focused checks, formatting, strict types and the offline
  prompt preflight. The code remains outside production retrieval.
- The frozen v4 comparison started from that commit, with 176 identical inputs
  and 352 planned observations over 44 development questions. Source, labels,
  thresholds and timeout limits remain fixed during the run. Its results are
  pending and will be reported separately from implementation validation.
- Public `master` was rechecked at `86644437` and integrated locally at
  `7fcca905` with an identical file tree. A separate minimal backport at
  `d8a5e647` passes the full 316-test public suite and all formatting checks.
  Its four full-mypy diagnostics were reproduced unchanged on the public
  baseline. The [storage delivery checkpoint](../validation/2026-11-06/storage-delivery-checkpoint.md)
  records the patch, PR draft, verification and publication limits.
- The account holder selected two projects for a private pilot. Their identities
  and privacy condition are retained outside tracked files. Raw pilot memory
  and activity logs must remain private. Selection does not authorize activation
  or satisfy the entry checks; no pilot days have occurred.
- After fresh sign-in to the isolated Claude validation profile, metadata passed
  the one-hour lifetime guard and the CLI matched version 2.1.191. Runner
  `93ff6bb4` binds a new Claude attempt to the already verified product `929c44a4`
  wheel. Forty-eight focused runner/process tests and strict runner types passed.
  Native results remain pending until all phases and cleanup are verified.
- The [v4 temporal comparison](../validation/2026-11-06/answer-support-temporal-v4.md)
  finished all 352 observations. Hybrid answerable hits were 24/27 for both
  treatments, while correct abstention fell from 14/17 to 11/17 with temporal
  comparison context. Both had three timeouts in that arm. All 129 source
  hashes, 176 paired groups, 22 stores and 620-request/609-response accounting
  verified. Cleanup passed. No promotion or fresh-evaluation claim is justified.
- [Claude v2](../validation/2026-11-06/native-claude-v2-checkpoint.md) authenticated
  and passed teaching, restart recall, correction and version-two recall. The
  strict parser rejected an unsuccessful tool result during forgetting; final
  replay/recall was not attempted. The failed run is preserved. Owned processes,
  the service descriptor and copied credentials were removed; source credentials
  were unchanged. Investigation remains open.
- At the account holder's requested stopping point, both runs are terminal and
  their owned processes are absent. No new long run was started. Resume with
  the Claude forgetting failure and temporal failure audit, then continue the
  original fresh/extraction/context/pilot/delivery obligations. The goal has not
  been marked complete or paused by the runner.
