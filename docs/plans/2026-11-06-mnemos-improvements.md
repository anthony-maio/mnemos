# Mnemos improvement work

Status: active. Started November 6, 2026 from delivery commit `9770c00` on
`wt/mnemos-improvements`. This goal resumes the work deferred on October 28.
No release, live configuration change, or pilot completion is implied.

## Scope and completion evidence

| Priority | Work | Completion evidence | Current state |
| --- | --- | --- | --- |
| 1 | Repair missing-source graph writes and transaction recovery; prepare the affected-user fix | Regression tests, separate-process client coverage, rollback checks, reviewed delivery patch | Implemented and locally tested; public history integrated locally at 7fcca905; minimal backport d8a5e647 preserved; separate 0.6.1 candidate 9ea3852a repairs four baseline typing errors and aligns metadata. Its 316 source tests, five installed regressions, typing, formatting and packaging pass. Legacy benchmark fails both in-memory contradiction comparisons; all six SQLite comparisons pass. Exact review packet prepared; release readiness and publication remain open |
| 2 | Add evidence-backed answer support and abstention | Development comparison and fresh evaluation with false-answer, supported-answer, conflict, and latency results | The bounded installed retrieval/support comparison is complete: 272 observations, 68 known questions, 136 matching candidate/context pairs and no ranking misses or operational failures. The fixed QA/contrast treatment retains 72/80 expected facts, correctly abstains on 52/56 absent answers, selects four unsupported facts and reaches 94.74% precision. Retention at 90% and complete p95 at 8,641.48 ms fail the original gates. The adapter has 23 boundary checks and 71 focused passes. This comparison and the earlier codec series are closed; no rerun or fresh acceptance is justified by this result. Treatment stays outside defaults; semantic-quality and critical-fact acceptance remain open |
| 3 | Separate extraction from retrieval evaluation | Identical accepted-fact snapshots for retrieval arms; frozen prospective rubric and dataset identities; independently reported extraction and ranking outcomes | First fresh retrieval-only comparison completed and audited: 200 sessions, 700 declared facts, 580 query strings and 5,220 observations. Vector/hybrid Recall@5 100%, lexical 93.5%; all have 50% clean hits and 22.7% correct abstention. Every successful abstention has zero eligible facts; none with available candidates. The 90% abstention requirement fails. Extraction, support and full-module comparisons remain separate |
| 4 | Complete explicit remember/inspect/restart/correct/forget workflow | Installed artifact, repeatable native-host acceptance, truthful job status, two-week daily-use pilot across two real projects | Product f5a944ea passes installed CLI/MCP checks and all six native Codex v12 phases: 19 correlated calls, six complete inventories, installed guidance delivery and one suppressed original-source replay. The independent audit verifies original jobs, same-ID correction, reviewed deletion, final recall and retained controls. The refreshed pilot preparation now uses the same wheel and passes installation/configuration/rollback checks. Earlier incomplete/failed attempts and Claude/capture evidence retain their source bindings. Applicable quality, normal hook trust, complete pilot-configuration acceptance and actual use remain open; pilot inactive with zero days |
| 5 | Improve context efficiency | Host tokenizer with conservative fallback, concise source-backed context, supported facts per token and recall/latency measurements | Host-tokenizer support with a conservative fallback is implemented; native tokenizer compatibility and fresh quality remain open. Product f5a944ea passes the audited 320-observation installed comparison using the same native launcher in both arms: recall median falls from 110.32 to 97.89 ms and prompt median from 257.89 to 245.86 ms. All 160 output/context pairs match and all 208 captures survive restart with completed jobs. Recall p95 passes 500 ms at 104.92 ms; complete Stop/prompt p95 remains 321.73/365.53 ms, failing the experiment's whole-command 100 ms bound. The prospective installed pilot command comparison is complete on the same f5 wheel. It retains 320 observations, 160 exact output pairs and 208 correlated persistence spans and recovered captures. Latest gates: recall: Python FAIL, native FAIL; Stop: Python FAIL, native FAIL; prompt: Python FAIL, native FAIL. Internal transaction and dispatch-to-commit timings remain diagnostics. Neither launcher exposes an early durable acknowledgment, and command stdout alone cannot prove capture success. Native-host acknowledgment acceptance remains unproven. The rejected sparse-support experiment remains disabled. The native launcher remains experimental; Git identity resolution, command startup, semantic span compression and broader consumption remain open |
| 6 | Align code, installation and public claims | Reviewed canonical branch, verified artifact, matching README/site/package claims, separately recorded publication state | Product f5a944ea passes isolated real-BGE CLI/MCP restart recall, scope isolation and indexed evidence ordering. Its wheel verifies 96 installed package files, 85 committed Python files, 57 dependency versions and 17 unchanged model assets. Full regression on 1299ed9b passes 3,674 cases with four platform skips, including all 75 native-launcher subprocess tests. README and compatibility notes identify this artifact and preserve older native-host bindings. The refreshed delivery bundle reconstructs fb062fe3 and includes this unchanged wheel plus a source archive that reproduces all 101 wheel members. It remains unapplied. Canonical integration, website alignment, package delivery and publication remain open |

Keep the existing frozen scores, raw evidence and thresholds intact. Use new
development data for tuning, then new evaluation data. The recovered 600-row
run is regression evidence, not a fresh blinded evaluation. Do not substitute
synthetic sessions for actual pilot days or treat protocol checks as native
host acceptance.

## Work order

The November 7 adversarial review changes the order without reducing the six
original goals. Follow the [candidate convergence plan](2026-11-07-candidate-convergence.md):

1. Close and archive the completed codec series. No further output variants.
2. Finish the minimal storage backport's installed-package review independently.
3. Freeze product `f5a944ea`, refresh one canonical delivery bundle and the
   private pilot drafts, and validate the selected native host on that wheel.
4. Run one bounded retrieval-plus-support increment with at most one integration
   repair/rerun. Fresh acceptance follows only a passing development candidate.
5. Measure durable acknowledgment, warm recall and whole-command time separately
   under a prospective protocol. Preserve the prior whole-command FAIL labels.
6. Prepare exact pilot changes and rollback for activation only after entry
   checks pass. Count actual daily use. Publication is a separate final decision.

A manual MCP-only learning pilot is an alternative scope requiring an explicit
prospective decision. It does not waive the original quality or pilot gates.

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
- The account holder subsequently paused the goal and explicitly resumed it.
  The new work retains the two selected private pilot projects and isolated
  validation profile; the pilot remains unstarted.
- The [Claude v2 diagnosis](../validation/2026-11-06/native-claude-v2-diagnosis.md)
  traced the rejected request to target-linked source IDs copied into the
  additional selection. Product `5de8b1f1` clarifies MCP descriptions, parameter
  schema and operating guidance. Forty-nine existing checks, formatting and
  strict module types passed. Its wheel/source build, Twine and fresh installed
  smoke check passed; installed dependency versions match the preceding attempt.
- [Claude v3](../validation/2026-11-06/native-claude-v3-checkpoint.md) at runner
  `67166200` kept native prompts and acceptance rules unchanged. The first four
  phases passed. A 50,711-byte forget preview was replaced with a partial native
  view and saved file; the parser failed. The model also mistyped a source ID,
  then corrected it and deleted only a subset. Six fact rows still contained
  forgotten labels. Final replay was not attempted. All 138 private archive
  members verified, cleanup passed, and source credentials were unchanged.
  A bounded, complete forget-review contract is the next native-workflow repair.
- The [v4 failure review](../validation/2026-11-06/answer-support-temporal-v4-failure-review.md)
  inspected the ten fully assessed vector-arm error observations, containing
  thirteen wrong selections. Each reported probability 1.0. Missing requested
  values, unsupported time extrapolation and unresolved conflicts need distinct
  checks. No model call, threshold change, rescoring or default promotion occurred.
- Product `5de1844f` adds lossless paged MCP forget previews and digest-bound
  short fact/source selections while retaining the storage deletion guards.
  Seventy-seven distinct focused checks pass, including independent-writer
  mutation between resolution and apply. Build, Twine, installed smoke,
  formatting and strict source types pass. The saved oversized preview
  reconstructs exactly in five bounded pages; this is not native acceptance.
- Verifier `aebe2a4e` adds independent page reconstruction and exact alias-to-ID
  selection checks. Its 77-test focused run includes 36 new adversarial cases.
  Original inspection, source-dependency, correlation and receipt gates remain.
  The [preparation checkpoint](../validation/2026-11-06/forget-review-pages-checkpoint.md)
  records the new artifact and prospective Claude v4 protocol.
- The first full run at `5de1844f` finished with 2,749 passes, four skips, two
  failures and 19 setup errors from absent historical fixtures. All 208 restored
  fixture files match Git bytes, and the affected files pass all 50 checks.
  The original failure is archived. A fresh full suite and the next native
  attempt remain pending; the private pilot has no recorded days.
- The fresh [full regression](../validation/2026-11-06/full-regression-review-pages-v2.md)
  at `7f5aec4b` passed 2,806 tests with four platform-specific skips and no source
  changes during execution. Its archive is verified; the prior fixture failure
  remains failed.
- [Claude v4](../validation/2026-11-06/native-claude-v4-checkpoint.md) passed the
  first four phases, then failed on an unapproved extraction-review request.
  Its five forget-preview pages arrived intact and reconstructed completely.
  One explicit short-ref selection committed, but omitted an advisory copy and
  two source-only captures. One fact, three source rows and one evidence row
  retained forgotten labels. Final restart/replay did not run. Both original
  and diagnostic archives verified; cleanup and source-credential preservation
  passed. Tool discovery and complete selection remain repair work.
- [Private pilot preparation](../validation/2026-11-06/private-pilot-preparation.md)
  now includes a read-only setup inventory and inactive configuration draft for
  both selected projects. It separates the proposed local runtime from existing
  provider/store settings. Both draft configurations parse. No live host changes,
  activation approval, pilot days or publication are claimed. Original quality,
  recovery and actual-use requirements remain open.
- A final explicit website check exposed an obsolete preset-label assertion.
  The default test could also discover the older ancestor website checkout.
  The check now requires the canonical path explicitly and verifies the actual
  install commands, SQLite guidance and published/development distinction.
  All twelve documentation checks, formatting and strict test-module types pass.
  The [receipt](../validation/2026-11-06/canonical-website-check.json) preserves
  the initial 11-pass/1-failure result. This test-only repair postdates the full
  regression; product and native-runner code are unchanged.
- Product `6d0e5ba6` clarifies complete forget selection across advisory wording
  and eligible source-only captures. The native runner explicitly denies tools
  outside each phase and verifies the advertised startup catalog. All 180
  focused checks pass, including fourteen new boundary cases. The original nine
  red cases and an intermediate read-only-description failure are preserved.
  Build, Twine, strict types, formatting and fresh installed CLI/MCP checks pass.
  The [v5 preparation](../validation/2026-11-06/review-selection-v5-preparation.md)
  binds the new wheel, matching native dependencies, offline page reconstruction
  and prospective protocol. Full regression and native validation are the next
  checks; none of this reclassifies v4 or starts the pilot.

- The [v5 full regression](../validation/2026-11-06/full-regression-review-selection-v5.md)
  passed 2,820 tests with four platform-specific skips at `15229fb7`. Source,
  all 208 restored fixtures and the explicitly selected canonical website were
  unchanged. The private archive and all six members verified.
- [Claude v5](../validation/2026-11-06/native-claude-v5-checkpoint.md) then passed
  all six phases on product `6d0e5ba6`. The 29 correlated calls used exact phase
  tool catalogs, completed reviewed deletion and suppressed ten exact original
  source replays. Final recall, retained controls, SQLite checks and cleanup
  passed; source credentials were unchanged. All 158 archive members verified.
  This supplies one synthetic native pass, not a reliability rate or pilot day.
- The [three-check offline preflight](../validation/2026-11-06/support-three-checks-v5-preflight.md)
  measured 78 candidate requests over the 44 reused development questions.
  The largest rendered prompt was 4,210 bytes and the largest per-query typed
  request total was 9,591 bytes, within existing limits. No provider call or
  new quality score occurred. Production integration and prospective model
  comparison remained next development work at that checkpoint.
- Product `e5373791` subsequently adds optional `checked_v4` with three typed
  decisions and the existing comparison dependency checks. The checker stays
  outside production retrieval. Its [preparation](../validation/2026-11-06/answer-support-checked-v5-preparation.md)
  records 79 distinct passing checks, preserved red cases, formatting and strict
  types. Actual requests fit the existing budgets at 4,473 bytes per prompt and
  10,383 per query. Runner `64045705` and the prospective v5 protocol are frozen
  before model comparison. No new model result or default promotion is implied.
- [Private pilot preparation v2](../validation/2026-11-06/private-pilot-preparation-v2.md)
  preserves the first draft and binds a second inactive draft to the passing
  `6d0e5ba6` wheel. That installed artifact passed synthetic backup/restore,
  newer-forget suppression, exact source replay and retained/foreign controls.
  Four configuration probes verified the draft launcher's local selectors
  against conflicting inherited settings; no service started. Seven live paths
  remain unchanged. Exact host diffs, persistent runtime, real launcher/MCP/hook
  checks, quality/provider-recovery gates and actual pilot use remain open.
- The [installed launcher check and private host patches](../validation/2026-11-06/private-pilot-launcher-and-diffs-v2.md)
  now cover MCP store/job completion, restart recall, project isolation and
  both synthetic hook captures on the verified `6d0e5ba6` wheel. A failed probe
  assertion and a patch line-ending hash mismatch are preserved alongside
  their corrected receipts. All 58 archive members verified. Seven patches
  are prepared but unapplied; nine live paths remain unchanged. Native-host
  application, persistent installation, rollback and pilot gates remain open.
- The [three-check comparison v5](../validation/2026-11-06/answer-support-checked-v5.md)
  completed all 352 observations with 130 source hashes, 176 pairs, 22 stores
  and 620 request/response bindings verified. Hybrid hits were 27/27 for both
  treatments, but clean hits fell from 26 to 19 and abstentions from 14/17 to
  13/17 with the candidate. Four candidate contract failures received no
  abstention credit. Exact prompt reconstruction corrected an audit overcount
  from 4,848 to 4,473 bytes without changing any quality result. Both archives
  verified and runtime cleanup passed. Keep the candidate experimental; no
  quality promotion, fresh-evaluation pass or pilot day follows from this run.
- The prospective [Codex v7 checkpoint](2026-11-06-native-codex-review-selection-v7.md)
  binds a fresh native attempt to the same installed `6d0e5ba6` artifact that
  passed Claude v5. Twelve runner checks and formatting pass. The original
  Codex v6 pass remains bound to its earlier artifact. Full regression at the
  new frozen runner source is required before this attempt starts.
- The [Codex v7 full regression](../validation/2026-11-06/full-regression-codex-v7.md)
  passed 2,845 tests with four platform-specific skips at `6c12ff94`. Source,
  the canonical website and 208 restored fixtures stayed unchanged. All six
  archive members verified.
- [Codex v7](../validation/2026-11-06/native-codex-v7-checkpoint.md) passed all
  six native phases on the same installed `6d0e5ba6` wheel as Claude v5. All
  19 MCP calls correlated, reviewed deletion and one exact source replay passed,
  controls survived, credentials stayed unchanged and cleanup passed. All 788
  private archive files verified. Neither native run supplies a pilot day.
- [Persistent pilot revision 3](../validation/2026-11-06/private-pilot-persistent-v3.md)
  installs the verified wheel in an inactive private user-profile directory.
  All 86 package files, 57 dependency versions and four configuration probes
  pass. The earlier LocalAppData path mismatch is preserved. Seven regenerated
  patches are unapplied and nine live paths remain unchanged. No service or
  database was created. All 26 archive members verified; native integration,
  rollback, quality/recovery and actual-use gates remain open.
- README and compatibility claims now identify the common installed artifact,
  the current source regression and the negative support result. All twelve
  documentation checks pass with the canonical website explicitly selected.
  These documentation changes do not publish or activate anything.
- The [native Claude pilot integration](../validation/2026-11-07/native-pilot-claude.md)
  now verifies the drafted project MCP and local hook settings on the persistent
  `6d0e5ba6` installation, using only disposable state. Two correlated native
  calls retrieved and inspected the seed; five raw captures had correct physical
  identity. One provisional prompt and the separate transcript record preserve
  the known degraded association diagnostics. The capture-only and extraction
  jobs remain pending, with one explicit seed fact and zero automatic facts.
  Cleanup, source credentials and inactive pilot files passed independent checks.
  The first pre-native shutdown failure and audit status correction are preserved.
  All 53 v2 archive members verified. Seventy focused parser/runner checks and
  static checks pass; the earlier full regression retains its own source binding.
  Interactive approval, Codex hooks, rollback, recovery, quality and actual pilot
  use remain open. No live configuration was applied and no pilot day was counted.
- The [native Codex pilot integration](../validation/2026-11-07/native-pilot-codex.md)
  passed one exec-mode session with the drafted launcher and hooks on `6d0e5ba6`.
  Two calls correlated through original Code Mode evidence. One user prompt and
  one assistant reply persisted with matching project/profile/session/turn
  identities; completed jobs produced two unasserted raw chunks and no automatic
  accepted facts. Cleanup and source-file preservation passed. All 143 private
  archive members verified. The one-run vetted-hook trust override does not
  establish ordinary interactive trust. Fifty-two focused checks pass; no
  native retry, live activation or pilot day occurred.
- The [configuration rollback rehearsal](../validation/2026-11-07/pilot-config-rollback.md)
  reproduced all seven proposed byte hashes, restored two existing files and
  five original absences, and preserved deliberate drift in fourteen refusals.
  Nine live reference paths were unchanged. Eight focused checks pass; the
  first overbroad-directory preflight failure is preserved. Four temporary full
  configuration copies were removed and none entered the 17-member archive.
  Fresh activation backups, ordinary host trust, current-artifact recovery,
  quality gates and actual use remain open.
- The [native Codex pilot startup outage](../validation/2026-11-07/native-pilot-codex-outage.md)
  preserved two native messages while a held runtime lock blocked startup with
  normal auto-start enabled. An ordinary health request drained them after the
  lock was released; maintenance completed two unasserted raw chunks. Both
  original payload replays and a second restart preserved completed rows and
  the explicit control. All 141 private archive members verified, and cleanup
  passed. The role-qualification and transient PID audit failures are retained;
  no host retry occurred. Ninety-five focused checks passed. Provider outage,
  mid-write crash, ordinary trust, quality and actual pilot use remain open.
- The [installed embedding recovery check](../validation/2026-11-07/pilot-embedding-recovery.md)
  passed on `6d0e5ba6` with a typed exception injected at the real provider's
  vector boundary. Two writes stayed durably pending with refunded attempts,
  released leases and a 15-second backoff. Existing recall reported lexical-only
  degradation. A normal installed restart restored real semantic processing,
  producing one explicit fact and one unasserted raw chunk. Exact replay and
  another restart preserved rows and controls. Both service children exited
  zero; all 82 private archive members verified. Twenty-two focused checks,
  formatting and strict types passed. The initial audit's evidence-column error
  is preserved. No native host, natural provider failure, quality acceptance,
  live activation or pilot day is implied.
- A [contrastive support candidate](2026-11-07-answer-support-contrast-v6.md)
  now places the visible candidates in one comparison batch with explicit
  rejection choices. Ninety-six focused checks, formatting and strict types
  pass. Offline guarded prompt construction peaks at 4,861 bytes, within the
  unchanged 5,120-byte limit. The [preparation receipt](../validation/2026-11-07/answer-support-contrast-v6-preparation.json)
  binds the unchanged datasets and prospective comparison. This is development
  preparation, with no new quality score, installed artifact or pilot claim.
- The [completed contrastive comparison](../validation/2026-11-07/answer-support-contrast-v6.md)
  passed all source, snapshot, request and grade audits but failed the quality
  gate. The hybrid candidate returned 27/27 clean answerable results, 12/17
  abstentions and five unsupported facts, versus 26/27, 13/17 and seven for
  joint support. Median support latency rose from 2,401 to 2,736 ms. All 486
  responses and 352 decisions verified; all 215 private archive members passed
  readback. The owned runtime unloaded and stopped. The candidate stays
  experimental; no fresh evaluation, pilot day or installed policy is implied.
- The [native local context measurement](../validation/2026-11-07/context-density-v1.md)
  verified 88 distinct literal texts with repeated Qwen token IDs and exact
  round trips. Its 4,224 packing rows and 96 groups were independently rebuilt;
  2,112 injected failures matched byte fallback. Hybrid returned context fell
  from 1,978 unfiltered tokens to 767 after contrastive selection, retaining
  27/27 expected facts but five unsupported facts. Selection overhead remains
  separate, and small byte budgets showed severe recall loss despite higher
  density. Seventeen focused checks passed. All 26 corrected archive members
  verified; the original packaging failure and cleanup warning remain retained.
  Both endpoints closed and all owned processes were absent. Native Codex and
  Claude counters, span compression, fresh quality and pilot use remain open.
- The [isolated extraction check](../validation/2026-11-07/isolated-extraction-v1.md)
  matched 14/15 quality cases and passed a separate pre-provider scope guard.
  Nine emitted proposals matched annotations, but only eight of nine critical
  facts survived: a correction carried a revision flag forbidden for its kind.
  The decoder schema allowed that combination and the runtime rejected it.
  All 25 responses, 132 source hashes and successful grades verified; all 156
  private archive members passed readback. Seventy-seven focused checks passed.
  The run performed no retrieval or accepted-fact commits. Model and owned
  runtime cleanup passed. The failure is frozen before a schema repair; fresh
  evaluation and actual pilot use remain open.
- The [extraction schema follow-up](../validation/2026-11-07/isolated-extraction-v2.md)
  prevents the invalid kind/revision-flag combination while preserving runtime
  guards. Ten failing regressions now pass; all 98 focused checks pass. The
  repeated 16-case measurement has no operational failures but remains at
  14/15 exact quality cases and 8/9 critical facts: the known correction now
  reaches reconciliation and becomes an unresolved review despite its visible
  target. All 26 replies, 132 source hashes and 16 identical input pairs verified;
  all 160 private archive members passed readback. Both endpoints closed and
  all listed processes were absent after the original tree-stop warning.
  Target selection, fresh quality, persisted extraction and pilot use remain open.
- The [four-arm reconciliation comparison](../validation/2026-11-07/reconciliation-v1.md)
  completed 240 observations across 20 preclassified cases and three repeats.
  Direct instructions fixed all 12 matching-target trials but made nine wrong
  corrections under ambiguous or mismatched qualifiers. Direct/full and legacy
  each scored 48/60; direct/relevant scored 45/60. Removing the irrelevant flag
  alone changed no case outcome. No variant met the frozen selection rule.
  All raw decisions, 133 source hashes, paired inputs and schemas verified;
  all 165 private archive members passed readback. Sixty-four focused checks,
  formatting and strict types passed. The model unloaded and owned processes
  stopped. Defaults, the installed pilot artifact and prior scores stay unchanged.
- The [duplicate-identity repair](../validation/2026-11-07/duplicate-identity-v1.md)
  enforces unresolved review when a direct user candidate exactly matches
  multiple fact identities. Both decoder and runtime reject alternative
  operations; the judged path skips unnecessary model calls. Fourteen initial
  and six judged-path regressions are repaired; all 142 focused checks pass.
  Replaying the 240 frozen choices rejects the six duplicate disagreements and
  preserves the other 234 outcomes, with zero model calls and no score changes.
  All 13 private repair archive members verified. Semantic target ambiguity,
  fresh evaluation, installed delivery and actual pilot use remain open.
- The [identity comparison](../validation/2026-11-07/identity-v2.md) completed
  180 observations across three arms on the same 20 cases and three repeats.
  Identity recovered six matching-correction trials with no wrong-target
  corrections, but scored 45/60 with six quote failures, six missed valid
  targets and three incorrect requirement additions. Legacy scored 48/60;
  direct scored 51/60 with nine wrong corrections. No candidate qualifies.
  All 171 responses, 134 source hashes, 51 identity comparisons and 180 grades
  verified; all 164 private archive members passed readback. The duplicate
  support repair now has bounded real-model evidence. All 132 focused checks,
  formatting and strict types passed. Model and owned-runtime cleanup passed;
  defaults, installed artifact, earlier scores and the inactive pilot remain
  unchanged. Quote constraints alone would not resolve the semantic failures.
- The [route-controlled reasoning screen](../validation/2026-11-07/reasoning-v3.md)
  completed 60 observations: generate/off and chat/off each scored 17/20,
  while chat/on scored 15/20 with four wrong corrections. All responses were
  valid and stopped normally; thinking emission was confirmed in all enabled
  calls and absent in both controls. Median chat latency rose from 1,405 to
  12,380 ms and generated tokens from 1,112 to 14,126. The candidate fails its
  confirmation gate. All 134 source hashes, 20 identical input/prompt/schema
  groups, request options and final scores verified; all 166 archive members
  passed readback. All 107 focused tests, formatting and strict types passed.
  Model and owned-runtime cleanup passed. Final responses sometimes describe
  mismatched targets yet choose correction; decoder effects need investigation
  before assuming more reasoning or another prompt decomposition will help.
- The [decoder screen](../validation/2026-11-07/decoder-v4.md) completed 80
  observations. Reversing decision branches preserved all 20 operation/target
  outcomes: both bounded arms scored 17/20 with three wrong corrections.
  JSON-only and unconstrained output each scored 10/20, kept the same three
  wrong targets and added seven runtime rejections. No variant qualifies for
  confirmation. All 80 scores and offline response replays, 135 source hashes,
  paired inputs and outbound formats verified; all 165 private archive members
  passed readback. The 89 focused and 12 documentation checks, formatting and
  strict types passed. Model and owned-runtime cleanup passed, with the original
  tree-stop warning retained. These decoder changes did not resolve the known
  semantic identity errors. Defaults, installed pilot state and prior scores
  remain unchanged; fresh quality, persisted extraction and actual use stay open.
- The [value-span referent screen](../validation/2026-11-07/referent-v5.md)
  completed 60 observations. The new contract scored 19/20 with 9/9 review
  controls, no wrong accepted targets or additions and no operational failures.
  It missed the known single-target correction and fails confirmation. Direct
  scored 17/20; quoted identity scored 15/20 with two quote failures and one
  wrong addition. A post-run inspection found seven incorrect intermediate
  relations among 17 comparisons, six hidden by correct final review outcomes.
  All 60 scores and runtime replays, 136 source hashes, 20 paired inputs and
  29 value fields verified; all 176 archive members passed readback. Preflight
  also preserved all 60 earlier identity outcomes. The 159 focused and 12
  documentation checks, formatting and strict types passed. Model and owned
  runtime cleanup passed. Defaults and installed pilot state remain unchanged;
  intermediate relation quality, fresh evaluation and actual use remain open.
- The [relation and consistency diagnostic](../validation/2026-11-07/relations-v6.md)
  completed 72 matched views of twelve seed scenarios in five clusters. The
  unchanged comparator produced 49 correct operations, five wrong corrections,
  eleven wrong additions and 110/198 correct relations, with no operational
  failures. Only two seeds were correct at both layers in all six views; three
  stayed consistent. ID-only changes altered seven of 36 paired operations.
  All 72 scores and replays, 138 source hashes, 198 relations and twelve groups
  verified; all 175 archive members passed readback. The 183 focused and 12
  documentation checks passed, including 24 new checks. Cleanup passed.
- A [local Qwen3.5 candidate](../validation/2026-11-07/qwen35-local-setup.md)
  downloaded and passed zero-generation loading at context 8192. The package's
  original blobs and appended manifest reference verified; two earlier manifest
  verifier failures remain preserved. Its separate archive has 24 verified
  members, and all setup processes stopped. It has no semantic-quality result;
  sampling defaults and native tokenizer compatibility must be bound before a
  model comparison. Production defaults and the inactive pilot remain unchanged.
- The [matched model comparison](../validation/2026-11-07/model-v7.md) completed
  the original 20 cases and 72 diagnostic views for each model with explicit
  sampling controls. Qwen3 scored 18/20 and Qwen3.5 scored 17/20. On the diagnostic,
  they scored 49/72 and 52/72 operations, but accepted five and eleven wrong
  correction targets respectively. Combined operation/relation scores were 33/72
  and 40/72. Neither model passes its gate; no confirmation or adoption followed.
  All 184 observations were retained and audited. The 228 checks and cleanup
  passed; the archive has 204 verified members. Earlier scores remain unchanged.
- The [compact transport measurement](../validation/2026-11-07/compact-v1.md)
  preserved all 36 returned fact occurrences across 18 paired retrievals. Actual
  compact reply text was 55.82% smaller across six cases; inspecting every
  reference made the combined reply larger than full recall. The opt-in format
  keeps literal context, scope, versions, status and source references. It does
  not change retrieval or summarize away qualifications. Sixty-four checks passed.
- The [native compact screen](../validation/2026-11-07/native-compact-v1.md) then
  passed all eight typed answer fields in six fresh Codex sessions. Complete
  context reached the model every time, and compact MCP replies were 58.67%
  smaller for this fixture. Total input tokens fell in all three pairs, but
  discovery steps, cache behavior and output formatting also varied. This is
  bounded native-consumption evidence, not a general token-saving percentage or
  fresh retrieval-quality pass. All 122 checks, the independent trace/input audit
  and cleanup passed. The archive has 338 verified members. The pilot remains
  inactive with zero days; live configuration and publication remain unchanged.
- The [isolated pairwise comparison](../validation/2026-11-07/pairwise-v8.md)
  sends each literal pair with a constant local identity while the runtime keeps
  actual storage IDs, versions and scope. All 198 planned requests are invariant
  to ID aliases, order and distractors. Qwen3 improved to 57/72 operations and
  144/198 relations; Qwen3.5 improved to 66/72 and 167/198. Three and six wrong
  corrections remain. Qwen3.5's six wrong corrections all guess the production
  environment from an unqualified requirement. Each model required 198 requests
  instead of 72 and nearly doubled total input tokens. Neither passes the gate.
  All 144 observations, 396 quality responses and 140 source hashes were audited.
  The 187 checks and cleanup passed; the archive has 191 verified members. No
  confirmation, promotion, fresh-quality pass or pilot day follows from this run.
- The [identity-span screen](../validation/2026-11-07/spans-v9.md) rejected a
  three-dimension witness contract. Qwen3.5 fell from 66/72 to 30/72 operations,
  with ten wrong additions and twelve missing-witness failures. It made no
  corrections, including all thirty required corrections. Valid literal spans
  still mislabeled changed values as different attributes. All 172 responses,
  72 runtime replays and 141 source hashes were audited; 26 unrequested pairs
  remained in the denominator. The 219 checks and cleanup passed, and the
  174-member archive verified. V8 remains the stronger experimental baseline;
  no quality gate, confirmation, default change or pilot day is implied.
- The [pilot entry audit](../validation/2026-11-07/pilot-entry-audit-v1.md)
  verifies all 86 installed files, 57 dependency versions, four configuration
  probes and the seven unapplied patch preconditions. The prepared pilot uses
  explicit writes with model extraction disabled. Failed automatic comparators
  remain experimental; their path does not execute in this configuration.
  Retrieval quality still needs separate fresh evaluation on accepted facts.
  The original thresholds remain, normal host trust is unverified, and the pilot
  has zero days. All thirteen watched paths stayed unchanged; no service started.
- The [749f1213 checkpoint](../validation/2026-11-07/checkpoint-749f1213.md)
  passes the full suite with 3,221 tests and four platform skips, plus build,
  Twine and installed CLI/MCP restart-recall checks using real embeddings.
  Ninety installed package files match the new wheel; 83 sources also match
  the source archive and checkout. The compact response option is available
  with full responses still the default. All 22 archived members verify. The
  pilot retains its older native-tested artifact; no live upgrade or release
  occurred. Neither checkpoint supplies fresh quality or actual daily use.
- The [accepted-state retrieval screen](../validation/2026-11-07/accepted-retrieval-v1.md)
  adds a benchmark input path that never runs extraction or maintenance. It
  validates declared versions, scope partitions, source excerpts and timestamps
  before embedding, then clones the same database for vector, lexical and hybrid
  retrieval. All 65 checks pass, including 36 new cases. The local BGE screen
  completed 63 observations: every arm found 12/12 expected answers but returned
  nonempty results on all nine repeated no-answer checks. Scope leaks and runtime
  failures were zero. All 27 embedding calls, 63 grades and 21 stable groups were
  independently audited, and 29 archive members verified. These are seven
  development questions; no fresh evaluation or pilot readiness is implied.
- The [first fresh accepted-state comparison](../validation/2026-11-07/fresh-accepted-v1.md)
  completed all 5,220 observations. Vector and hybrid found 600/600 expected
  ordinary answers; lexical found 561/600. Each strategy had 300/600 clean hits
  and 150/660 correct abstentions. Every correct abstention came from an empty
  eligible pool; all 510 repeated no-answer observations with available facts
  returned nonrelevant context. The original 90% abstention threshold fails.
  Independent audits checked 152 input bindings, 2,440 role vectors, 200 source
  databases, all raw grades and intervals, and all 700 stored document vectors
  and SQLite vector-index entries. Both archives verified; no scope leak,
  stale return or runtime failure was observed. This evaluates supplied gold
  states with synthetic labels, not extraction or lifecycle execution. No
  human acceptance, full-module benefit, release or pilot readiness is implied.

- The [installed performance check](../validation/2026-11-07/installed-performance-v3.md)
  completed all 155 observations, with 50 measured attempts per path. Warm
  recall passes its 500 ms requirement at 403.37 ms p95. Complete capture and
  prompt hooks fail the unchanged 100 ms requirement at 2089.01 and 2103.02 ms.
  All 102 captured sources and durable jobs survive restart; 100 background
  jobs completed and two remain pending. The audit verified all 10,000 real
  fixture vectors, their index entries, 104 file bindings and 17 model assets.
  The 125-member private archive verifies. Both owned services exited cleanly.
  These installed adapter checks use synthetic payloads; native-host latency,
  semantic quality and actual daily use remain separate obligations.
- The [hook startup diagnosis](../validation/2026-11-07/hook-startup-diagnostics-v1.md)
  identifies configuration imports as a substantial cost before dispatch.
  Retrieval settings now live separately from memory and cognitive-state models,
  with the existing import path preserved. The three new boundary checks fail
  before the change; 37 configuration/retrieval and 68 adapter/client/service
  checks pass afterward, with three shared cases. The candidate wheel at
  `3e137634` builds and passes Twine; installed files match the wheel and source.
  The [forty-observation comparison](../validation/2026-11-07/hook-import-comparison-v1.md)
  finds a -24.12 ms median paired empty-hook difference, with one much slower
  candidate sample. Empty-command startup still has a 689.58 ms median. The
  full hook gate remains open; no native acceptance or pilot upgrade occurred.
- The [support composition runtime](../validation/2026-11-07/support-intersection-runtime-v1.md)
  binds two selectors to one immutable cohort and rechecks the entire view around
  provider awaits, including individual QA source calls. Forty-three new checks
  cover source ownership, stale replies, correction/forget races, shared deadlines,
  cancellation, failure status and unchanged component inputs; 118 related checks
  also pass. A model-free replay through the real adapters exactly reproduces
  all 44 saved intersections using 78 QA outputs and 44 contrast responses.
  Eighty-six file bindings and all 145 archive members verify. The existing
  three failed questions remain failures. Actual combined inference, latency,
  fresh support acceptance and pilot use remain open.
- The [live composition](../validation/2026-11-07/support-intersection-live-v1.md)
  completed three new repetitions of all 44 known questions. All component and
  intersection selections match the original diagnostic: 25/27 clean answers,
  16/17 abstentions and one unsupported selection in each repetition. All 234
  QA source checks and 132 contrast requests completed without retry or failure.
  Support alone takes 2,990.80 ms median and 3,984.61 ms p95, failing the unchanged
  500 ms recall requirement. The audit reconstructs raw selections and verifies
  105 bindings, 16 model/runtime assets and 192 archive members. The worker and
  private model server are stopped. Fresh quality, a faster qualifying path,
  current installed/native validation and the actual pilot remain open.
- The [resident hook worker](../validation/2026-11-07/resident-hook-worker-v1.md)
  keeps the existing adapters and durable outbox behind a small command client.
  It reloads membership, detects settings/environment/directory changes, renews
  startup and replay checks per event, and refuses blind retry after an ambiguous
  response. All 21 new checks and 153 related regressions pass. The `feb6ac3b`
  wheel builds and passes Twine; 94 installed files and the command import boundary
  verify. This opt-in path is not installed in a live host. Complete installed
  timing with real embeddings and the original launcher checks is the next gate;
  the failed 100 ms hook results and zero-day pilot remain unchanged.
- The [complete scoped ranking comparison](../validation/2026-11-07/scoped-rank-performance-v1.md)
  retains 320 observations and 160 identical response/context pairs. Recall p95
  falls from 311.35 to 134.88 ms. All 208 captures survive restart, and their
  jobs complete without adding accepted facts. Complete Stop and prompt hooks
  still fail 100 ms at 547.11 and 860.06 ms. The independent audit and archive
  verification pass; all owned processes stopped.
- The [startup diagnostic](../validation/2026-11-07/hook-startup-v1.md) retains
  88 observations. Skipping site initialization saves only 13.72 ms per matched
  client command at the median, leaves about 114 ms of client imports, and
  omits normal virtual environment initialization. Separate import traces point
  to the HTTP/email parser dependency chain and package initialization. The
  audit passes; no launcher or product change follows from this small gain.
- The [Windows transport comparison](../validation/2026-11-07/native-hook-performance-v1.md)
  replaces the HTTP/email parser import chain with a bounded system WinHTTP
  request. Forty new checks and 48 related checks pass. The installed comparison
  keeps all 160 response/context pairs identical and all 208 captures durable.
  Stop median improves from 402.43 to 324.19 ms, but prompt submission shows no
  consistent gain and both complete hook paths still fail 100 ms. The audit
  correction handles Windows process ID reuse without changing measured data.
  Measured source `788766aa` remains frozen. Portability follow-up `cddf8adc`
  passes 62 focused checks, strict type checking for both platforms, a verified
  wheel install and one real Windows request. No live host or pilot changed.
- The [full regression checkpoint](../validation/2026-11-07/full-regression-native-hook-v1.md)
  passes 3,495 tests with four WSL/POSIX skips at `9c84909f`. No failure or error
  occurred. The independent audit confirms all counts, unchanged tracked source,
  the selected website file and 208 historical fixtures, and stopped test
  processes. This updates source-wide regression evidence without extending
  the older native-host results or counting any pilot days.
- The [current native Codex lifecycle](../validation/2026-11-07/native-codex-v8.md)
  passes all six phases on `cddf8adc`, with 19 correlated calls, completed original
  jobs, same-ID correction, reviewed deletion and one suppressed source replay.
  The independent audit verifies all calls, retained controls and database state.
  The full prospective protocol remains incomplete because filtered discovery
  records do not prove the complete advertised tool set. This is an observation
  gap to address explicitly in the next validation. The current wheel also
  passes full installed CLI/MCP restart recall; no actual pilot days accrued.
- [Codex v9](../validation/2026-11-07/native-codex-v9.md) adds a complete
  inventory observation with 34 new checks and 179 focused passes. Its fresh
  native attempt fails during first restart recall: `fact_id` is rejected by
  MCP inspection, which requires `chunk_id`. The host's corrected call succeeds,
  but the strict failed-response rejection remains. Four later phases are unrun;
  the failure and cleanup are archived. Product `acc7a317` adds `fact_id` as a
  compatible alias, rejects conflicting IDs and retains host-resolved scope.
  All 39 focused MCP checks, strict type checking and formatting pass. This
  product change still needs installed and native validation; the earlier full
  suite remains bound to its original source. The pilot remains inactive.
- [Codex v10](../validation/2026-11-07/native-codex-v10.md) verifies the new
  `acc7a317` wheel in an isolated installation and passes four native phases
  with 12 correlated calls and complete inventories. Three native inspections
  use `fact_id` successfully. Forgetting fails after two preview requests omit
  `expected_version`, including one failed code cell. The host subsequently
  supplies version two and commits deletion, but strict acceptance remains
  failed; final restart and source replay are unrun. The audit confirms the
  argument errors, retained controls and database absence, preserving its
  initial Windows newline-accounting failure and correction. All 732 archive
  files verify, and all ten recorded process identities are absent. Improve
  discovery of the required preview version without removing its stale-state
  guard or changing this result. No live configuration or pilot day changed.
- [Codex v11](../validation/2026-11-07/native-codex-v11.md) verifies the revised
  shipped tool-discovery and versioned-preview guidance on product `c80eb051`.
  Ninety-five focused checks pass, with two POSIX skips. The fresh installed
  wheel matches its source and dependencies. All six native phases pass with
  19 correlated calls, complete inventories, confirmed installed instructions
  and unchanged requests. The independent audit verifies one suppressed source
  replay, final absence and all retained controls. All 881 archive members
  verify; all 13 recorded process identities have exited. This closes the
  current synthetic native lifecycle gate. Full regression, canonical delivery,
  remaining pilot entry checks and actual daily use remain open.

- [Preview-guidance full regression](../validation/2026-11-07/full-regression-preview-guidance-v1.md) retains its original failure: 3,547 passes, one atomic descriptor replacement failure and four platform skips. A held reader reproduces the same Windows error; delete sharing alone is insufficient. [Descriptor publication fix](../validation/2026-11-07/descriptor-sharing-v1.md) adds an extended atomic rename fallback for cooperating readers while preserving ACLs and bounded retries. All 120 focused checks pass, including 12 new cases, with Windows/Linux strict type checks and formatting. The new artifact also passes installed CLI/MCP restart/recall with real BGE embeddings; 96 package files, 85 committed Python files and 57 dependency versions match. All 17 cached model assets are unchanged. Full regression on 0456cb5d passes 3,560 cases with four platform skips; the earlier native v11 pass retains its c80eb051 binding.

- [Full regression after descriptor repair](../validation/2026-11-07/full-regression-descriptor-sharing-v1.md) passes 3,560 cases with four platform skips on 0456cb5d, with the source, canonical website and 208 historical fixtures unchanged. The independent audit confirms recorded process exit and verifies the private archive. [Updated delivery review](../validation/2026-11-07/delivery-review-v2.md) reconstructs this tested tree from preserved base 9770c009 in a private index, with the verified current wheel. Real indexes and branch heads remain unchanged. The bundle is prepared, not applied; later reports and a source archive remain for final integration. Quality, full hook latency, canonical/site delivery and actual private pilot work remain open.

- The [sparse choice comparison](../validation/2026-11-07/sparse-choice-support-v1.md) completes 272 observations across 44 known and 24 new development questions, two arms and two repetitions. Sparse reduces median support time from 2,610.06 to 693.29 ms and generated tokens from 143 to 33, but fails quality with 72/136 contract failures and 6/80 answerable hits. Its 1,342.82 ms p95 still exceeds the 500 ms recall budget before retrieval. All contract failures contain forbidden explicit zero entries; among originally valid replies only 6/47 answerable cases hit, so relaxing that constraint would leave severe semantic misses. The independent audit verifies 174 bound files, eight model/runtime assets, 68 stores and all 127 fact/evidence pairs. All processes exit and assets remain unchanged. The 425-member private archive preserves raw evidence, initial setup failures and the offline diagnostic. The candidate remains disabled; any follow-up must preserve named option identity and isolate serialization changes. The 90 focused checks, strict typing and formatting pass; production checkpoint 3d1866d5, earlier full regression and native acceptance retain their original bindings. Fresh quality, complete latency, delivery and actual pilot use remain open.

- The [native command-boundary probe](../validation/2026-11-07/native-hook-launcher-v1.md) completes 88 commands and 44 authenticated loopback dispatches on the installed 3d1866d5 product and a separate compiled client. Native dispatch median/p95 is 31.86/36.46 ms versus Python 171.40/214.47 ms; every decoded output matches. Empty native commands retain a 140.08 ms p95 outlier, so there is no general startup bound. The responder is a protocol fixture, not the memory service: no scope lookup, capture transaction, retrieval or inference runs. All 112 subprocess/related checks and two total-deadline checks pass. A large-integer precision regression was reproduced and corrected, with the original binary and failures retained. The independent audit verifies 107 file bindings, all 88 command exits, responder shutdown and descriptor removal; the 128-member private archive verifies. Before complete installed comparison, integrate the private launcher's environment setup and guarded fallback. Then use the unchanged 10,000-fact seed, real embeddings, original worker checks and complete output without relaxing the latency gates. Production, live configuration and the inactive zero-day pilot remain unchanged.

- The [complete native comparison](../validation/2026-11-07/native-launcher-performance-v1.md) retains 320 observations, 160 exact output/context pairs and all 104 failed native routes. The initial success-only auditor stopped at the missing dispatch file; its failure is retained. The revised audit verifies 104 baseline dispatches, 208 durable captures and completed jobs, unchanged accepted data and cleanup of all 342 recorded process IDs. The verified private archive contains 250 members and has SHA-256 `0001291717adcc2396b0967dae86039243e5fe90273b2d209d6384a234474d85`. A model-free observer finds a Windows hidden drive entry in the native environment that Python omits. Four inherited-environment regressions fail on the original executable, while two ordinary-mismatch controls pass. The diagnostic reproduces a decline mechanism; it does not reconstruct the original environment exactly or prove that this was its only difference. A narrow Windows conversion fix and a new prospective comparison preserve the original failure and all original gates.

- The [corrected complete native comparison](../validation/2026-11-07/native-launcher-performance-v2.md) binds source 08a1d61e and the unchanged installed 3d1866d5 product. Windows hidden drive entries are excluded to match Python; ordinary environment mismatches still decline. Four original regression failures and two passing controls are preserved; the corrected candidate passes all 137 subprocess/related checks, two deadline checks, strict Clippy/formatting, 12 documentation checks and 12 exact private guard pairs. The separate frozen run completes all 320 observations and 208 handled worker dispatches with identical paired output, durable captures, completed jobs and unchanged accepted facts/evidence/vectors. Native median paired improvements are 152.20 ms for Stop and 167.83 ms for prompt submission, while complete p95 still fails 100 ms. The independent audit verifies the raw evidence and cleanup; the private archive has SHA-256 `731606d75a70bd13391f4cb4993370c1f4a285f25b7e763a248e5d3f536bfbf5`. The next latency step is diagnosis of the remaining current-worker costs, not a relaxed gate or live activation. The private pilot remains inactive at zero days.

- The [current native-worker diagnosis](../validation/2026-11-07/native-worker-diagnostics-v1.md) completes 20 instrumented observations on the unchanged installed 3d1866d5 product and corrected native launcher. All outputs match the corresponding prior complete-run outputs. The independent audit verifies 12 worker dispatches, 18 service dispatches, six real prompt embeddings, 12 captures surviving restart, 12 completed jobs and unchanged accepted data. Git resolution median is 61.00 ms for Stop and 70.08 ms for prompt submission, nearly all in its fresh subprocess. Service retrieval median is 93.53 ms: vector/lexical ranking is 32.04/30.68 ms, fact-evidence lookup totals 11.32 ms, and real query embedding is 4.56 ms. Nested timings overlap and do not replace the failed complete latency gate. Eight tracing-contract checks and entry-script preflight pass. The verified 162-member private archive has SHA-256 `097bbaccc6d31b5a569976d49671eb00b9d13b3c2ff5f85ee7141539b0a1ae27`. Next compare Git executable paths on real repositories/worktrees and probe an evidence-lookup index on copied data, preserving fresh identity checks and exact output. Production, live configuration and the zero-day pilot remain unchanged.

- The [latency component probes](../validation/2026-11-07/latency-component-probes-v1.md) retain 114 Git resolver calls and 740 evidence lookups without model calls or services. All 57 Git behavior pairs and 370 evidence pairs match exactly; all 116 measured Git processes exit. Direct Git saves roughly 17-20 ms in median paired resolver time across five fixture types, but ordinary-repository p95 worsens and product Git selection stays unchanged. The evidence index reduces present/absent median lookup from 1.901/1.812 to 0.043/0.023 ms with unchanged accepted data, tie order and post-index inserts. The independent audit and 52-member private archive verify. Product f5a944ea adds the same index in transactional schema setup; three prior regression failures are preserved, and 120 focused tests, strict mypy and Black pass. The new wheel matches all 85 committed Python files and changes only storage.py among 96 package files; Twine passes. Isolated installation, fresh full regression and complete installed comparison remain pending, with the original 100 ms gate intact.

- The [new installed artifact](../validation/2026-11-07/fact-evidence-index-install-v1.md) passes all four preparation stages. All 96 package files, 85 committed Python files, 57 dependency versions and 17 model assets verify. Indexed lookup retains equal-time evidence order across restart. Fresh full regression and the original complete-hook latency gate remain open.

- The [evidence-index full regression](../validation/2026-11-07/full-regression-fact-evidence-index-v1.md) passes 3,674 tests with 4 platform skips on 1299ed9b. All 75 native-launcher subprocess checks run and pass; source, website, 208 fixtures and the executable remain unchanged. The complete installed comparison is frozen with the same native launcher in both arms, changing only the package version.

- The [complete evidence-index comparison](../validation/2026-11-07/fact-evidence-index-performance-v1.md) changes only the installed package while retaining the native launcher in both arms. All 320 observations, 160 matching output/context pairs and 208 restart-surviving captures are audited. Candidate p95 gates: Stop: FAIL; prompt submission: FAIL; recall: PASS. Accepted data, artifact/model bindings and process cleanup verify; prior evidence is preserved.

- The requested [adversarial review](../validation/2026-11-07/adversarial-goal-review-v1.md)
  found that delivery and pilot convergence had fallen behind experiments.
  The final [named-map comparison](../validation/2026-11-07/named-choice-support-v1.md)
  completed 272 observations and passed the evidence audit, but selected 41
  unsupported facts against the baseline's 14. Its 3,209.62 ms support-only p95
  still fails the recall budget. The codec remains disabled and this series is
  closed. The revised work order freezes product f5a944ea, gives the small
  storage backport independent delivery priority and caps the next quality
  increment at two development runs. Existing gate failures remain unchanged.

- The [backport package check](../validation/2026-11-07/storage-backport-package-v1.md)
  builds the wheel from the source archive on unchanged d8a5e647 and verifies
  all 35 source/installed package files. Five installed graph regressions,
  CLI help, Twine and dependency checks pass. The refreshed 25-member review
  archive retains the original patch and baseline typing failures. Its 0.6.0
  verification artifacts are not a new release and must not be uploaded.

- The [current delivery bundle](../validation/2026-11-07/delivery-review-v3.md)
  reconstructs fb062fe3 from preserved delivery base 9770c009 in a private
  index. It includes the unchanged f5a944ea wheel and a new source archive
  reproducing all 101 wheel members. The first build's newline-only mismatch
  and its correction remain recorded. All 14 archive members verify. Real
  delivery/site checkouts, public branches and live host settings stay unchanged.
  The read-only follow-up review found no material plan findings; it emphasized
  that experimental support results cannot qualify baseline pilot retrieval.

- The [refreshed private pilot](../validation/2026-11-07/persistent-pilot-v4.md)
  now uses the frozen f5a944ea wheel in a separate inactive installation. Its
  96 package files, 57 dependency versions and four configuration probes pass.
  All seven current host patches pass exact rollback and fourteen drift refusals
  on disposable copies. Nine live paths and the older inactive pilot stay
  unchanged. Normal hook trust requires the user's native review flow; the
  separately frozen v12 attempt covers explicit MCP lifecycle with hooks off.

- The [current-wheel native Codex attempt](../validation/2026-11-07/native-codex-v12.md)
  passes all six phases at f5a944ea with 19 correlated calls, six complete
  inventories and installed guidance delivery. Same-ID correction, reviewed
  forgetting, one suppressed original-source replay and three unchanged controls
  verify independently. All 13 recorded process identities exit, copied
  credentials/descriptors are removed, and the pilot remains inactive. The
  810-member archive binds this result. Hooks were disabled, so normal trust,
  capture and full pilot configuration remain separate requirements.

- The [normal hook review preparation](../validation/2026-11-07/native-codex-hook-review-v1.md)
  verifies six exact native definitions and passes the installed check-only
  launcher. All six remain untrusted pending the user's `/hooks` review.
  The 39 focused checks and three metadata-only native processes do not
  establish hook execution or activate the pilot.

- The [bounded installed retrieval/support run](../validation/2026-11-07/retrieval-support-development-v1.md)
  completes and passes its read-only audit. All 272 database copies and 136
  candidate/context pairs match. Treatment abstention improves from 0/56 to
  52/56 and unsupported selections fall from 174 to four, but it rejects eight
  expected answers. Precision, retention and complete latency fail. The
  499-member private archive preserves original preparation failure and all
  raw outputs; one measured run, no integration rerun or fresh acceptance.
  The next independent work is installed-pilot timing and normal hook
  validation. Any MCP-only learning pilot requires a separate prospective
  decision and cannot close the original pilot or semantic-quality obligations.

- The [installed pilot command comparison](../validation/2026-11-07/installed-pilot-command-timing-v1.md) measures the actual proposed Python subprocess and experimental native v3 command on installed f5. All 320 observations, 160 matching output/context pairs and 208 transaction spans and restart-surviving captures are audited. Gate results: recall: Python FAIL, native FAIL; Stop: Python FAIL, native FAIL; prompt: Python FAIL, native FAIL. Internal persistence metrics remain separate diagnostics. The pilot stays inactive at zero days; prior failures are unchanged.


- The [optional learning-pilot proposal](../validation/2026-11-07/explicit-learning-pilot-v1.md)
  contains four private project-local patches, twelve allowed tools and no capture
  hooks. Metadata parsing and twelve rollback checks pass; fourteen live paths
  remain unchanged. Scope and activation are unapproved. It cannot replace the
  original quality acceptance or two-week pilot, and both pilot clocks remain zero.

- The [0.6.1 storage release candidate](../validation/2026-11-07/storage-release-package-v1.md)
  passes 316 source tests, five installed regressions, full local typing, formatting
  and package checks. All 35 package files match source, archive, wheel and installed
  bytes; the exact review patch reconstructs the candidate tree. The preserved legacy
  benchmark fails two in-memory contradiction comparisons, with ten of twelve pairs
  passing. The 56-member private archive includes both earlier setup failures, the
  completed benchmark and review drafts. No publication or live activation occurred.


- The [legacy recency diagnosis](../validation/2026-11-07/legacy-recency-diagnostic-v1.md)
  reproduces the timestamp tie behind a stale deployment answer and shows that
  accessing an old fact makes it appear current. All 24 controlled query pairs
  match across stores with identical content, embeddings and scope. Ordered
  timestamps yield 8/8 expected answers; prior stale access reduces both stores
  to 0/8. This is diagnostic evidence only. The relevant ranking code predates
  the patch, the original failed gate remains unchanged and no production
  retrieval or benchmark repair has been folded into the storage candidate.


- The [local canonical integration](../validation/2026-11-07/canonical-integration-v1.md)
  passes 3,758 tests with four platform skips, all 75 native-launcher cases,
  strict typing across 228 source files and Python formatting. Desktop/mobile
  browser checks and six exact receipt downloads pass. Both local canonical
  checkouts are fast-forwarded after exact patch reconstruction and clean-tree
  checks. The original wheel, fourteen watched live paths and failed acceptance
  gates are unchanged. Publication, full native configuration and actual pilot
  use remain open; this does not repeat native-host or semantic acceptance.

- The separate [legacy content-recency repair](../validation/2026-11-07/legacy-content-recency-v1.md)
  prevents prior access from promoting stale content. All 344 legacy tests,
  including 28 recency cases, pass with strict typing and formatting. Schema 3
  preserves unknown revision history and migrates copied rows, vectors, graph
  and feedback controls without loss. The 0.6.1 candidate and frozen f5 product
  remain unchanged. Original scores are preserved; installed-package validation
  and a prospective chronology benchmark for this repair remain open.

- The [installed recency package](../validation/2026-11-07/legacy-content-package-v1.md)
  is separately identified as unpublished 0.6.2. All 35 package files match the
  source, archive, wheel and clean installation. Its 56 metadata/CLI/docs checks
  and 41 installed regressions pass. Two additional copied-database migration
  checks pass with graph and feedback controls. The original 0.6.1 and f5 artifacts
  are unchanged. No benchmark was rerun; chronology evaluation, native acceptance
  and release qualification remain open.
