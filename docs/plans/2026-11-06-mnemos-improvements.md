# Mnemos improvement work

Status: active. Started November 6, 2026 from delivery commit `9770c00` on
`wt/mnemos-improvements`. This goal resumes the work deferred on October 28.
No release, live configuration change, or pilot completion is implied.

## Scope and completion evidence

| Priority | Work | Completion evidence | Current state |
| --- | --- | --- | --- |
| 1 | Repair missing-source graph writes and transaction recovery; prepare the affected-user fix | Regression tests, separate-process client coverage, rollback checks, reviewed delivery patch | Implemented and locally tested; public history integrated locally at 7fcca905; minimal backport d8a5e647 passes all 316 public-branch tests; review package prepared, publication pending |
| 2 | Add evidence-backed answer support and abstention | Development comparison and fresh evaluation with false-answer, supported-answer, conflict, and latency results | V5 completed 352 observations: hybrid hits were 27/27 for both treatments, but the candidate reduced clean hits from 26 to 19 and abstentions from 14/17 to 13/17; four candidate contract failures; no promotion; further development and fresh evaluation open |
| 3 | Separate extraction from retrieval evaluation | Identical accepted-fact snapshots for retrieval arms; frozen prospective rubric and dataset identities; independently reported extraction and ranking outcomes | Accepted-fact development comparison completed; scope-fixture defect repaired; separate extraction and fresh evaluation open |
| 4 | Complete explicit remember/inspect/restart/correct/forget workflow | Installed artifact, repeatable native-host acceptance, truthful job status, two-week daily-use pilot across two real projects | Codex v7 and Claude v5 pass all six phases on 6d0e5ba6; drafted pilot integrations, configuration rollback, native Codex startup recovery and injected embedding-error recovery pass bounded checks; Claude capture identity and normal Codex trust remain limited; broader recovery, quality gates and actual use open; pilot inactive |
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
