# Converge the improvement work on one candidate

The adversarial review found a real planning problem: useful repairs were spread
across several validated checkpoints, while the daily-use configuration was
old and inactive. Delivery and pilot preparation now use one frozen artifact.
The bounded retrieval/support comparison has completed and failed its gates.
The six original goals and their acceptance targets stay open.

## Candidate and delivery boundary

Freeze product `f5a944ea9a7f61499733b8b64e829aa10f5920be` for delivery preparation.
Use its existing `0.7.0b1` wheel, SHA-256
`07c5b424bc7731cacbebb45f44709b3f19509c2cde3150a6451c92d12b27e6ae`.
Later commits may add reports and benchmark code, but must not silently change
the package under that identity. The next review bundle must prove its package
tree matches this product and include the later verification reports.

This is a tested delivery candidate, not a pilot-ready configuration. Installed
CLI/MCP checks and the [native Codex v12 lifecycle](../validation/2026-11-07/native-codex-v12.md)
cover this wheel. The [refreshed inactive pilot](../validation/2026-11-07/persistent-pilot-v4.md)
uses the same artifact. Native v12 had hooks disabled; normal hook review is
[prepared and awaiting the user](../validation/2026-11-07/native-codex-hook-review-v1.md).
Earlier passes retain their original artifact bindings. The native launcher is a development
experiment and is not part of this wheel's supported installation.

Prepare and verify the small storage backport independently. Public master was
rechecked on November 7 at `866444378e5a42598bba0548f6460d07adcded10`, with PR #5
merged. Backport `d8a5e647` adds transaction rollback and caller-transaction
preservation to that guard. Keep its review focused on the three changed files.
Build a private wheel and source archive, check their metadata and source, and
run the five storage regressions from an isolated installed environment. Its
existing `0.6.0` metadata is for verification only; these artifacts cannot be
uploaded over the published version. A release needs its own version and the
remaining release checks. The four recorded baseline typing errors remain
visible. Local review preparation does not authorize a push or publication.

The [0.6.1 release preparation](../validation/2026-11-07/storage-release-package-v1.md)
now extends that backport on a separate branch at `9ea3852a`. Its four typing
errors are repaired and version metadata is aligned. All 316 source tests,
five installed storage regressions, typing, formatting and packaging pass in
the pinned local environment. The existing production benchmark fails both
in-memory contradiction comparisons; all six SQLite comparisons pass. Its
original provider/isolation failures are preserved. The original three-file
backport remains unchanged. The release candidate is not ready to tag or
publish; CI, native-host and pilot requirements also remain open. Investigate
the existing benchmark failure before deciding whether another run is useful.

The [completed diagnosis](../validation/2026-11-07/legacy-recency-diagnostic-v1.md)
reproduces tied ingestion timestamps and access being treated as content
recency. With fixed snapshots, both stores return 2/8 expected answers under
tied timestamps, 8/8 under ordered timestamps and 0/8 after prior access to old
facts. The relevant ranking logic predates the patch. Preserve the failed gate;
another unchanged benchmark run cannot repair these semantics. Any repair must
separate content revision from access and use an explicit, versioned chronology
fixture with prior-access coverage. Keep that work separate from the bounded
storage patch and the frozen f5 candidate.

## Stop the codec series

The final named-map run completed and passed its independent evidence audit:
272 observations, 68 known questions and no new acceptance cases. Named output
retained 74/80 answerable hits, produced 67/80 clean hits, abstained on 33/56
absent-answer observations and selected 41 unsupported facts. Its one contract
failure is not a successful abstention. Support-only p95 was 3,209.62 ms. The
baseline selected 14 unsupported facts and abstained on 42/56 observations.

Neither output format qualifies for promotion. No additional sparse, positional
or named-output variants are planned. Keep the codecs in the development
harness, with their original scores, raw replies and source bindings intact.

## One bounded end-to-end quality increment

The next semantic result must exercise retrieval followed by support selection
on the actual returned candidates. It must not supply the correct candidates
directly to a support-only test. Use the existing QA/contrast intersection as
the single experimental treatment; it has better recorded clean selection than
the output-format variants, but already exceeds the recall latency budget.
That limitation is known and cannot be fixed by composing it with retrieval.

Prepare one adapter and one development run after the delivery bundle is ready:

- Call `MemoryService.retrieve` with accepted-memory view, baseline profile,
  top-k five, project/global scope and a 2,048-unit context budget using the
  recorded conservative byte counter. Both arms use identical disposable
  accepted-fact snapshots, source evidence, embeddings and request settings.
- The experimental arm receives only those returned IDs. Apply the existing
  intersection and its full-cohort revision fence. Pack the selected current
  facts under the same context limit, then verify scope, source and version
  again before returning them. Provider or contract failure returns an explicit
  unavailable result with no context; never count it as correct abstention.
- Use the existing 68 known questions for development, with two paired
  repetitions. Label ranking misses separately from support rejection,
  unsupported selection and operational failure. Record complete elapsed time,
  support time, selected IDs, retained critical facts and output context bytes.
  Report per-question results and uncertainty; repetitions are not fresh cases.
- Add deterministic boundary checks for correction/forgetting during support,
  foreign-project candidates, unavailable sources, provider failure and final
  budget enforcement. The adapter stays outside default host configuration.
- Permit one repair of a demonstrated integration defect and one rerun of the
  same development comparison. Do not alter the prompt, model, thresholds,
  dataset or gold labels during this increment. Stop after these two runs at
  most. A semantic or latency failure is a recorded failed candidate, not a
  reason to open another codec search.

Fresh acceptance starts only if a candidate passes the development gates and
has a plausible complete latency result. Freeze its package, settings and
protocol before reading fresh cases. Use new sessions and queries under the
original minimum sample sizes, three repetitions, confidence intervals and
paired comparisons. Retain the 95% supported-fact precision and critical-fact
retention targets, 90% Recall@5 and correct abstention, and zero scope/lifecycle
failures. Extraction remains a separate evaluation. If development fails,
report the failed obligation and return a concrete scope/design decision;
do not spend fresh cases on an already failing candidate.

The [completed run](../validation/2026-11-07/retrieval-support-development-v1.md)
used the actual installed retrieval path for all 272 observations and preserved
all 136 matched candidate/context pairs. Treatment precision is 94.74%, expected
fact retention 90%, correct abstention 52/56 and complete p95 8,641.48 ms. There
are no ranking misses or operational failures; support rejects eight expected
answers and still selects four unsupported facts. One run was sufficient to
establish failure. No integration defect warrants the permitted rerun, and
this bounded comparison is closed. Keep the treatment disabled and preserve
the original gates. Critical-fact retention and 10,000-fact timing are not
established by these known small scenarios.

## Separate the timing contracts

The original requirement at line 1017 of the functional-memory plan is warm
recall p95 <= 500 ms at 10,000 accepted facts and hook durable-capture p95 <=
100 ms. The recent complete-command experiment applied 100 ms to both Stop and
prompt submission, including prompt retrieval. Preserve those results and
their original FAIL labels.

A new protocol must record these boundaries separately before measurement:

| Measurement | Boundary and required evidence |
| --- | --- |
| Durable capture | From host dispatch through persistence of the event and its required durable receipt/outbox state. Identify the actual durability operation and host-visible acknowledgment. Internal dispatch or queue admission alone cannot establish acknowledgment. Verify restart recovery of every acknowledged event. |
| Warm recall | From the installed recall request through its complete returned context, including any support selector. Use 10,000 accepted facts and bind model, dependencies, launcher and context settings. |
| Whole hook command | From launching the host command through exit and complete output, including capture, retrieval and transport. Report separately even when it exceeds the capture-only target. |

Use matched inputs, alternating arm order, at least 50 timed observations per
event/arm and independently reported warmups/cold starts. Keep retries and
failures in the record. Do not add component percentiles together or subtract
retrieval p95 from command p95. If the host only acknowledges at command exit,
report that fact; internal commit time must not be relabeled host acknowledgment.
No new timing pass is claimed by this clarification.

## Pilot mode and remaining decision

Keep the selected private two-project pilot inactive. The proposed full pilot
uses the frozen wheel, local BGE-small embeddings, explicit remember/correct/
forget operations, baseline retrieval and disabled experimental modules and
automatic extraction. It still needs applicable retrieval quality, the timing
requirements, and native-host acceptance on this exact configuration. Results
from the experimental support adapter cannot qualify this baseline retrieval
configuration. Changing the pilot to use that adapter would require a new
artifact/configuration binding and acceptance.

Prepare Codex as the first host on the frozen candidate. Use normal trust
settings and the installed guidance; exercise remember, inspect, restart recall,
same-ID correction, reviewed forgetting and original-source replay, retaining
an unrelated control and a foreign-project check. Record Claude separately
when its isolated authentication and capture-identity limitations are resolved.
Do not extend a one-run trust override into ordinary-host acceptance.

Refresh the private pilot wheel/environment, exact configuration patches and
rollback rehearsal without applying them to live hosts. Keep project names and
unredacted material private. After entry checks, create fresh backups and present
the exact changes for activation authorization. Only actual daily use starts
the two-week clock; zero days have occurred.

If the full configuration still fails entry, an explicit MCP-only learning
pilot can be proposed with automatic recall and capture disabled and source
inspection required for each use. That is a different scope, requires a
prospective user decision, and cannot count as completion of the original pilot
or semantic-quality obligations. It is not the default action of this plan.

The [optional proposal](../validation/2026-11-07/explicit-learning-pilot-v1.md)
is prepared as four project-local patches with twelve allowed tools, no capture
hooks and required source inspection. Metadata parsing and disposable rollback
checks pass. The separate scope question is pending; neither native validation
of this exact proposal nor activation is approved by elapsed time. It starts
neither pilot clock. The original normal-hook review remains a separate request.

## Order and stop points

1. Archive the completed named-map run and record the review. Completed.
2. Verify the minimal backport package and prepare its exact PR body/patch.
   Completed; see the [installed check](../validation/2026-11-07/storage-backport-package-v1.md).
   The separately prepared 0.6.1 release candidate passes package checks but
   fails the legacy benchmark, as recorded above. Its bounded diagnosis is now
   complete. Preserve the failure and require a separately reviewed recency and
   fixture repair before another gate attempt; do not retry for a passing clock.
3. Refresh one delivery bundle on the frozen product, then refresh the private
   pilot drafts and validate the selected native host on that artifact. Bundle,
   pilot drafts and explicit MCP lifecycle are complete. Normal hook review and
   full-configuration execution remain pending.
4. Execute the single bounded end-to-end quality increment and the prospective
   timing protocol. Both increments are complete. The quality comparison failed;
   the [installed pilot timing result](../validation/2026-11-07/installed-pilot-command-timing-v1.md)
   records complete-command and recall gates separately from internal persistence.
   Keep the failed entry checks and prior results intact. Do not launch unrelated
   tuning experiments.
5. If entry passes, request activation on concrete reviewed files and observe
   actual use. If entry fails, present the remaining failure and the scoped
   alternative above. Keep all unmet goal requirements explicit.

Sources: [adversarial review](../validation/2026-11-07/adversarial-goal-review-v1.md),
[named-map result](../validation/2026-11-07/named-choice-support-v1.md),
[original targets](2026-09-16-functional-local-memory.md), and
[pilot protocol](2026-09-20-local-memory-daily-use-pilot.md).


The [canonical integration record](../validation/2026-11-07/canonical-integration-v1.md)
closes the local code/site integration step. Exact patches reconstruct both
reviewed trees; local canonical checkouts now include the frozen product and
current limits. The isolated full suite, typing and local browser checks pass.
The original wheel is preserved. External publication still requires a final
decision, and the failed quality/timing gates, native configuration and original
two-week pilot remain unresolved. No new tuning or pilot activation follows
from these integration checks.

The separate [content-recency repair](../validation/2026-11-07/legacy-content-recency-v1.md)
at `777c5445` addresses the demonstrated legacy access-promotion mechanism.
Its source suite, typing and copied-database migration checks pass. Unknown
revision times remain null and the original failed benchmark is unchanged.
This repair stays outside the narrow storage backport, 0.6.1 candidate and f5
product. Any later benchmark must first declare explicit chronology and
prior-access coverage; the existing ambiguous fixture is not a fresh gate.

Its [installed 0.6.2 package](../validation/2026-11-07/legacy-content-package-v1.md)
now passes source/artifact comparisons and isolated storage, recency and migration
checks. Version and documentation changes preserve the tested repair's runtime
implementation. This is a separate unpublished candidate, with no replacement
benchmark score and no change to the frozen f5 product or inactive pilot.

The [controlled chronology comparison](../validation/2026-11-07/legacy-chronology-v1.md)
now closes that separate development check. Across both stores and three
repetitions, 0.6.2 retains all eight expected facts in each known-chronology
condition before and after stale access; 0.6.1 drops from eight to zero after
access. The repair has zero access-induced selection changes in 192 pairs.
All routes still return candidates when chronology is tied or unknown, so no
abstention or temporal-correctness claim follows for those conditions. The single
run and corrected read-only numeric audit are archived. The original benchmark
and failed gates remain unchanged; this does not reopen the closed f5 tuning
experiments or authorize publication or pilot activation.

The [runtime coverage check](../validation/2026-11-07/legacy-platform-v1.md)
adds Windows 3.10/3.11 and Debian/WSL 3.10-3.12 evidence. It finds a plugin
interpreter-selection failure on both Windows versions. The separate source
repair at `a229d8fd` passes 349 tests in each environment; the unchanged memory
wheel passes 41 installed regressions in each. Original failures and dependency
setup corrections are preserved. The source repair has an exact review patch
and remains outside the frozen candidates and canonical delivery. The functional
plugin has a similar bare-command branch that needs its own scoped follow-up.
Native application acceptance, hosted CI, publication and pilot use remain open.

The [functional plugin follow-up](../validation/2026-11-07/functional-plugin-python-v1.md)
now repairs both command-resolution entry points at `d854db7e`. All 38 wrapper
cases pass in six Python/OS environments, and five checks against the unchanged
installed f5 wheel pass on Windows 3.12. This source-only branch has an exact
patch and review description; it has not changed canonical code, the frozen
wheel or pilot configuration. Native hook review and the optional pilot-scope
decision remain pending. The completed wrapper matrices need no further runs
without a new defect or change.

The [local source integration](../validation/2026-11-07/functional-plugin-integration-v1.md)
now brings the exact verified plugin repair into canonical code at `a7c6dcc8`,
with its unchanged evidence and clearer runtime setup instructions. The main
working branch contains that canonical revision. No source tests were repeated
for the fast-forward; the documentation checks pass. Live configuration, the
wheel and website remain unchanged.

A fresh [upstream overlap check](../validation/2026-11-07/storage-delivery-overlap-v1.md)
finds that the prepared transaction backport conflicts with open PRs #6 and #7.
Prepare local resolutions against their exact heads, preserve their unrelated
changes, and verify the combined storage behavior before choosing a delivery
order. Keep original source/test bindings and all failed product gates intact.

Both [storage compatibility candidates](../validation/2026-11-07/storage-upstream-overlap-v1.md)
are now prepared: `22a74b9d` on PR #6 and `e02c3951` on PR #7. Their selected
storage/engine checks pass, including caller-owned full-text edits across graph
failure; the broader typing check still reports three errors in unchanged
embedding code. Exact patches and independent review drafts are saved. Neither
upstream PR or release order has changed.

The optional explicit-memory learning-pilot scope is now approved for native
validation. Validate its prepared project configuration and guidance in
isolation, keeping activation separate. This approval does not waive original
quality/timing requirements or start either pilot clock.
