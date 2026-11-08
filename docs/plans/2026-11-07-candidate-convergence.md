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

## Order and stop points

1. Archive the completed named-map run and record the review. Completed.
2. Verify the minimal backport package and prepare its exact PR body/patch.
   Completed; see the [installed check](../validation/2026-11-07/storage-backport-package-v1.md).
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
