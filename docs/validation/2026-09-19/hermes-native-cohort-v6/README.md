# Native Hermes fixed cohort v6: reviewed-copy workflow failures

All three scheduled attempts ran and failed at forgetting. No attempt was
replaced or retried. Runner `d0b3649789e1ed4f54d8865d061fed241cc5c3e8`
used verified runtime `db932befcd4d0aac9220f9f1e73f043ccd00093c` and wheel
`7df060463b35853d97898ca839b500a66e233bef26b8e52a7de1db799681190a`.
All three passed teaching, restarted recall, correction, and restarted
corrected recall. The cohort ended at 2026-09-20 02:32:32 UTC, September 19
local time. Cleanup was established after every attempt.

| Attempt | Forget behavior | Observed failure |
| --- | --- | --- |
| 1 | Retrieved the target, requested a preview, then applied target-only deletion with empty additional selections. | Skipped all inspections. The inspected-mutation and reviewed-workflow gates failed, and retained project records still contained the labels. |
| 2 | Retrieved and inspected the target, requested a preview, then applied target-only deletion with no additional facts or sources. | The reviewed target deletion and its persisted state passed. Only the retained-label gate failed: conversational copies remained. |
| 3 | Retrieved and inspected the target, requested a preview, then twice requested additional deletions without their source-event selections. | Both apply calls returned degraded HTTP 500. The trace evaluator stopped on the unsuccessful result before persisted forget-state checks. |

Attempts 1 and 2 received complete previews containing 12 and 8 candidate
facts, respectively. Both models deliberately excluded conversational copies
despite the request to forget them. The service's empty-selection behavior
was correct: it deleted only the target. Attempt 2 demonstrates why a valid
reviewed operation is separate from fulfilling the broader forgetting request.

Attempt 1 did not inspect even the target. That missing inspection makes its
reviewed-deletion validation fail; empty additional selections alone are
legal and do not make a target-only operation invalid. Its retained-marker
failure independently shows that the requested copies survived.

Attempt 3 selected the target and two other fact IDs but omitted source-event
IDs. Neither additional candidate was inspected. Its HTTP 500 results require
a separate service-validation investigation; the raw trace does not itself
identify the underlying exception. The evaluator's rejection of degraded
results is preserved. No source replay or final recall ran in any attempt.

This prospective successor exposes eight tools and changes the forget prompt
to the reviewed-copy workflow. It also strengthens persistence checking by
scanning scoped facts, events, evidence, revisions, and jobs for the two
fixture label tokens, retaining observed scoped fact IDs after deletion.
Earlier request and control gates remain. See the copied v6 protocol for the
fixed settings, scope, turn budgets, and exact criteria. This is not a
controlled estimate of the new tools' effect or universal paraphrase erasure.

The result is zero of three complete lifecycles. Earlier v3-v5 failures remain
unchanged, and the three-host prerequisite is still unmet. Passing package
tests does not establish that this model reliably completes the native
reviewed-copy workflow.

`cohort.json` retains all outcomes. `evidence-index.json` binds 172 raw files,
including executed inputs, native traces, synthetic state captured by the
runner, and logs. This interpretive README is outside that index. Databases,
runtime access descriptors, credentials, and live user content are excluded.
There is no post-forget state snapshot for attempt 3 because evaluation
stopped before that step; absence of a snapshot is not evidence of erasure.
