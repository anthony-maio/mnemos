# Extraction grader candidate accounting

Two synthetic regressions reproduced incomplete pipeline output receiving quality credit. With one retained candidate but no reconciliation or proposal, the old grader marked an empty-expected case as successful abstention. With two retained candidates and only one reconciliation, it could mark the case exact against the one expected outcome.

The grader now requires exactly one reconciliation per retained candidate. Missing, duplicate and out-of-range candidate indices produce a separate candidate_accounting_errors count. An accounting failure prevents operational success, exact-case, abstention and retention credit. Accepted-output counts remain visible, so rejected structure does not erase the emitted proposals. Duplicate reconciliation previously received partial one-to-one credit; it now fails the operation rather than crediting any retained fact.

Each new grade identifies exact-span-candidate-accounting-v2. Historical observations, reports and scores remain unchanged and were not regraded. This is an annotation-proxy correction, not evidence of improved model extraction or semantic precision. No provider, retrieval, embedding or database operation ran.

Both new regressions failed before the change. All 22 initial grader checks passed after it. An additional out-of-range-index check brought the final combined grader/source-pipeline suite to 31 passing tests. Formatting and diff checks passed. Private JUnit files retain the red reproduction, focused green run and combined runs under .venv/extraction-accounting-*.xml.

Independent review found no blocker and confirmed that failures remain in the quality denominator. Accounting and provenance counters are not disjoint: an out-of-range index contributes to both. Do not sum them as unique failed outcomes.

Scope is the retained candidate list returned by SourcePipelineResult. Oversized candidates are explicitly filtered earlier by the pipeline and are not silently counted as retained candidates. This patch does not reconstruct the classification inventory, validate persisted extraction, establish fresh acceptance quality or complete the original extraction requirement.
