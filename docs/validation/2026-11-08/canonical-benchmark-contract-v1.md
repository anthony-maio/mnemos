# Recovered benchmark chronology and timing repair

Canonical `7a77e40f` integrates the recovered 0.7.0 chronology correction with
current delivery behavior and fixes a separate ingestion-timing defect. It is
pushed on `wt/backup-20261108-delivery`. The original dirty checkout, PR #9,
installed packages, live databases and frozen evaluation reports are unchanged.

Future legacy benchmark runs assign dataset observations one second apart and
record policy `dataset_order_1s_v1` plus the exclusive UTC end anchor. The CLI
shares that anchor across retrievers, stores and repetitions. Programmatic
comparisons can pass the same timezone-aware observation_end. Naive anchors are
rejected before storage cleanup or creation. Dataset order is an explicit fixture
assumption, not inferred real-world chronology.

Baseline and engine paths now assign created_at, updated_at and last_observed_at
consistently; engine interaction timestamps match the observation. Review found
that repeated content can return an existing chunk ID. The adaptation preserves
the first creation timestamp for that ID while advancing the later observation
timestamps. Both memory and SQLite cover this case.

The engine path previously measured ingest_seconds until after all queries.
A deterministic timing regression reproduced 15 seconds reported as ingestion
where the baseline correctly reported one second. The measurement now ends
before query execution in both paths; query latency retains its own boundaries.
These synthetic clock values are a regression fixture, not measured performance.

## Validation and evidence limits

The original recovered test expected perfect MRR on its old/new fixture. Current
canonical retrieval produced 0.375 on both stores before and after timestamp
assignment. Its ranking differs from the old implementation, so that score
assertion is not a valid migration contract. Both failed attempts are preserved;
no semantic improvement follows from this repair.

Direct persisted-timestamp and timing assertions were then run against an
isolated archive of pre-change canonical `036357fe`. Four chronology cases
(baseline/engine x memory/SQLite) and the engine timing case failed. The baseline
timing control passed. The copied tests and baseline archive are hash-bound in
`.venv/canonical-benchmark-red-v2/result.json`; the original checkout was not
rewritten to reproduce the failures.

Final candidate validation: 26 benchmark tests pass in 10.50 seconds, including
shared explicit anchors, timezone rejection and repeated observations. Mypy
accepts the changed runtime module, Black accepts both Python files, and diff
checks pass. Documentation checks: 11 pass, one external website check skips
because MNEMOS_WEBSITE_INDEX was not supplied. This is not website verification.
Final JUnit is `.venv/canonical-benchmark-final-v3.xml`; the documentation JUnit is
`.venv/canonical-benchmark-docs-v1.xml`. Earlier failed receipts remain intact.

This legacy harness still does not establish identical accepted-fact snapshots,
answer support, abstention quality or complete native latency. New benchmark
results require their own source/data bindings; existing scores must not be
relabeled or regraded. No acceptance benchmark or new model experiment ran.
