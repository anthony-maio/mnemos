# Installed local-memory performance check

Status: prepared, not run. Complete and archive the fresh accepted-state
comparison before starting this check. Do not run another model workload at
the same time. Keep the previous performance failures and their configurations.

## Checkpoint and isolation

Use the installed wheel built from `749f1213381848a0d6040bf8a241829b18b655a1`,
SHA-256 `2884c412f10ce130aed0afad0910c461ba945657f465f73ec604def78ab3ba2e`.
Verify installed files and dependency versions against that checkpoint before
execution. Run outside the source checkout with no source-directory import
override. Record the interpreter, package paths, machine and storage volume.

Create a new private runtime on the system volume, separate from the inactive
pilot, with a synthetic project and profile registry. Use the proposed pilot
launcher unchanged, with a private copy of its settings that points only to the
new runtime. Bind the launcher, settings, registry, harness and input manifest
before timed calls. Do not edit live settings or start the pilot clock.

Keep baseline retrieval, local FastEmbed 0.9.1 and BGE-small-en-v1.5 at 384
dimensions. Record the seventeen model assets and cache location. Require
offline loading; no LLM, judgment, query-state classification or cognition
module is enabled. Keep the proposed safety and capture settings.

## Load and retrieval

Prepare 10,000 declared accepted synthetic facts with retained source excerpts,
stable IDs, asserted status and version 1. Use 9,000 project and 1,000 global
facts so all 10,000 are visible under the project/global request. This load is
not an extraction, lifecycle or semantic-quality evaluation. Record fixture
construction and real document-embedding time separately; do not derive the
load vectors from one seed or substitute mock embeddings.

Start the installed service with its normal runtime factory and maintenance.
Record cold process/service/model startup separately. Check database integrity,
fact/source/evidence counts, vector dimensions, journal mode and synchronous
setting before measurements. Keep top-k 5, the existing relevance floor and
the 1,200-byte conservative context bound.

Use fifty fixed synthetic query strings after three declared warmups. Time
the installed client's complete HTTP retrieval request, including real query
embedding, ranking, metadata writes, source lookup, context construction and
transport. Preserve every raw response and elapsed time. Record returned fact
counts, scope and current-version checks. Empty or degraded results do not
establish successful full-path latency; retain and report them separately.

## Complete hook command

Measure fifty unique `Stop` captures and fifty unique `UserPromptSubmit`
captures after one separately recorded warmup for each event. Use the complete
proposed command: installed Python, unchanged launcher, and Codex hook adapter.
Every invocation starts a new process, as the actual command-hook contract does.
Include process startup, configuration checks and normal request overhead in
the timer. Leave prompt-triggered recall enabled and report it separately from
the capture-only `Stop` command. Do not subtract interpreter startup from the
reported hook duration.

After each invocation, independently find the exact source event and its
durable job in SQLite, including identity, role, content and scope. Verify
survival after a service restart. Inspect the private outbox if capture did not
reach SQLite; queued capture is distinct from committed capture. Neither exit
code zero nor the hook's nonblocking response proves durability. A timeout,
missing event, degraded response or persistence mismatch stays in the failure
count. Do not retry a timed attempt or drop it from the denominator.

These are installed adapter subprocesses with synthetic host payloads. No
native Codex conversation or Claude lifecycle acceptance is claimed.

## Reporting and cleanup

Keep the original requirements: warm recall p95 <= 500 ms at 10,000 accepted
facts and hook durable-capture p95 <= 100 ms. Report sample counts, median,
interpolated p95, maximum and operational failures for each path. A measured
command p95 above 100 ms does not meet the hook requirement, even if its inner
HTTP request is fast. Do not pool distinct paths or call repetitions independent
workloads. This bounded run does not establish a general reliability rate.

Audit raw responses, durable records, source/model/package hashes and percentile
calculations without further inference. Stop only owned services and preserve
the complete private evidence archive. Record any unrelated resource contention
without terminating other applications. Pilot activation, full-module quality,
fresh answer-support acceptance and publication remain separate obligations.
