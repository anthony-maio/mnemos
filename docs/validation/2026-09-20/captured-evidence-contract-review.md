# Captured conversation and the ordinary memory retrieval contract

Status: source-grounded diagnosis and proposed contract, not an implemented
repair or a new acceptance result.

The [v7 native cohort](hermes-native-cohort-v7/README.md) failed all three
lifecycles despite successful model readiness. In attempt 1, correction
retrieval returned older captured instructions not to change memory, and the
model announced a correction without executing it. The trace establishes
that both occurred; it does not establish causation. Attempt 3 omitted a
job-status call during fresh teaching, which this context explanation alone
cannot explain.

## Current behavior

The accepted [service contract](../../plans/2026-09-16-functional-local-memory.md)
distinguishes explicit requested writes from capture for later processing.
Assistant statements must not silently become observed facts or preferences.

In [memory_service.py](../../../mnemos/memory_service.py), `process_job`
deliberately creates a whole-event proposal when processing an extraction job
with sleep disabled. It marks the stored chunk `captured_passthrough` with
`assertion_status: unknown`; assistant content is attributed to the assistant.
Full sleep extraction uses a separate source-backed proposal path. Explicit
remember and correct retain asserted status.

[retrieval.py](../../../mnemos/retrieval.py) labels passthrough content as
unverified, possibly outdated conversation. The service nevertheless returns
these chunks in the same `facts` array and joined `context` as durable
assertions. This behavior is intentional and covered by
[captured-status tests](../../../tests/test_captured_assertion_status.py) and
[service tests](../../../tests/test_memory_service.py). It is not corruption
of the stored assertion status. The product contract still gives ordinary
memory consumers raw conversation and task instructions in the fact channel.

## Proposed next implementation

Preserve episodic events, indexed captured evidence, provenance, and deletion
coverage. Give ordinary host recall a durable-memory view; expose raw captured
conversation through an explicit evidence-review option with a separate
response field and context label. Keep explicit remember/correct behavior.

Apply the distinction to `source: captured_passthrough`, not every legacy or
unverified record. Do not relabel old facts, migrate live data, erase source
history, or change capture consent. A request for evidence review must retain
the existing project/profile/scope restrictions and report source status.
Returning evidence separately reduces ambiguity; it cannot guarantee that
an LLM will interpret evidence correctly or follow every tool instruction.

The filter must operate before candidate limits and context budgeting for
vector, lexical, and graph retrieval. Removing items after top-k could let
high-scoring captures hide valid assertions. Host adapters, MCP schemas,
service responses, UI review behavior, token accounting, and public examples
need a coherent contract. Existing callers must have an explicit way to
review captured history, without automatically injecting it as ordinary
memory. This is a retrieval behavior change and must be documented as such.

## Required evidence before accepting that change

- With many higher-scoring captured instructions and one valid assertion,
  ordinary retrieval still returns the assertion within the requested limit.
- Raw capture, evidence, revisions, pending work, and suppression identities
  remain intact; explicitly requested evidence review can find permitted
  captured records and their provenance.
- Foreign profiles/projects and unassigned data remain excluded from both
  channels, including lexical and graph paths and installed-host adapters.
- Explicit remember/correct remain asserted after overlapping capture, and
  non-passthrough legacy facts are not silently removed from retrieval.
- Related-forget preview still inventories captured copies. The native label
  scans still inspect every retained record, regardless of recall filtering.
  No copy-erasure pass may result merely from omitting a record from context.
- Future native runs retain the existing action, persistence, source replay,
  erasure, and cleanup gates. Preserve all current failures. A passing source
  suite alone does not establish improved model behavior.

## Separate model-response diagnostic

The current native bootstrap records model-request metadata, tool names and
descriptions, plus public transcript messages. It does not retain the exact
transformed system messages and complete request body sent to the provider.
A direct replay assembled from v7 transcripts would therefore be an
approximation, not a faithful replay of the failed request.

If testing the context hypothesis directly, first record exact request bytes
from a prospective, disposable native run, with a frozen model/profile and
synthetic data. Predeclare the compared transformations and next-action
criteria. Preserve the original arm, all failures, and raw responses. A
paired next-action diagnostic is separate from native lifecycle acceptance;
it must not replace native state checks or rewrite v7's outcomes.
