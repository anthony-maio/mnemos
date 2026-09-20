# Historical assertion extraction: independent contract review

A separate read-only review examined the archived lifecycle failures, the
extraction prompt and schema, durable job handling, and correction target
resolution. It made no production changes, model calls, or held-out label
changes. This is an engineering review, not human acceptance of the labels.

The [diagnosis](../2026-09-19/lifecycle-extraction-diagnosis.md) records 50
partial lifecycle observations after `no_durable_facts`, each followed by
`missing_accepted_target`. Their responses were valid structured abstentions;
the reviewed records did not show parser, evidence-ID, firewall, timeout, or
truncation errors. The [original archive](../2026-09-19/lifecycle-incomplete-local-v2/README.md)
retains all failed observations. These results are compatible with an
underspecified policy for attributed historical statements, rather than
establishing a target-resolution defect.

## What the code establishes

The [extraction prompt](../../../mnemos/modules/sleep.py) asks for durable
source-backed facts, preferences, and continuing requirements, preserves who
made a claim, and permits abstention. It does not define a complete temporal
memory policy. The [response schema](../../../mnemos/contracts.py) requires
an explanatory reason for abstention and an explicit existing target for
support or correction; a correction also needs the expected version.

The [service](../../../mnemos/memory_service.py) commits a no-facts job and
retains its episodic source without creating a fact. The
[evaluation adapter](../../../benchmarks/local_memory/ollama_adapter.py)
requires exactly one accepted fact linked to the correction's source target.
When extraction created none, target absence is an observable consequence.
The [deterministic regressions](../../../tests/test_lifecycle_extraction_outcomes.py)
cover both no-facts followed by an unavailable correction target and a direct
assertion followed by a version-2 correction. They do not settle extraction
quality or the historical policy.

## Policy boundary and additional development cases

An attributed statement such as "the earlier note said Redis" must not
silently become a claim that the current system uses Redis. That does not
mean all historical facts should be discarded. "The April outage was caused
by a Redis failover" can be useful dated information. The product needs a
clear distinction between current state, historical assertions, and raw
source quotations, including what each retrieval view returns.

The prior [12-case matrix](../../../docs/plans/2026-09-19-historical-assertion-contract-review.md)
is a useful minimum, but the next prospective development protocol should
also cover these cases before changing extraction behavior:

| Additional case | Required distinction |
| --- | --- |
| Two incompatible current assertions without an explicit correction | Report a conflict; do not silently select one |
| A dated outage cause versus the current cache configuration | Retain temporal meaning and prevent a historical answer from posing as current state |
| "My previous manager required weekly releases, and I still follow that" | Explicit current adoption differs from attribution alone |
| A correction against a stale version or two possible accepted targets | Preserve version checks and refuse ambiguous target resolution |
| Equivalent project, workspace, and global examples | Retain scope authority independently of wording |
| An assistant quoting a user statement without authoritative user evidence | Quotation alone does not establish a new user assertion |

Bootstrapping a new fact when correction has no target is a separate policy
choice. It must not masquerade as a successful revision of an existing fact.
The current target-based correction contract remains unchanged.

A product preference question about separately dated historical memories is
pending. The recommendation is to retain useful historical information with
explicit temporal meaning and separate current-state semantics. No answer is
assumed, and no dependent product change has been made. The old synthetic
labels remain fixed and their human-acceptance status remains NOT RUN.
