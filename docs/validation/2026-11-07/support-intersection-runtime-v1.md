# Support composition: runtime contract and compatibility replay

The QA/contrast intersection now has an experimental runtime contract. Both
selectors receive the same immutable query, date and candidate cohort, and
their replies must identify that snapshot and cite each selected fact's own
evidence. The result keeps complete qualified facts in baseline order.

The implementation rechecks the full comparison and scope membership before
and after component work. The QA adapter also rechecks around each individual
source inference, so forgetting another candidate during the first call stops
later calls. A correction, evidence change, scope change or newly eligible
candidate invalidates the result. Provider failure and timeout remain
unavailable; cancellation and configuration errors propagate. Both components
share a deadline, and QA selecting nothing does not remove candidates from the
contrast request.

There are 43 new deterministic checks covering these boundaries and the two
adapters, plus 118 passing related support/decoder checks. Tests compare the
contrast adapter's complete question/state request with the original builder,
exercise corrected QA source offsets and null handling, and reject malformed
tokens, foreign evidence and replies for another query or date. Black and
strict mypy pass for the new modules. These checks do not establish model quality.

## Replay of original model outputs

All 44 reused development questions reproduce the saved intersection exactly.
The replay sends 78 original QA outputs through the actual corrected decoder
and 44 original contrast responses through the actual typed judge and contrast
adapter. Query text, dates, candidate order/content, retained source records and
contrast question/state inputs match the original artifacts. No model or
retrieval calls ran.

The reproduced counts remain 25/27 answerable hits, all 25 clean, 16/17 correct
empty abstentions and one unsupported selection. These are the same known
results, not new observations of semantic correctness. The trial-period batch
answer and audit-log reviewer remain omitted; the unmarked retention conflict
still produces the unsupported longer-retention selection. Agreement between
components does not resolve that conflict.

The replay reconstructs 44 disposable SQLite views containing 82 fact records
and 82 retained evidence records. It preserves the input projection's content,
versions, scopes, assertion status, source references and excerpts. The source
fixtures have no local original events; reconstructed views contain no source
events or jobs. This does not reproduce the original database metadata,
embeddings, extraction or lifecycle execution. The new views remain logically
unchanged during selection, and the audit leaves their bytes unchanged.

The independent audit verifies every replayed output, both component witness
sets, all intersections, the original saved result, database integrity and 86
file bindings. The private archive contains 145 verified members, SHA-256
`c1ff1be4040580efb6a5d5729240a0709e515da1e3a73b5934b6ad312358f9dd`.
The initial preflight assumed archive members had directory prefixes; it stopped
before creating replay output. Accepting the existing root member names fixed
the lookup, and exact archived bytes were then verified. Both scripts and logs
are retained.

## What remains

Run the actual local QA and contrast models together under frozen inputs and
settings, with three new repetitions and complete latency accounting. The
saved-output replay cannot predict whether their agreement is stable or meets
the original 500 ms recall budget. Fresh support evaluation must follow any
development changes. Production retrieval, model dependencies, host settings
and the inactive pilot remain unchanged.

- [Runtime protocol](../../plans/2026-11-07-support-intersection-runtime-v1.md)
- [Replay receipt](support-intersection-runtime-v1-receipt.json)
- [Original saved-selection diagnostic](support-intersection-v1.md)
