# Frozen retrieval fixture and oracle audit

September 20, 2026. Read-only diagnosis of previously observed synthetic
data. No model calls, production changes, new grades, or acceptance changes.

The saved retrieval run contains two conflicts between its fixtures and the
contracts it evaluates. Neither finding changes the original run's results.
The [terminal archive](../../2026-09-19/retrieval-terminal-local-v1/README.md)
remains incomplete: 589 of 600 session observations completed.

## Findings

The original full strategy has 106 hits and 183 misses among 289 eligible
answerable observations. Eleven answerable observations belong to partial
sessions and remain excluded. The 183 misses divide as follows:

| Recorded misses | Saved evidence | Interpretation |
| --- | --- | --- |
| 120 | The source-linked returned fact exactly copies the explicit remember body, including an instruction prefix omitted from the gold fact. | The explicit API preserves that body; the exact-text judge expects a different body. Literal inclusion is a diagnostic, not a semantic pass. |
| 60 | Twenty foreign-profile/project queries, each repeated three times, return no facts from any of the four strategies. They are labeled answerable using normal-project facts. | Empty results follow the intended scope boundary. The fixture's answerable label conflicts with that boundary. |
| 3 | Source-linked returned facts use different wording from the gold text. | Source association alone does not establish semantic correctness. These remain unadjudicated misses. |

All 20 conflicting canaries are `heldout-061` through `heldout-080`. Their
only capture event uses the normal session context. The query changes both
profile and project; the assistant's foreign-target event has `write_mode=none`
and does not seed a foreign fact. All 60 complete canary observations return
empty fact lists in all four strategies. Accepting foreign facts would defeat
the isolation check.

Across the dataset's 200 heldout-labeled sessions, all 60 explicit remember
events have a body different from the authored gold fact. The prefixes are
`Please remember this project decision explicitly: `, `Remember for this project: `,
and `Remember temporarily: `, with 20 events each. The temporary-memory
family is distinct from the 40 answerable sessions behind the 120 wrapper
misses. These counts must not be conflated.

The adapter passes `event.content` as the explicit `remember` body. The
service creates its fact proposal from that body without asking a model to
strip an instruction prefix. The answerable judge compares the entire
returned fact against either the expected answer or the gold fact, after
case folding and trimming. It does not evaluate an answer generated from
the returned evidence.

The saved `extraction_judgements.accepted` field is also an exact-text and
evidence grading proxy. A false value does not mean the runtime refused to
store the fact: these rows are generated from persisted facts. The audit
joins source event identities to each runtime fact before comparing the
returned set; an unrelated returned fact cannot inherit another fact's
evidence link.

## Reproduce and inspect

- [fixture-audit.json](fixture-audit.json) enumerates all 20 scope conflicts,
  their 60 observations, and all 60 explicit remember/gold mismatches.
- [full-strategy-boundaries.json](full-strategy-boundaries.json) retains the
  183 original full-strategy misses with runtime identities and source links.
- [all-strategy-boundaries.json](all-strategy-boundaries.json) applies the
  diagnostic to all four strategies: 1,156 complete strategy rows over the
  same 289 eligible answerable observations, plus 44 separately excluded
  rows from partial sessions. These are repeated strategy measurements,
  not 1,156 independent queries. Replay checks all 1,200 original verdicts
  and confirms that the source report was not mutated.
- [dataset-exact.jsonl](dataset-exact.jsonl) preserves the original frozen
  input bytes. Its SHA-256 is
  `036fa51ee13271a85a14a62a2fe4ef65d13eaeb290331fe325aa5bf32ce893fc`.
- The original `full.json` remains inside the linked terminal archive's
  `retrieval-reports.zip`; its SHA-256 is
  `9553827a7c5f8b14079607c7fb5b38a4c40457d8987ad41efac1b825a028b8de`.
  Evaluation source is `fc72cc251aeafa161a1a390269b8dfa34d83cf2b`.
- [contract-source.json](contract-source.json) records the relevant adapter
  and service methods from that exact evaluation commit.

From the repository root, run
`python docs/validation/2026-09-20/retrieval-fixture-audit-v1/verify_fixture.py`
and then `python docs/validation/2026-09-20/retrieval-fixture-audit-v1/verify_boundaries.py`.
They read the saved inputs and regenerate the two diagnostic JSON files;
they do not contact a model or modify the frozen evaluation.
The reusable diagnostic is replayed by
`python docs/validation/2026-09-20/retrieval-fixture-audit-v1/run_miss_audit.py`.

The original dataset's raw hash differs from a line-ending-normalized Git
checkout. This audit uses the byte-identical frozen input, checks both
input hashes, and leaves the strict original guards intact.

## Verification

Thirteen diagnostic tests and 71 adjacent benchmark/documentation tests
pass. Black and strict Python 3.10 mypy pass for the two new source files.
The archived red tests include the initial missing implementation and a
subsequent missing-grade denominator failure. The repaired diagnostic keeps
unknown grades separate from misses and does not copy the full strategy's
grade into another strategy.

The original dataset, judge, comparison runner, and production package are
unchanged. This checkpoint does not rerun the full product suite or claim
that the 84 offline checks validate model quality. The evidence index binds
the new diagnostic source, replay files, exact dataset, and test results.

## Prospective repair, not run

Keep v3 and every old result immutable. Prepare a separately versioned
contract-repair dataset and freeze it before further inference. Label it
previously observed synthetic data, never new unseen heldout evidence.

1. Preserve the 20 foreign queries as no-answer scope canaries. Add 20
   separately identified in-scope answerable queries so the original
   minimum of 100 answerable queries remains intact.
2. Define explicit-memory gold against the exact API body. Keep normalized
   semantic values in a separate, unscored annotation if useful. Do not
   introduce substring acceptance or silently rewrite old source events.
3. Audit correction, forgetting, source links, and retention labels against
   those bodies before freezing the successor. Preserve at least 200
   sessions, 100 critical facts, 100 no-answer queries, 50 state pairs,
   and three repetitions, along with the existing gates and uncertainty
   reporting.
4. Run fresh baseline/full/ablation comparisons under the new manifest and
   report them separately. Do not recalculate the old run as a pass.

If explicit remember should instead normalize instructions, specify and
test that product behavior separately. The existing fixtures do not grant
that contract. Automatic extraction quality, partial lifecycle runs,
graph utility, classifier fidelity, complete native capture, and the pilot
remain open.
