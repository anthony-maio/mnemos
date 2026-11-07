# Reconciliation comparison v1

Neither change qualifies for a full-pipeline regression. Direct instructions
recovered the four missed correction cases but introduced wrong-target
corrections in three controls. Removing the requirement-revision flag from
other source kinds did not improve the legacy treatment. Keep the existing
default and preserve both variants as development experiments.

The [protocol](../../plans/2026-11-07-reconciliation-development-v1.md) and
implementation were frozen at
`e13e957e9e2e1e086793e4f310c610dc7a624717`. Twenty authored cases in six
clusters ran three times under each of four arms, producing 240 observations.
Each arm received the same supplied classification, exact source, scoped target
snapshot and eligible-operation schema. This isolates reconciliation; it does
not measure extraction or fresh quality. The known correction exactly matches
the source, candidate, target, baseline prompt and schema from
[isolated extraction v2](isolated-extraction-v2.md).

| Measure | Legacy/full | Legacy/relevant flag | Direct/full | Direct/relevant flag |
| --- | ---: | ---: | ---: | ---: |
| Exact operation and target observations | 48/60 | 48/60 | 48/60 | 45/60 |
| Cases correct in all three repetitions | 16/20 | 16/20 | 16/20 | 15/20 |
| Matching correction trials recovered | 0/12 | 0/12 | 12/12 | 12/12 |
| Correct unresolved controls | 27/27 | 27/27 | 15/27 | 15/27 |
| Wrong support or correction target choices | 0 | 0 | 9 | 9 |
| Operational failures | 0 | 0 | 0 | 0 |
| Median observation latency | 1,820 ms | 1,771 ms | 1,459 ms | 1,420 ms |
| p95 latency, nearest rank | 2,662 ms | 2,944 ms | 2,035 ms | 1,997 ms |

The legacy arms left every matching correction unresolved: the original case,
a target among distractors, an explicitly qualified environment and a numeric
setting. The direct arms corrected all four, but also chose a target when the
environment was ambiguous, applied a production correction to a staging fact,
and chose between two numeric settings without a named subject. Each mistake
occurred in all three repetitions. A schema-valid ID and version did not
establish that the target was semantically appropriate.

Both direct arms also called two normalized duplicate targets a disagreement
instead of leaving their identity unresolved. Direct/relevant additionally
called an independent assistant statement a conflict with an unrelated
attribute. Exact support with one target, independent user facts, both
continuing-requirement cases and the known assistant disagreement passed in all
arms. The original source text and role restrictions remained intact.

The direct/full arm tied the baseline aggregate score by exchanging four fixed
cases for four regressions. It failed the prospectively frozen selection rule,
as did every other arm. Its lower latency does not justify adoption. Three
repetitions produced the same operation/target correctness within each case
and arm; they do not create 60 independent cases or establish broader
reliability. The requirement-flag hypothesis received no positive support in
the legacy comparison.

| Provider-reported tokens | Legacy/full | Legacy/relevant flag | Direct/full | Direct/relevant flag |
| --- | ---: | ---: | ---: | ---: |
| Prompt | 39,741 | 39,201 | 44,241 | 43,701 |
| Cached prompt | 30,739 | 30,397 | 34,560 | 35,124 |
| Output | 5,293 | 5,073 | 3,599 | 3,614 |

All 240 logical calls produced one recorded response with a normal stop reason.
The maximum native prompt-plus-schema count was 1,578 against the 5,632 limit;
maximum provider-reported prompt and output counts were 801 and 131. The model
was the same local Qwen3:14b Q4_K_M, with thinking disabled, temperature zero
and context 8192. No database, retrieval, embedding or native host was used.

The independent audit reconstructed all 240 scores from the frozen labels and
raw decisions. It verified 133 source hashes, all source provenance and target
versions, all 20 identical input groups, schemas, exact preflight prompts and
token accounting. Sixty-four focused tests passed, including 33 new checks;
Black and strict mypy passed. The earlier full-suite result remains bound to
its original source.

The [receipt](reconciliation-v1-receipt.json) binds 165 verified private archive
members under SHA-256
`0e767f85faadd73206b20c1d248e2dbd30d1bd98bd2be0b0855f5c5237ec50d3`.
The model unloaded, all listed owned processes were absent and both endpoints
closed. The original tree-stop warning is retained; no extra kill or model
rerun occurred. No live settings, installed pilot artifact or prior evaluation
record changed.

Further reconciliation work needs an independently checked target identity,
including qualifiers and ambiguity, before granting correction authority.
Exact duplicate identity is also a mechanical constraint worth tightening.
These findings do not authorize a new default, a fresh-quality claim or pilot
activation.
