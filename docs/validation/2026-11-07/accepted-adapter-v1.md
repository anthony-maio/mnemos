# Terminal-gold adapter validation

The adapter now converts the existing v4/v5/v6 evaluation contract into declared
accepted states for retrieval-only comparisons. It preserves the original query
wording, scope permissions and retained evidence. The real-embedding development
screen completed 81 observations. All three strategies found the expected answers;
all three still failed the empty-result abstention criterion. This validates the
conversion path without claiming a retrieval-quality improvement.

## Conversion boundaries

The original evaluator runs its ordered events before asking ordinary and
state-pair questions. The adapter therefore requires an explicit
`after_all_events` phase. It retains supported and conflict gold facts, excludes
superseded, forgotten and rejected facts, and marks conflicts explicitly. Each
query is placed one microsecond after the final declared event. Earlier questions
need explicitly authored query-time snapshots through `AcceptedCorpus`.

Gold fact IDs, contents and full evidence excerpts remain intact. Their versions
start at declared version 1; they are not runtime correction versions. Original
source IDs are retained once, with reciprocal fact/evidence links and explicit
scope partitions. No capture, extraction, correction, forgetting or replay is
executed by this conversion. Removal from a declared snapshot is not evidence of
a successful forgetting workflow.

The adapter preserves the original project/global query permissions. A workspace
fact cannot satisfy a query under those permissions. An explicitly expanded
permission list is recorded in conversion provenance. It also preserves the
synthetic, unreviewed source labels. Held-out conversion requires a validated
source file and the original full-dataset minimums. File-backed objects must match
their bound file; small development fixtures do not weaken those requirements.

The original state pair contains two different question strings. Corpus format
v2 records that distinction explicitly and preserves both strings. Existing v1
fixed-query serialization is unchanged. Changing the wording can change the query
embedding, as it did here. All three baseline strategies ignore the declared
state labels, so this comparison cannot isolate a state-module effect.

## Frozen development screen

Two authored v6 development sessions contain fourteen events and ten gold facts.
The adapter retained five supported facts and two conflicts, excluding one
superseded, one forgotten and one rejected fact. Its two snapshots contain five
and two facts, respectively, with seven retained sources and seven evidence links.
The fixture includes correction and replay declarations, foreign-profile data,
global data and a workspace fact. These describe expected terminal state.

The nine query strings comprise four ordinary answerable questions, three
no-answer questions and both variants of one state pair. Each ran three times
against vector, lexical and hybrid retrieval, with the same accepted state,
document vectors, scope permissions, top-five limit and 1,200-token budget.
Vector and hybrid shared each query vector; lexical retrieval used no query vector.

The run used local FastEmbed 0.9.1 with `BAAI/bge-small-en-v1.5`, 384 dimensions,
offline loading and seventeen unchanged cache assets. It made seven document and
twenty-seven query embedding calls. No extraction or answer-support model ran.

| Strategy | Ordinary answer hits | Clean ordinary hits | Correct empty abstentions | Semantic-pair hits | State-wording hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 12/12 | 6/12 | 0/9 | 3/3 | 3/3 |
| Lexical | 12/12 | 9/12 | 0/9 | 3/3 | 3/3 |
| Hybrid | 12/12 | 6/12 | 0/9 | 3/3 | 3/3 |

These denominators repeat four ordinary answerable questions, three no-answer
questions and one pair three times. Across all twenty-seven observations per arm,
vector and hybrid each returned 63 nonrelevant fact selections; lexical returned
45. A nonrelevant selection means the returned ID is outside the query's expected
answer set. It does not mean that the stored statement itself is false. Conflict
qualifications were preserved but did not meet the empty-result criterion.

There were no operational failures, scope leaks or stale returns. All twenty-seven
query/strategy groups returned stable IDs across repetitions. The foreign fact
was available only under its declared target profile/project; workspace data
remained excluded under the original permissions. Both question variants reached
the embedder unchanged and produced distinct, repeat-stable vectors.

The independent audit checked the original source fields against the converted
facts, evidence and questions; all 34 recorded role-specific vectors; all 81 raw
results; and 153 bound files. It verified the two SQLite databases' exact fact
contents, zero jobs and access writes, integrity checks and foreign keys. It also
recomputed the reported counts and intervals without model calls.

Wilson intervals use four, three or one unique questions per repetition,
depending on the category. Paired ordinary comparisons retain twelve or nine
observations in two session clusters; pair variants each have three observations
in one cluster. Every paired hit/abstention and fractional-recall difference was
zero, so all bootstrap draws are zero. These tiny development samples do not
establish population uncertainty or fresh quality acceptance.

The run took 37.65 seconds including provider construction, binding checks,
database preparation/copies, retrieval and reporting. This is not a native-host
or 10,000-fact latency result.

## Verification and remaining work

The focused and adjacent suite passed 145 checks, including 35 new adapter cases.
Twelve documentation checks also passed. Black passed for the three changed Python
files, and strict mypy passed for both benchmark modules. During development,
checks caught a test import, a missing scenario tag and type declarations; those
were corrected before the frozen embedding run. The old v4 development split was
used only for conversion and bound-file drift tests, without model inference.

Executed implementation: `1c8962d1c4c7d3a3d3030996fa96cfaaec4ebcef`.
Frozen source file SHA-256:
`e3fb5bd282e2fe4a1480039b3092ebcee1715893ea5f3785f8c3de2cf68da8de`.
Canonical accepted corpus SHA-256:
`e0ffb7fa53e8d7140f1f45d84ee59ef83f211bcd80c4b443096e3381e0bade68`.

The terminal-phase adapter is implemented and development-tested. Applying it to
the prepared fresh corpus and running that comparison remain unrun. The fresh
corpus remains unopened. Earlier query phases still require explicit snapshots;
the adapter does not reconstruct them from terminal labels. Answer-support quality,
the full-module comparison, remaining native/scale checks and the actual two-week
pilot remain open. Production package sources, dependencies and live configuration
are unchanged. The pilot remains inactive at zero days.

- [Prospective protocol](../../plans/2026-11-07-accepted-adapter-development-v1.md)
- [Machine-readable receipt](accepted-adapter-v1-receipt.json)
- [Accepted-state corpus runner](accepted-corpus-v1.md)
- [Separate answer-support experiment](extractive-answer-support-v1-v2.md)
