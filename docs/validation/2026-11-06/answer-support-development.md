# Answer-support development results

The first local support checker failed this development comparison. It stays
experimental and is not enabled in production retrieval. Existing frozen scores
and release thresholds are unchanged.

## Setup

The corrected run used commit `553c147a`, Qwen3 4B Q4_K_M through local Ollama
0.40.9, thinking disabled, temperature zero and BGE small English v1.5 embeddings.
The model digest, dataset hash, complete source hashes, limits and platform are
in the archived manifest. The source hashes still matched after the run.

Ten synthetic development scenarios provide ten answerable and ten no-answer
queries. Each query runs vector, lexical, hybrid and full retrieval against
copies of one accepted-fact SQLite snapshot. Extraction is disabled. The support
treatment filters exactly the returned baseline IDs, without backfilling. Full
ranking enables spreading but disables affective scoring; this run says nothing
about affective utility. The original 1,200-byte context bound is used throughout.

This is one development repetition on a small hand-authored set, not held-out
evaluation. Scenarios cover paths, ownership, time, negation, uncertainty,
source mismatch, attribution, quoted instructions, scope and conflicts.

## Corrected run

| Retrieval | Baseline answerable hits | Support answerable hits | Baseline correct abstentions | Support correct abstentions | Support operational failures |
| --- | --- | --- | --- | --- | --- |
| Vector | 10/10 | 3/10 | 0/10 | 1/10 | 11/20 |
| Lexical | 10/10 | 4/10 | 0/10 | 2/10 | 9/20 |
| Hybrid | 10/10 | 3/10 | 0/10 | 1/10 | 11/20 |
| Full | 10/10 | 3/10 | 0/10 | 1/10 | 11/20 |

All 80 observations completed. There were 42 response-contract failures, all
from probability distributions that did not sum to one. These count against
operational success and receive no abstention credit. Support processing took
722 ms at the median and 1,211 ms at the maximum, excluding retrieval and setup.
Local model probability values are not calibrated confidence estimates.

Correctly formatted replies also made semantic mistakes. The full arm treated
separate owner assignments as evidence that one owner approved the other,
accepted an ownership claim whose source only described meeting attendance,
and returned an approval requirement when asked for an unresolved retention
duration. Fixing response formatting alone cannot establish answer support.

The baseline includes the expected fact for every answerable query in this
small set, but also returns related facts for no-answer queries. Baseline hits
therefore establish candidate coverage here, not answer reliability. The new
checker currently loses too much coverage and still accepts unsupported facts.

## Original run and fixture repair

The original run at `0553aeff` is retained separately. Its local and foreign
fixture directories inherited the enclosing checkout's Git identity. Both
were therefore assigned to the same project. Its scope scenario is invalid;
do not interpret that result as a demonstrated production scope leak or include
its aggregate scores in a release claim.

A regression test inside a temporary Git checkout reproduced the collision.
Each fixture project now has its own Git repository, and the runner rejects
equal identities before seeding. The regression verifies that all four arms
exclude the foreign fact. The corrected live run also excluded it.

The original run had 43 response-contract failures and one timeout. These
records have not been replaced by the corrected run.

## Evidence and next work

The two ZIP archives in this directory contain raw model responses, observations,
manifests, summaries, the dataset and accepted SQLite databases. All content is
synthetic. [Receipts](answer-support-receipts.json) identify every archived file
and both archive hashes. The private model/cache directories are not included.

Next experiments should separate response-contract repair from semantic support
selection, including whether judging one candidate at a time reduces mistaken
relationships. Preserve these runs, record any new prompt or batching strategy,
and evaluate answerable coverage as well as abstention. Shortening evidence
spans and default integration remain deferred until support selection improves.

Unit and integration verification before the live runs: 53 tests passed across
support, comparison, retrieval comparison and typed/local judgment behavior;
the new fixture regression then failed before its repair and all four comparison
tests passed afterward. Black, strict mypy for both new source modules and
`git diff --check` passed. No native host or live memory database was involved.
