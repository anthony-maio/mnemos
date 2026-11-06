# Answer-support development v2 results

Requiring complete probability distributions fixed most response-contract
failures. It did not make this model reliable at deciding whether a fact answers
the question. Every treatment misses the required abstention target, and several
wrong decisions carry probability 1.0. Answer support remains experimental and
disabled in production retrieval.

## Comparison

Commit `6d46aa1a` implements the four treatments frozen in the
[prospective protocol](../../plans/2026-11-06-answer-support-development-v2.md).
The run used local Qwen3 4B Q4_K_M, Ollama 0.40.9, a 16,384-token context,
temperature zero, thinking disabled and BGE small English v1.5 embeddings on
an RTX 4080. The source commit, model digest, runtime settings and dataset hash
are recorded in the archive.

All 320 observations completed: 20 reused development questions across four
retrieval strategies and four support treatments. These are 20 distinct
questions, not 320 independent questions. Each retrieval arm ran once per query;
the treatments received identical baseline IDs, facts and accepted-fact snapshot
hashes. Extraction was disabled. The threshold stayed at 0.8 throughout.

Every baseline retrieved the expected fact on all ten answerable queries, and
none abstained on the ten no-answer queries. Baseline retrieval returned 30
unsupported facts per arm, except lexical with 28. Those hits establish candidate
coverage on this small set, not answer reliability.

## Results

Each row contains ten answerable and ten no-answer queries. A clean answerable
hit returns exactly the expected fact IDs; an ordinary hit may include extra
unsupported facts. An operational failure receives no abstention credit.
Unsupported facts are counted across all 20 queries. Latency is the median
support-operation time, excluding retrieval, snapshot creation and setup.

| Treatment | Retrieval | Hits /10 | Clean hits /10 | Correct abstentions /10 | Unsupported facts | Failures /20 | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy-batch | vector | 3 | 3 | 1 | 5 | 12 | 741 |
| legacy-batch | lexical | 4 | 3 | 1 | 7 | 10 | 696 |
| legacy-batch | hybrid | 3 | 3 | 1 | 5 | 12 | 726 |
| legacy-batch | full | 3 | 3 | 1 | 5 | 12 | 710 |
| normalized-batch | vector | 8 | 5 | 2 | 14 | 1 | 942 |
| normalized-batch | lexical | 9 | 6 | 3 | 13 | 0 | 950 |
| normalized-batch | hybrid | 8 | 5 | 2 | 14 | 1 | 940 |
| normalized-batch | full | 8 | 5 | 2 | 14 | 1 | 948 |
| normalized-candidate | vector | 9 | 4 | 5 | 12 | 0 | 1,093 |
| normalized-candidate | lexical | 9 | 5 | 5 | 11 | 0 | 1,047 |
| normalized-candidate | hybrid | 9 | 4 | 5 | 12 | 0 | 1,114 |
| normalized-candidate | full | 9 | 4 | 5 | 12 | 0 | 1,070 |
| separate-candidate | vector | 10 | 3 | 3 | 17 | 0 | 1,831 |
| separate-candidate | lexical | 10 | 4 | 3 | 16 | 0 | 1,774 |
| separate-candidate | hybrid | 10 | 3 | 3 | 17 | 0 | 1,795 |
| separate-candidate | full | 10 | 3 | 3 | 17 | 0 | 1,797 |

The legacy treatment had 46 failed observations. Requiring a complete
distribution reduced that to three in the batched treatment; those three chose
an option whose probability was below another option. Candidate-at-a-time
treatments had no contract failures. No treatment timed out. All 444 requests
produced a recorded response, with no missing or duplicate response IDs.

The model supplied probabilities on every request, including the legacy arm;
there were no omitted distributions converted to one-hot choices. Of the 142
legacy choice answers, 108 omitted at least one offered key and 89 failed the
unit-sum condition. These counts overlap and describe answers, not observations.
The three stricter treatments supplied every key and passed the sum check.
No values were renormalized or filled by the new policy.

## Remaining semantic failures

Candidate-at-a-time judgment improved abstention to 5/10 across all retrieval
arms, but five no-answer queries still returned unsupported facts. In the hybrid
arm it also omitted the explicitly unapproved migration when asked whether that
migration had been approved. Uncertainty was present in the answerable fact;
discarding it lost useful information.

Both candidate-at-a-time treatments accepted a statement identifying the current
queue as an answer to a question about its cloud region. Both returned migration
rules when asked who approves migrations. They also selected old queue history
alongside the current queue and staging rules alongside a production rule.
These examples received probability 1.0, so raising the existing threshold would
leave them in place.

Separating answer relevance from source entailment recovered the ten expected
answers but accepted more unrelated facts. In the source-mismatch case, it
correctly rejected an ownership claim supported only by meeting attendance,
then selected a code path as the answer to the ownership question. A supported
fact is still insufficient when it does not supply the requested attribute.
The two checks share one model request; they are not independent corroboration.

The separate treatment was also slower: hybrid median support time rose from
1,114 ms for the joint candidate check to 1,795 ms. Provider output-token counts
for the 20 hybrid observations were 2,310 and 4,257 respectively. The archive
retains provider-reported prompt, cached-prompt and output counts separately
from the typed trace's character-based estimates. These measurements do not
establish host-context efficiency or end-to-end application latency.

## Integrity, scope and delivery state

All 127 recorded source hashes matched after the run. Independent checks verified
80 groups of identical baseline facts, request/response accounting, all ten
accepted databases, and their reconstructed SQLite backup hashes against every
recorded snapshot hash. All ten databases passed SQLite integrity checks.
The foreign-project fact was excluded in every retrieval arm.

The largest rendered prompt was 3,752 UTF-8 bytes, below the 12,288-byte guard.
The largest provider context sequence was 910 tokens; all replies ended with
`stop`, and server logs recorded no truncation. The model was warm-loaded before
measurement. Prompt caching and treatment order affect timings; this is one
local development repetition, not a latency benchmark. The private model was
unloaded, its server stopped and its listening port verified closed.

Before inference, 76 focused tests passed, along with Black, strict mypy for
the three changed source modules and `git diff --check`. Tests cover stale-fact
rejection between candidate calls, aggregate deadlines, source contradictions,
strict response handling and unchanged default judge behavior. They establish
implementation contracts, not model quality.

The [archive](answer-support-development-v2.zip) contains the protocol, dataset,
source files, raw requests and responses, observations, full per-arm analysis,
failure audit, databases, reconstructed snapshots and runtime receipts. Its
[hash receipt](answer-support-development-v2-receipt.json) verifies every member.
Earlier archives and scores remain unchanged. No model, prompt or threshold was
retuned inside this run, and no release or live-host installation occurred.

Next development work must test the requested attribute and qualified answer
directly, including explicit negative answers, temporal distinctions and exact
source support. Any new prompt or backend comparison needs another frozen
development protocol. The 90% recall and 90% abstention targets remain unchanged;
fresh evaluation, separate extraction results, native lifecycle acceptance,
context-efficiency measurements and the real daily-use pilot remain open.
