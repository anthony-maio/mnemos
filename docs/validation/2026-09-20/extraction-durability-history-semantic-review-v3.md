# Durability and history v3: completed semantic review

All 92 planned calls completed across 46 reused development cases and two
prompt arms. Production returned 45 runtime-valid results and one error;
the candidate returned 46 runtime-valid results. Runtime validity is not
semantic accuracy. This review is unblinded, with one stochastic repetition.

The candidate is not adopted. It loses the independent marker in the mixed
ongoing-policy case, loses the explicitly requested open consideration in P05,
and retains unwanted fictional material in N04. Production has its own
omissions and a generation failure. Neither arm establishes release quality.

## Durability cases: 26 of 46

The complete case-by-case review, original sources, rubrics and proposals are
preserved in [the durability subset](extraction-durability-history-v3-durability-subset/review.json).
Those 52 observations are an exact prefix of the completed terminal record.
Its case-level findings remain unchanged. In particular, candidate policy
retention can improve while losing an independent fact in the same message.
Observed timestamp retention is not evidence of inventing a recurring schedule.
The original ambiguities remain unresolved rather than being relabeled to
favor either prompt.

## Positive history cases: 10 of 46

- P01: Both preserve concise project plans, decision-first ordering and short paragraphs.
- P02: Both preserve both release gates and the explicit unrun-remains-open condition.
- P03: Both retain PDF as current and DOCX as obsolete; add is appropriate because the existing control is unrelated.
- P04: Candidate preserves review date, QA attribution, fixture-only success and untested recovery. Production errors: empty final response at 8192 tokens with done_reason length. Do not treat this as an abstention or silently retry it away.
- P05: Both abstain despite the explicit request to retain the open cache consideration. This loses required unresolved project state; neither output falsely asserts adoption.
- P06: Both preserve disabled telemetry in the evaluation build and only possible different future production context.
- P07: Both retain baseline-art preservation and MVP scope but lose explicit assistant-reporting-user provenance in the text. Evidence IDs point to the assistant source. Preserve the existing rubric ambiguity; this proposal-only fixture does not establish service authorization to commit assistant assertions.
- P08: Both preserve fictional context and the real manuscript glossary consistency requirement.
- P09: Candidate retains possible schema bump, unconfirmed status and review boundary. Production retains unconfirmed status and review boundary in a compressed statement; it does not assert confirmation.
- P10: Candidate preserves earlier wrong streaming claim, current batch-only support and release exclusion. Production preserves current capability and release exclusion but omits the explicit prior-claim correction history.

## Negative history cases: 10 of 46

- N01: Both abstain, consistent with the frozen no-durable-facts rubric.
- N02: Both abstain, consistent with the frozen no-durable-facts rubric.
- N03: Both abstain, consistent with the frozen no-durable-facts rubric.
- N04: Production abstains. Candidate retains the explicitly fictional novel quotation. This is unwanted retention under the frozen rubric, not a claim that the real moon is glass.
- N05: Both abstain, consistent with the frozen no-durable-facts rubric.
- N06: Both abstain, consistent with the frozen no-durable-facts rubric.
- N07: Both abstain, consistent with the frozen no-durable-facts rubric.
- N08: Both abstain, consistent with the frozen no-durable-facts rubric.
- N09: Both abstain, consistent with the frozen no-durable-facts rubric.
- N10: Both retain that offline sync was not accepted and is not a current requirement. Preserve the preregistered negative-project-state ambiguity: neither asserts acceptance; a binary abstention failure would conceal correct negation.

## Evidence and limits

[Terminal evidence](extraction-durability-history-terminal-v3/README.md) retains
all raw outputs, exact settings, source bindings and failures. P04 production
exhausted the 8192-token output budget with an empty final response; it is an
error, not abstention. It has not been retried or replaced.

The [pre-execution ambiguity review](extraction-durability-history-terminal-v3/history-rubric-review.json)
continues to govern P07 and N10. Evidence IDs alone do not restore attribution
omitted from proposal text, and assistant proposals do not establish service
permission to commit an assertion. No facts were committed in this experiment.
No aggregate semantic-accuracy percentage is claimed. No held-out acceptance,
full-module recovery, host acceptance or production configuration change follows.
