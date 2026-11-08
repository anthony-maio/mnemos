# Conditional-selection filter bound

No deterministic filter using only the readable probe's returned relation judgment, returned condition judgment and presence of source conditions can meet both 95% precision and 95% expected-fact retention on its existing selections. An offline audit enumerated every mapping of the observed states to keep/reject. No model was called, no result was rewritten and no acceptance case was used.

| Existing selected state | Expected observations | Unsupported observations |
| --- | ---: | ---: |
| matches, applies, no source conditions | 6 | 0 |
| matches, applies, source conditions present | 2 | 2 |

Keeping both states reproduces 8/10 precision and 8/8 retention. Rejecting all conditional selections gives 6/6 precision but only 6/8 retention. Keeping only conditional selections gives 2/4 precision and 2/8 retention. Rejecting everything leaves precision undefined and retention zero. None passes both gates. These counts are repeated forward/reversed observations, not independent scenarios.

The result rules out this specific output-only repair. It does not rule out methods that use additional query/source semantics or evidence. In particular, the valid and invalid conditional selections have identical returned states; correcting them requires information that distinguishes whether the question establishes the prerequisite. The earlier source-aware prompt already explicitly required unresolved applicability when the question does not establish it. Another restatement of that instruction is not yet a demonstrated mechanism.

A next candidate should represent prerequisites and question-established circumstances separately, retain their exact provenance, and compute applicability with an explicit unknown state. Exact quotations alone do not prove entailment: a quote can be unrelated, negated or hypothetical. Tests must include those cases and preserve legitimate conditional answers. The interpretation stage must also fit the remaining complete-query latency budget; adding another sequential call to the failed 1.85-second median path is not an established solution.

The audit read the original hash-chained events, matched selected facts to their recorded claim judgments and source conditions, and checked expected facts against the frozen grading file. Grading SHA-256 remains `fbd4c9f17fe9fa339c4f9544d80ccb192745fcd2014528c6de85b2a8b4c10878`. The [receipt](readable-condition-filter-bound-v1.json) retains all four mappings, the ten selected observations and original evidence hashes. Helper: `.venv/readable-condition-filter-bound-v1.py`.

The original probe remains closed and failed. This is a bound on filtering existing selections, not a new treatment score, complete support architecture or fresh acceptance result. No runtime, package, host configuration or pilot state changed.
