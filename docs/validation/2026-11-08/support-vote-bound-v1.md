# Support selection needs more information than two votes

An offline audit of the frozen retrieval/support study rules out every fixed Boolean combination of its QA and contrast selections. All 16 mappings from the four possible vote states to select/reject were enumerated. None reaches 95% precision, 95% expected-fact retention and 90% absent-answer abstention together. No model was called, no label changed, and no fresh case was used.

| QA selects | Contrast selects | Expected fact observations | Unsupported candidate observations |
| --- | --- | ---: | ---: |
| No | No | 2 | 144 |
| No | Yes | 6 | 12 |
| Yes | No | 0 | 14 |
| Yes | Yes | 72 | 4 |

The current intersection keeps 72 expected facts and four unsupported facts, reproducing the frozen result exactly. Trusting contrast alone would retain 78 expected facts but also select 16 unsupported facts, with correct abstention dropping to 40/56. Those are counterfactual counts from existing outputs, not new measured treatment scores. The bound applies only to a fixed decision using these two votes; it does not rule out methods using additional source, query or model information.

All four false selections occur on two unresolved current-duration conflicts, each repeated twice. QA selects both incompatible durations; contrast chooses one, and the intersection wrongly treats agreement on that one as sufficient support. Six lost answers are three questions repeated twice: two historical trial-window values and a retention-change reviewer. Contrast selects each correct fact, while QA rejects it. Both models miss the remaining historical-owner answer in both repetitions. Retrieval supplied every expected answer, so widening the candidate pool would not repair these losses.

The next semantic candidate must address query-specific ambiguity and historical applicability, not merely change voting, output encoding or a probability cutoff over the same binary choices. It must distinguish conflicts about the requested attribute from unrelated facts about that entity: conflicting retention durations must not suppress an explicitly supported reviewer. It must also retain historical answers within their stated validity intervals even when a newer value exists.

Any candidate needs a separate prospective design and a complete latency measurement. A query-conditioned check should retain source spans for the requested relation, temporal interval and incompatible values; those spans must come from the evaluated method, never from fixture labels at runtime. A global rule rejecting every multi-fact result or a hand-written list of these six questions would not meet that requirement. The existing roughly 8.6-second p95 support path also requires a different execution approach, not an extra sequential model call added without a timing budget.

The analysis verifies the raw observations, labels and inputs against the original archive member hashes before deriving results. It covers 136 observations of 68 known questions; repeated questions are not independent cases. The adjacent JSON retains every mapping, the twelve error observations and source hashes. The helper is preserved at `.venv/analyze-support-vote-bound-v1.py`, with its hash in that receipt. The original experiment remains closed and failed; this diagnostic does not reopen it or establish acceptance.
