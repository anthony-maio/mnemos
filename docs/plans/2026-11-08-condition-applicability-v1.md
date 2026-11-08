# Separate prerequisites from question circumstances

The readable-query failure assigned the same matches/applies state to valid conditional answers and unsupported answers whose prerequisite was unstated. Filtering that state cannot meet both precision and retention. The new development module `benchmarks/local_memory/condition_applicability.py` therefore evaluates source prerequisites against a separate interpretation of the circumstances asked about. It does not accept an applicability label from a model.

Source conditions use explicit subject/relation/value predicates and bounded conjunction, disjunction and negation. Query circumstances supply signed predicates only when interpreted as constraints on the case being asked about. A case constraint does not establish that the circumstance is true in the world; the answer must retain that scope. Mentioned or unresolved text cannot satisfy a prerequisite. Exact predicate identity is required; the evaluator does not guess aliases or treat another subject's state as evidence.

Missing or conflicting circumstances yield unknown for the predicate. Negating unknown remains unknown. Conjunction is false when a required clause is false and otherwise remains unknown until all clauses are true. Disjunction is true when an alternative is true and otherwise remains unknown until all are false. Incomplete source interpretation cannot become unconditional support, and incomplete query interpretation cannot authorize a conditional answer. No expression means unconditional only when source interpretation is explicitly complete.

Inputs carry separate source/query hashes and a shared interpretation-policy revision. Every atomic source clause and query circumstance has an exact character-offset span in its own text. All branches are validated, even if another branch would decide the logical result. Invalid locations, stale hashes, mismatched policies, malformed predicates/operators and excessive depth or size are rejected.

## Evidence and limits

All 26 tests passed in 0.71 seconds, including all nine combinations of true/false/unknown for conjunction and disjunction, negation, missing circumstances, mentions, conflicting polarity, foreign subjects, incomplete interpretation, invalid spans and bounded expressions. JUnit is `.venv/condition-logic-v1.xml`. Black and diff checks passed.

These are model-free logic checks over authored interpretations. They do not show that a model can extract complete prerequisites, distinguish a requested hypothetical case from a report or incidental mention, resolve entity identity, or normalize paraphrases. Span binding proves location, not entailment. A parser can still assign a wrong predicate or mode to an exact quote. The completeness flags also require independent evidence; an untrusted self-declaration is insufficient.

Before this can affect answer selection, develop and evaluate the source and question interpretation stages separately, retaining full input and provenance. Include negation, reports, unrelated mentions, hypothetical requested cases, conflicting constraints, aliases, conjunctions, alternatives and unsupported syntax. An unsupported interpretation must remain unresolved. Test real conditional-answer retention alongside abstention, then measure the complete path against the original latency requirement. No claim of support quality, extraction correctness or latency acceptance follows from these unit tests.

This candidate is separate from all frozen evaluators and results. It is not connected to production retrieval, the existing source inventory or the optional pilot. The original goal and all acceptance requirements remain unchanged.
