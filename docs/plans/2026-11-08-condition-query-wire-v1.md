# Query circumstances with their own evidence

`benchmarks/local_memory/condition_query_wire.py` adds a separate development boundary between question interpretation and condition evaluation. The request carries the source condition tree, exact source and query text, a deduplicated predicate catalog, interpreter identity and an owner-generated request binding. The model sees only the question and predicate catalog, not source prose from which it might assume the requested circumstances.

Every predicate requires one of three states: positive, negative or not established. Positive and negative require a unique exact quotation from the question. The decoder computes character offsets locally and rejects absent, empty or ambiguous quotations. Not established requires a null quotation and supplies no circumstance; the logic layer consequently preserves unknown. Source text cannot substitute for missing question evidence.

The decoder rejects omitted or foreign predicates, duplicate JSON keys, extra fields, stale/tampered requests, cross-request replies and incomplete transport. It invokes the live fence before and after decoding. Full source clauses are validated during request preparation; incomplete source interpretations block this step, and unconditional sources require no circumstance call. Model-generated request identities or applicability decisions are not accepted.

All 43 selected boundary and condition-logic tests passed in 0.95 seconds. Coverage includes independent positive/negative/unknown states, absent question evidence, repeated quotation ambiguity, complete predicate coverage, invalid witnesses, request binding and source revocation. JUnit is `.venv/condition-query-wire-v1.xml`. Black and diff checks passed.

This boundary validates provenance and structure, not meaning. An exact quote can still be irrelevant, negated, merely reported or hypothetical in a way that does not constrain the requested answer. The decoder does not independently prove that the model's polarity or case interpretation is correct. The predicate catalog and source completeness are also supplied interpretations, not validated extraction results. Semantic tests must grade both the state and the witness against the whole question, including those distinctions.

No model call has run for this boundary, and no latency or accuracy claim is established. The next screen needs separate development inputs and frozen labels, with explicit requested hypothetical cases distinguished from incidental or reported mentions. It must account for all prerequisite predicates and retain valid conditional answers. A final implementation still needs separately validated source extraction, relation matching, temporal reasoning and complete-query timing; adding this as another sequential call to the failed readable path is not an accepted solution.

Frozen candidate code/results, canonical runtime and live pilot state remain unchanged.
