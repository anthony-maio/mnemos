# Query-independent relation inventory development

Status: an in-memory record boundary and source-only producer are implemented and tested. No model has produced inventories for this design, no latency gain is claimed, and nothing is connected to production retrieval or capture.

The closed relational screen generated a complete source inventory during each query and exceeded its first 15-second deadline. It also exposed unreliable model-generated character offsets. This design separates query-independent source annotation from query interpretation and uses exact unique quotations for source locations. It is an architectural change requiring new evaluation; it does not reopen or regrade the failed screen.

## Implemented boundary

`relation_inventory_cache.py` reuses `qualification_key` and `QualificationIdentity`. Keys bind the complete fact, version, scope, fingerprint, all retained own sources and policy/model/runtime digests. A change to a sibling source also invalidates the record. The cache stores only an explicit complete or unresolved inventory for one own source. Provider failure is not a valid inventory status.

A source relation contains canonical subject/relation/value keys, exact source quotations, any source-stated half-open validity interval, whole-fact support assessment and the source's condition quotations. It has no query text, requested date, selected answer or condition-match result. Source annotations can therefore be reused across questions without reusing one question's applicability judgment. Unbounded temporal validity must be explicit; uncertainty remains unresolved.

Quotations must bind uniquely to that fact's own excerpt. Supported whole facts require a whole-fact quotation, and temporal boundaries require a temporal quotation. Exact text does not prove entailment or the semantic correctness of normalized keys. The next model evaluation must grade those judgments separately.

Records are immutable tuples with at most 32 claims and 32 condition quotations per claim, bounded to 64,000 serialized UTF-8 bytes. The owner-local cache defaults to 256 records and uses FIFO eviction. It is neither persistent nor a background worker. The owner serializes calls and supplies live validation before and after record/lookup work. Lookup preserves the supplied candidate/source order and only annotates that current cohort. Missing, changed or evicted entries are missing work, not unsupported facts. An omitted candidate is never reconstructed from historical records.

The cache, quotation resolver and reused qualification boundary passed 59 checks in 3.24 seconds. JUnit is retained at `.venv/relation-inventory-v2.xml`. Checks cover reuse across different questions without conditional answers, correction/content/scope/source/reference/sibling changes, policy/model/runtime changes, omitted candidates, denied scope fences, failed postvalidation, conflicting records, eviction and byte limits. These tests establish control flow only.

## Source-only producer

`source_relation_producer.py` receives the original fact, all its retained evidence and the policy/model/runtime identity. Its API has no query or answer labels. The prompt asks for conditions, modality, negation and attribution to be preserved and whole-fact support to cover every original clause. This instruction remains an untested semantic requirement, not a guaranteed property of model output.

The strict response schema binds the request key, fact and target evidence. It rejects query condition-match results and model-generated offsets. The cache validates exact unique quotations and source intervals before recording. Requests and replies are each bounded to 64,000 UTF-8 bytes without truncation. Source and authorization fences run around cache access, provider work and recording. An unresolved inventory remains unresolved on reuse; provider errors cannot become negative facts.

The injected provider has a 15-second cooperative timeout. Cancellation propagates, while ordinary failures return unavailable without recording an inventory. A provider that delays cancellation can exceed that deadline; a future runtime harness must record actual wall time and cleanup rather than claiming a hard bound. There is no concrete model connection, persistent worker or capture integration.

The producer and existing cache, quotation and qualification boundaries passed 73 tests in 1.93 seconds. JUnit is retained at `.venv/source-relation-producer-v2.xml`; the earlier 69-test run remains at `.venv/source-relation-producer-v1.xml`. Tests cover one-call reuse, request identity, foreign/stale replies, invalid quotations and fields, source changes during generation, provider failure/cancellation, denied cached lookup, unresolved reuse, timeout cleanup and oversized input rejection. These are model-free control-flow checks.

## Remaining implementation and evaluation

Implement query interpretation and condition applicability separately, using every eligible original candidate and inventory, including unresolved and unsupported annotations. Conditions must be compared with the actual query; unresolved conditions get no successful-abstention credit. A missing inventory cannot silently remove a competing value. Query-specific conflict and historical-interval rules remain required.

Freeze the model/runtime identities, both prompts and response schemas, exact input cohort, decision rules, limits and request schedule before semantic evaluation. Grade source annotation accuracy separately from final precision, critical retention, recall and abstention. Reuse known development inputs only as development; fresh acceptance remains unopened until a complete candidate passes.

Measure cold annotation cost, warm lookup, invalidation, misses and complete retrieval including query interpretation and condition checks. If a query waits for annotation, include that work in its timer. If annotation is asynchronous and unavailable, report unavailable and include it in availability denominators. Report annotation backlog/processing cost separately and preserve durable queued versus completed status. A cache hit or a cheap deterministic comparison is not proof of the original 500 ms complete warm retrieval target, 100 ms durable capture target, semantic gates or daily-use pilot.
