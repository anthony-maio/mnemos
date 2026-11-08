# Relation inventory grading preparation

Status: pre-execution review. No new semantic responses have been generated. This is a grading specification, not a passing evaluation or a frozen label set.

The prepared cohort contains 68 development questions and 127 candidate observations. Keep final-answer labels separate from source annotations. Grade source records against the original fact and its own evidence, without using the question to decide what to extract. Grade query interpretation and final selection separately.

For each source record, record these judgments before inspecting model output:

- Whole-fact support: supported, unsupported or unresolved. Assess every clause, including proposal status, negation, attribution and quantities.
- Required propositions: the independently meaningful claims that must survive. Allow equivalent wording rather than requiring one canonical key spelling.
- Conditions and qualifications: the exact evidence that limits a claim, including environment, approval, implementation, trial periods and unverified reports.
- Temporal meaning: stated intervals, inclusive endpoint conversion, effective dates and unresolved temporal anchors. Distinguish a date in quoted test data from an actual scheduled change.
- Forbidden strengthening: claims the source does not establish, even when their subject and vocabulary overlap.

The retained inputs include two clear whole-fact failures: an ownership claim whose source only records meeting attendance and explicitly says no owner was assigned, and a completed pipeline replacement whose source describes an unapproved proposal with no implementation. Preserve these original facts as unsupported; rewriting them into weaker supported facts would change the evaluation target.

Matching fact and source text supports faithful quotation of that statement, not independent truth. An unverified report supports the fact that someone reported a result; it does not establish that the result occurred. Likewise, an instruction inside a test fixture remains quoted data.

## Temporal contract review required before execution

Many retained sources state current or ongoing facts without an absolute observation date. The source producer receives no query reference date. Its existing temporal categories are bounded, unbounded and unresolved, and its prompt requires positive justification for unbounded validity. The query decision rejects relevant unresolved temporal claims.

This creates a potential representational gap: a source's present-tense assertion may answer a current question without establishing timeless validity. Do not label it unbounded merely to increase recall, infer an observation date from the query, or silently relax the rule after seeing output. Resolve whether the contract can represent this distinction before freezing source temporal labels and running semantic evaluation. Any changed schema or prompt requires a new prepared policy version; preserve the existing preparation and tokenizer evidence.

### Explicit temporal contract

`benchmarks/local_memory/temporal_support.py` now distinguishes an explicit half-open interval, a present-tense assertion, a timeless statement and unresolved time. It requires an exact unique own-source quotation for any resolved temporal assessment. An assertion date must be established by that source; ingestion timestamps and query reference dates do not establish it. Exact quotation alone still does not prove that a model interpreted the date correctly.

Query intent is separate: an explicit request for what retained evidence says, a request for validity on a date, or unresolved intent. A question asking about current validity belongs to the dated branch after resolving its reference date. It must not fall back to source-state intent merely because evidence lacks a date.

An undated present-tense assertion can answer an explicit source-state question with basis `source_assertion`. It cannot prove a dated/current answer. An explicit as-of assertion supports its stated date but remains unresolved on other dates; no implicit persistence is assumed. An interval supports dates within its bounds and excludes dates outside them. Unresolved time never becomes successful support. The result preserves its basis so a caller cannot confuse a source assertion with established dated validity.

The new temporal contract and existing query/cache/quotation checks passed 62 tests in 1.92 seconds, retained at `.venv/temporal-support-v1.xml`. Tests cover undated assertions against past/current/future dates, as-of dates without extrapolation, half-open endpoints, incompatible fields and foreign quotations. These establish deterministic behavior, not semantic extraction accuracy.

The contract is now integrated in `relation_inventory_cache_v2.py`, `source_relation_producer_v2.py` and `relation_query_v2.py`. These versioned modules preserve the frozen v1 implementation. V2 source relations replace the old applicability fields with a structured temporal annotation. The strict response schema rejects legacy temporal fields, validates own-source quotations and keeps the source producer independent of query dates.

The v2 query response carries explicit temporal intent. Selected results retain a per-claim `selected_bases` record with fact, source, claim index and temporal basis. Unavailable, absent and conflict results carry no successful support basis. The classifier remains fallible: a model that incorrectly labels a current-validity question as source-state would violate the protocol and must fail semantic grading, even if its JSON is valid.

Nine v2 source-to-query checks passed together with the temporal contract and existing boundaries: 127 tests in 2.86 seconds, retained at `.venv/relation-inventory-v2-integration.xml`. They cover undated assertions against past/current/future dates, source-state support with an explicit basis, conflicting query intent/date fields, as-of support without next-day extrapolation, rejection of legacy fields and foreign temporal quotations. These are model-free integration tests.

Existing v1 preparation, schemas and tokenizer evidence remain unchanged. `relation_inventory_run_v2.py` now connects the versioned producer and selector with the same bounded two-pass schedule and uncertain-completion stop rule. Its durable-journal integration test verifies that per-claim temporal bases survive serialization. The v1/v2 boundary and harness checks passed 136 tests in 3.51 seconds, retained at `.venv/inventory-v2-harness.xml`.

The new preparation at `.venv/relation-inventory-development-v2` retains 68 questions, 127 candidate observations, 127 distinct source requests and 136 planned query observations. Maximum source prompt size is 5,616 UTF-8 bytes; this has not yet passed the model tokenizer. Both policy schemas bind the temporal implementation and v2 harness, with seed 2026110806. Manifest SHA-256 is `ce763184577d0306ab1fa528387440b0e9d075c19d3629e9d6289489b98e9a83`; policy-file SHA-256 is `02fd3b534c5a7684127e53459975b45af1f539de81de48cf2a63767c814f538d`. No semantic calls have run.

Source-grading targets are prepared separately from answer labels for all 127 requests. There are 123 verbatim whole-fact/source matches and four unsupported observations covering the two contradictions described above. The target file preserves each original proposition, own excerpt and support rationale. It also requires recorded semantic judgments for claim coverage, additions, qualification retention, temporal meaning and normalized-key meaning. Those dimensions require manual source-based review; no automated semantic accuracy is claimed. The targets' SHA-256 is `29195a64ed1b9291a8daf111c396ee4d531423f75812069a182f778bc314e9f9`.

The owner runtime wrapper, v2 tokenizer checks and execution freeze remain outstanding. The final freeze must bind source-grading targets as well as answer labels, without exposing either to producers. Any additional structured temporal labels must precede inference. Current evidence objects contain excerpt and source references but no assertion-date field; do not invent dates to fill that gap. If sufficient source timing is absent, report the resulting recall/availability limitation against the original gates. Do not redefine the original goal as source-state recall alone.

## Scoring and evidence

Report source inventory coverage, whole-fact support accuracy, qualification retention and temporal accuracy separately from final answer precision, critical retention, recall and abstention. Record omissions, additions and unsupported strengthening. A syntactically valid inventory with exact quotations can still be semantically wrong.

Use all scheduled query observations in availability denominators. Missing inventories, ambiguous conditions, transport failures and unattempted queries earn no successful-abstention credit. Preserve raw model replies and separate completed malformed replies from uncertain completion. First-pass timings include awaited annotation; reuse timings include lookup and query interpretation. Both exclude initial retrieval and therefore do not establish full retrieval latency.

Known development data cannot serve as fresh acceptance. Original quality, timing and daily-use gates remain unchanged.
