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

## Scoring and evidence

Report source inventory coverage, whole-fact support accuracy, qualification retention and temporal accuracy separately from final answer precision, critical retention, recall and abstention. Record omissions, additions and unsupported strengthening. A syntactically valid inventory with exact quotations can still be semantically wrong.

Use all scheduled query observations in availability denominators. Missing inventories, ambiguous conditions, transport failures and unattempted queries earn no successful-abstention credit. Preserve raw model replies and separate completed malformed replies from uncertain completion. First-pass timings include awaited annotation; reuse timings include lookup and query interpretation. Both exclude initial retrieval and therefore do not establish full retrieval latency.

Known development data cannot serve as fresh acceptance. Original quality, timing and daily-use gates remain unchanged.
