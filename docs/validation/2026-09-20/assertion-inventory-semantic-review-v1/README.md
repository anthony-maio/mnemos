# Source-only inventory semantic review

All 46 calls are runtime-valid. The candidate is not adopted. It improves
marker retention in both mixed-message cases, retains correction history, and
correctly discards the unaccepted assistant suggestion. It still retains
unrequested speculation in N02 and unwanted fixture material in benchmark-date-10.
It discards continuing handling conditions in temporary-plan-11-policy and P09
as temporary instructions, and produces independently ambiguous fragments such
as It currently supports batch input only in P10.

review.json contains all 46 original sources/rubrics, inventories, retained
outputs and descriptive case reviews. Original ambiguity notes remain in the
terminal archive. This is unblinded development analysis, not an aggregate
semantic accuracy score, unseen acceptance or production adoption.

Exact quote presence does not guarantee that the quote supports every clause
of the memory text. Several items expand a short quote using other parts of
the same event. Some retained qualifiers are split away from their subjects.
Those are distinct from unavailable source IDs and are not hidden by runtime
validation success. No facts were committed and no reconciliation was run.
The absence of unrelated target selection at a stage with no targets does not
prove support/correction correctness.

The original scope-validation gap is documented in the sibling
assertion-inventory-scope-review-v1.md. All frozen inputs are consistent and
pass the stronger offline check; this run used the unchanged original candidate.
Next experiments must preserve these failures and distinguish qualification
attachment, retention eligibility and target reconciliation.
