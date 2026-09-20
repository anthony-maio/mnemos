# Alternative graph seed structural audit

The frozen role-aware development measurement contains five cases, two embedding
contracts and three graph arms. This read-only audit examines repetition zero:
30 combinations, not 30 independent cases. Original failures remain unchanged.

Among the original vector candidates, the counts with a non-target seed that
has a structural path to the target within the existing three-hop limit are:
top one: 3; top three: 3; top five: 3; top ten: 6; all thirty: 18.
The target itself is excluded as a seed. Query-eligible fixture labels match
the production scope policy. An independent depth-first walk verifies the
breadth-first audit's reachability results.

The prefixed shared-capture case first gains such a seed at vector rank 6;
other newly reachable combinations require ranks 23-28. The semantic-only-chain
case remains disconnected from every non-target candidate in both embedding
contracts. This rules out merely adding the top three or five seeds as a
sufficient structural repair for these recorded cases.

These are optimistic structural bounds. The audit does not execute multi-seed
retrieval or allocate its visit/energy budget, apply intermediate activation
thresholds, reconstruct ranking, or measure recall. It does not justify enabling
more seeds or changing scoring weights. Candidate scoring, association formation
and embedding quality still need development evidence. No provider calls,
production settings, dataset labels, thresholds or acceptance criteria changed.
The prior zero-top-five result and 18 lifecycle control failures remain open.
