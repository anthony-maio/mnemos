# Association measurement visit-budget repair

The first development run's supplemental activation observation omitted the
configured visit budget. Production retrieval used `max_visited=1000`, while
the helper inherited the standalone limit of `1 + 3 * 4 = 13`.

A synthetic 15-node shared-source group reproduces the defect: the independent
oracle returns 15 nodes and the observation returns 13. The red JUnit report
retains that exact failing assertion. The repair passes the existing configured
`max_hops` and `max_visited` to the supplementary activation call.

All four measurement tests pass afterward, including full three-arm controls,
semantic parity, restart, and the new wider traversal. Black and strict Python
3.10 mypy pass for the two changed files. Production code, scoring, graph
construction, cases, vectors, thresholds, and acceptance criteria are unchanged.

The [first run](../source-association-measurement-v1/README.md) remains failed.
A fresh output directory is required for remeasurement. The separate failed
shared-source preservation control remains unchanged and must stay visible.
