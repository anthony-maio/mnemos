# Standalone spreading visit budget

Standalone activate and retrieve calls ignored SpreadingConfig.max_visited
when the caller omitted the method argument. They substituted
1 + max_hops * max_neighbors_per_node. A seven-node branching graph therefore
visited five nodes both when configured for one visit and when configured for
seven. Three regression cases fail before the repair; an explicit-override
control passes. The [baseline transcript](spreading-budget-baseline-v1.log)
reproduces these results against immutable source 2996d54.

The fallback now uses the configured max_visited. Explicit call overrides,
scope filtering, path energy, thresholds, hop limits and cycle handling remain
unchanged. All four new cases pass inside a 54-test graph selection; formatting
and Python 3.10 mypy pass. A separate source-association selection passes 73
tests. Full source/package verification is tracked separately.

The engine already passes its configured max_visited explicitly. This repair
does not explain or supersede the existing production retrieval failures.
The old 13-node supplementary measurement mismatch and original failed-control
experiments remain historical results bound to their original sources.

The separate graph diagnosis confirms a single semantic seed, even when the
caller supplies multiple allowed seed IDs. This is a coverage limitation, not
proof of incorrect score arithmetic or a measured benefit from multiple seeds.
The fixed-weight diagnostic retains 15 missing targets in 30 recorded scored
pools and only three optimistic top-five bounds. No weight or seed default was
changed to fit those cases.

The earlier source-association measurement used unprefixed vectors and found
no target paths. The later role-aware measurement separately tested unprefixed
and Nomic task-prefixed embeddings; only the latter reached the scope-isolation
target. The fixed-weight audit reads that later report. These different source
bindings explain the differing path counts; they are not interchangeable runs.
Neither experiment demonstrated top-five utility, and both preserve their
failed lifecycle control requiring retention of a forgotten shared source.
