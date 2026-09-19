# Why the graph experiment has no eligible cases

This is a read-only diagnosis of the frozen `fc72cc2` mechanism experiment.
The helper verifies the archived ZIP hash and reads its original 150 graph
observations: 50 held-out cases with three repetitions each. It changes no
facts, labels, thresholds, edges, or model outputs and makes no provider calls.

All 150 observations meet the semantic floor and place the target outside the
direct vector/lexical candidate union. All 150 fail the remaining eligibility
condition: the actual query seed has no structural path to the target.
An unrestricted walk over every persisted edge confirms the target is
disconnected from that seed even when scope, depth, and energy limits are
ignored. This unrestricted walk is diagnostic only, never a retrieval policy.

The engine selected the fixture's intended seed in 30 observations; the other
120 selected another candidate. The target remained disconnected in both
groups. Neither the baseline nor the graph returned the target in any of the
150 observations. These counts describe this fixture and graph construction;
they do not estimate utility on ordinary user memories.

Lowering the activation threshold or increasing traversal depth cannot create
a missing edge in these recorded graphs. That rules out traversal tuning as
a sufficient repair for this result. It does not show whether the fixture's
intended association, the embedding geometry, or the graph construction is
the right thing to change. The existing independent-path and insertion-order
controls still support structural correctness, not associative usefulness.

Further development should first inspect association formation on the
separate development split, specifying which source-backed relationships
should create a usable path and why. Do not add held-out case IDs, forced
target links, or rewritten labels to manufacture coverage. Any changed graph
construction needs independent regression tests and a prospective experiment;
the original zero-coverage result remains failed.

`report.json` preserves each observation's reasons and selected seed.
`evidence-index.json` binds the report and executed helper. The original
[mechanism archive](../mechanism-local-v1/README.md) retains the raw vectors,
edges, observations, and failed gate.
