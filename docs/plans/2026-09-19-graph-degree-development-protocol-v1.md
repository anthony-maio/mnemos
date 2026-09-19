# Development graph degree diagnostic

Before collecting these observations, fix the following comparison: the
existing five development graph cases, their saved local Nomic vectors, and
neighbor budgets of 4, 8, and 16. Four is the current default. Eight and sixteen
are two and four times that budget. Run every case at every budget once;
deterministic repetition is not independent evidence. No held-out input is
opened, rerun, relabeled, or used to select a setting.

Use the verified `5e4b057b16906c2ec7dd302ae80748ef1ac5eddc` source snapshot
and production SQLite graph measurement helper. Bind saved vectors to their
development manifest, expected SHA-256
`938537dc4b615358a36c748a31667d477a6aa556f7e62137ed281a08caaa7698`.
Set the degree through the engine's existing configuration before inserting
any fixture fact. Assert the engine and graph use the same config object.
Use separate fresh storage for every budget; never resume an observation
under a different budget. Bind the helper, protocol, source, manifest, and
all effective configurations in the outer diagnostic report.

Keep the actual query-selected seed, mutual nearest-neighbor rule, 0.6 edge
threshold, 0.2 relevance floor, 0.3 activation threshold, 0.2 decay, three hops,
0.1 graph weight, baseline candidate budget, top-5 output, scope controls, and
all labels unchanged. Do not force an authored seed or a target edge. Use no
provider calls. Saved vectors are observations from the earlier model run;
this is an offline configuration diagnostic, not new model evidence.

For all 15 combinations, report structural paths, eligible status, activation
energy, direct-pool membership, baseline and graph result IDs, target rank,
edge counts, and the independent-path, scope, edge-cut, and insertion-order
controls. Record elapsed time as shared-machine development timing only.
Stop on execution or correctness failure and preserve every completed row.

Interpret all budgets together. Do not select a new production default from
the highest score here. The five cases are renamed variants of one template,
so they cannot establish generalization. If wider graphs restore paths but
not retrieval, report that scoring/seed behavior still limits utility. If no
paths form, widening this bounded budget is insufficient. A positive result
would justify further disjoint development and cost analysis, not a claim
that the original failed held-out experiment passed.
