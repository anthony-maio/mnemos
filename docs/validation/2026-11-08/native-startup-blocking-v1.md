# Native validation waits for MCP startup

Runner commit `98a6a7edf2532940f77ab6e8bfe29550b834e64c` explicitly sets
MCP_CONNECTION_NONBLOCKING=false in disposable Claude phase environments.
This applies the startup behavior demonstrated by the actual patch-host check;
it does not change live host configuration or the product's runtime defaults.

Three inherited-environment cases failed before the change (unset, true and
false) because the child environment did not guarantee blocking startup. All
52 phase-process tests pass afterward. Black accepts both changed files, mypy
accepts the runner module, and git diff --check passes. Logs/JUnit are retained
under `.venv/native-blocking-*-v1.*`. No new model or native lifecycle run was
performed for this runner change; earlier frozen results retain their bindings.

The full functional workflow is not wholly unverified: Claude v5 passed on
6d0e5ba6, and the later Codex v12 passed six phases on f5a944ea, including
correction, forget preview/apply and original-source replay suppression. Latest
25126632 has installed authorization-fix tests but no exact native acceptance.
Prefer a focused long-lived-session revocation/re-enable check when validating
those changes, and a full lifecycle at actual delivery acceptance. Avoid another
unchanged lifecycle matrix while semantic quality and complete latency remain
the larger unresolved requirements.
