# Fair offline retrieval comparison diagnostic

This archive records a bounded offline control for the claim-driving retrieval benchmark on the frozen snapshot `39b9c799020a58ea2ac3d798189474ed10389bd0`. It does not establish production quality, model quality, or a release gate.

The original preserved reports were run with a stateful simple embedding provider and different scope behavior. The diagnostic used `baseline_scope_aware=true`, top-k 1, two repetitions, memory and SQLite stores, baseline and engine retrievers, and the three claim-driving datasets. The valid v3 helper injected one stateless SHA-256 token-bucket provider into both benchmark paths. Document and query vectors came from the same 42-entry corpus map, with one separately recorded deterministic fallback for an engine-internal symmetric prediction string.

Attempt status:

- v1 completed but is invalid as a fairness control. It generated the vector manifest without injecting that provider into the frozen benchmark, so the run used the snapshot provider. Its report is retained for audit.
- v2 attempted the provider injection and failed closed before report creation when the engine requested an unlisted internal symmetric prediction string. Its helper and exact failure are retained.
- v3 completed with provider injection and the internal deterministic fallback. Its report and vector manifest are retained byte-for-byte.

Valid v3 outcome:

- 24 runs and 184 per-query outcomes were recorded. Source tree hash was unchanged before and after: `408100623b27f6039858baf757f77d0207cd0e5863bebf236a3f5764ecf475f5`.
- Cross-project scope MRR was 1.0 for both baseline and engine in both stores and both repetitions. The earlier negative cross-project result did not survive this fair control.
- Mean MRR for contradiction was baseline/engine 0.0/0.1875 in memory and 0.5/0.25 in SQLite. Mean MRR for preference was 0.0/0.75 in memory and 0.75/0.4375 in SQLite. These synthetic results vary by store and path and do not support a consistent engine quality conclusion.
- The six production replacement comparisons failed overall in each repetition. That gate is retained as an unchanged benchmark calculation; zero-MRR baselines and synthetic vectors make it unsuitable as a production claim here. Latencies are descriptive only.

The valid v3 helper SHA256 is `9aa39a1d8d6195a6a45085edbdef6a46312f0f6b5d8892648f18c56b68767cb9`. The report binds the snapshot commit, source tree hashes, dataset hashes, provider map hash, and Python executable. No model or network calls were made. SQLite fixture directories remain outside this archive and are not copied.
