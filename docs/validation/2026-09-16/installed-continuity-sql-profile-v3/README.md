# Installed continuity SQL timing diagnostic

Both frozen `312b8d6` continuity cases passed in 15.89 seconds against the original installed wheel, with the original client/service timeouts. The dedicated fixture was outside Git and all loaded package paths were verified against the installed environment. Providers were synthetic.

The profiler observed 100 commits with median 53.366 ms and maximum 368.297 ms. The 1670 SQL execute calls had maximum 1.410 ms. The slowest retrieval metadata update took 368.422 ms, including a 368.297 ms commit. This locates commit latency as the dominant storage cost in this run. It does not distinguish device flush latency from operating-system scheduling or other activity.

The earlier multi-second stalls did not recur. No product change was made between these diagnostics, so this passing run is not evidence of a repair or proof that the earlier failures cannot recur. The [valid timeout reproduction](../installed-continuity-diagnostics-v1/README.md) and original installed failure remain part of the record. Complete installed contract/replay verification and the 10,000-fact latency gate remain separate.

The helper added thread-aware Python profiling for SQLite calls and kept the preceding client/service/retrieval timings. It records operation labels and durations without SQL values. These are instrumented diagnostics on a shared machine, not isolated benchmark results. Raw files are unchanged.
