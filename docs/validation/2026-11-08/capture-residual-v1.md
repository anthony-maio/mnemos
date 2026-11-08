# Capture latency outside identity resolution

Status: verified offline accounting of the earlier instrumented command run. The cache remains inactive. No runtime code, live settings or pilot data changed.

The project-identity cache still requires an owner to invalidate external Git policy, including global/system configuration and includes. Source inspection confirms that no such provider is wired into the candidate. A constant generation would bypass necessary changes, so the prototype cannot be enabled as it stands.

The recorded command samples also show why identity resolution should not be the sole latency priority. Subtracting that stage within each sample leaves substantial measured work. Subtracting the separate post-dispatch interval as well still leaves every measured command above 100 ms.

| Recorded interval remaining | Python p95 ms | Native p95 ms |
| --- | ---: | ---: |
| Full command | 647.82 | 624.79 |
| Outside identity resolution | 458.17 | 420.26 |
| Outside identity and post-dispatch delay | 435.06 | 385.73 |
| Outside the entire service submit interval | 348.77 | 266.63 |

There are fifty measured commands per launcher, following five warmups each. Outside identity and post-dispatch delay, minimum remaining durations are 226.10 ms for Python and 100.18 ms for native. Outside the entire submit interval, all fifty Python samples and eleven native samples still exceed 100 ms. The native minimum at that boundary is 38.04 ms, so this is not evidence that a changed native implementation cannot meet the target.

These are per-sample subtractions, not differences between independent percentiles. Nested intervals are not added twice. Removing a recorded interval arithmetically is neither an implemented optimization nor a causal prediction: scheduling interactions, tracing overhead and runtime conditions can change. In particular, this older diagnostic predates the response-completion patch. Its totals must not replace the later actual installed measurement, which still fails at Python/native p95 of 598.64/621.55 ms.

The analysis verifies the original archive hash and exact archived timings, control, freeze and audit files. It verifies all 110 correlated response hashes and diagnostic records, the frozen schedule, stage nesting and the original p95 totals. The supplementary archive preserves the analysis script and output. The first verifier attempt assumed nested archive paths; inspection found root-level members, and the corrected check uses their exact names. No earlier evidence was rewritten.

The next capture investigation should attribute launch, transport, response handling and flush costs on the current installed candidate before extending the cache. Preserve authoritative Git routing, registry authorization and durable flushes. Native stdout alone does not prove a queued receipt; continue correlating event identities and final job completion. This changes investigation priority, not any acceptance threshold.

See the [receipt](capture-residual-v1.json), [original boundary diagnostic](queued-boundary-diagnostic-v1.md), and [later installed timing](response-completion-command-timing-v1.md). Full quality, timing, fresh acceptance and actual pilot requirements remain open.
