# Installed pilot command timing

The proposed Python command and experimental native v3 command were measured
on the same installed `f5a944ea` wheel and 10,000-fact snapshot. The Python arm
uses the inactive pilot's interpreter, exact launcher bytes and settings, with
only disposable storage paths and the synthetic project registration relocated.
The [protocol](../../plans/2026-11-07-installed-pilot-command-timing-v1.md) was
committed before measurement. Each event/arm has 50 timed observations across
two fresh service processes, with independent warmups and reversed block order.

| Boundary | Python median / p95 ms | Native experiment median / p95 ms | Python / native gate |
| --- | ---: | ---: | --- |
| Complete recall | 732.98 / 1098.84 | 744.66 / 1508.41 | FAIL / FAIL (500 ms) |
| Complete Stop command | 859.55 / 1359.04 | 206.00 / 360.88 | FAIL / FAIL (100 ms) |
| Complete prompt command | 1821.95 / 2992.09 | 1334.81 / 2036.34 | FAIL / FAIL (100 ms) |

Command timing begins before executable launch and ends after process exit and
stdout/stderr closure. It excludes native-host dispatch and outer PowerShell
startup. Recall timing includes client transport, local embedding, retrieval
and context construction. It is measured separately from hook startup. These
are development measurements on existing synthetic data, not semantic quality
or native-host acceptance.

The real local BGE embedder runs during retrieval; no answer-support or
extraction LLM is called.

The service observer records the installed transactional persistence call in
memory, then flushes its spans after shutdown. It changes no package bytes and
adds no per-capture disk logging. Both arms use the same observer. Its
in-memory overhead is included in complete command timing. The component begins
inside the service; dispatch-to-return includes the earlier command and client
work. Neither exposes an early durable acknowledgment to the host.

| Event and arm | Transaction component p95 ms | Dispatch to transaction return p95 ms | Complete command p95 ms |
| --- | ---: | ---: | ---: |
| Stop, Python | 3.79 | 1297.36 | 1359.04 |
| Stop, Native experiment | 3.52 | 289.91 | 360.88 |
| Prompt, Python | 3.42 | 1678.20 | 2992.09 |
| Prompt, Native experiment | 3.43 | 817.10 | 2036.34 |

The component columns are diagnostics. They do not replace the 100 ms complete
command gate. Successful Stop stdout is not itself a persistence receipt.
The audit correlates all 208 captured events with exactly one completed
transaction span inside the corresponding command interval, and verifies the
event and job after service restart. Every job is completed; no pending outbox
entry remains at command completion. This run does not exercise an unavailable
service or establish queued capture latency.

All 320 observations are retained, including 300 timed observations and 160
exact output/context pairs. The audit verifies 104 direct Python commands and
104 handled native worker dispatches with matching payload/output hashes and
no fallback. Accepted facts, evidence, sources and vectors remain unchanged.
Both stores use the evidence index. Package files, model assets and watched
configuration files retain their hashes. All owned processes have stopped.

Service/worker startup-to-health timings are retained separately in the
[receipt](installed-pilot-command-timing-v1-receipt.json); those include process ownership checks and
are not pure initialization costs. There is no retry or second measured run.
The benchmark does not control other operating-system activity, and the results
do not establish that timing changes from earlier runs were caused by a product
change. The prior evidence-index comparison and all earlier FAIL labels remain
unchanged.

Gate results: recall: Python FAIL, native FAIL; Stop: Python FAIL, native FAIL; prompt: Python FAIL, native FAIL.

The full pilot still has failed quality entry checks. The earlier bounded
retrieval/support comparison is closed; this timing work does not reopen it.
The pilot remains inactive at zero days, normal native hook review remains
pending, and no live configuration or publication changed. A separate explicit
MCP learning pilot would need its own scope decision and cannot satisfy the
original quality or two-week pilot obligations.

Private archive SHA-256: `00b8db784fd242500b561611e12b1c5318beb2b78ea168e8cc2485476425c6c3`.
