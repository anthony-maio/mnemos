# Installed performance check preparation

At this preparation checkpoint, the performance check had not run. It targets the installed
`749f1213` package, local BGE embeddings and a separate runtime on the system
volume. The [protocol](../../plans/2026-11-07-installed-performance-v1.md) keeps
the original 500 ms warm-recall and 100 ms durable-hook requirements.

The older performance harness uses 768-dimensional Ollama embeddings and load
vectors derived from one seed. Its warm capture measurement covers HTTP, with
only one separate cold hook invocation. Those results remain valid for their
recorded setup; they do not establish performance for this configuration.

The prepared runner uses 10,000 synthetic accepted facts with source excerpts
and real document embeddings. It measures fifty complete retrieval requests,
fifty capture-only hook commands and fifty prompt hook commands with recall
enabled, with separate warmups. Hook timers include interpreter startup and
configuration checks. Durable event and job checks follow each command, then
repeat after a service restart.

The model-free preflight verified 90 installed package files and 57 dependency
versions. Its database probe accepted an actual adapter capture and rejected a
missing event and a mismatched source. The probe used a direct capture-client
stub; it did not test HTTP, native host execution or performance. Percentile and
failure-accounting checks passed. The runner's overlap guard refused setup
while the original retrieval evaluation was still running.

V1 subsequently stopped at the service's embedding-identity guard before any
timed requests. See the [failure and fixture repair](installed-performance-v1-failure.md).

The subsequent [V3 result](installed-performance-v3.md) completes the workload:
warm recall meets its target, while both complete hook commands miss theirs.

No timing result, new model inference, pilot day or readiness claim follows
from this preparation. Run only after the fresh retrieval comparison reaches a
terminal state and its audit and archive are complete. The installed checkpoint
still needs its remaining native and quality checks.

- [Preparation receipt](installed-performance-v1-preparation-receipt.json)
- [Installed package checkpoint](checkpoint-749f1213.md)
- [Fresh retrieval comparison](fresh-accepted-v1.md)
