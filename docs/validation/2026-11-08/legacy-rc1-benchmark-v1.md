# Legacy prerelease benchmark

The exact 0.6.1rc1 package failed the unchanged legacy production gate. Eleven of twelve dataset/store comparisons passed across two repetitions. In the first repetition, the in-memory contradiction-update comparison had zero MRR lift against the required 15%. The second repetition passed all six comparisons. All latency ratios passed. The process exited 2, the benchmark's gate-failure status.

This is a release check on known synthetic fixtures. It does not establish fresh retrieval quality, answer support, abstention, complete host latency, or pilot acceptance. The original 0.6.1 failures remain preserved. No retry was run to replace this result.

The candidate is commit `c36ca712c47e9c8c0f46226e74ad805fd7be0cef`. Its wheel SHA256 is `939427361b3afdd816f88e1c05e8b7b0664fcdf7f878e5c9691fa767a63a0080`. The test used an isolated copy of the previously verified installed wheel on Windows Python 3.11.13, simple 384-dimensional embeddings, and mock LLM settings. All package files matched wheel bytes before execution and remained unchanged afterward.

The wheel does not bundle the named dataset pack. The three unchanged repository fixtures were copied beside the isolated package so the existing pack resolver could load them. This check therefore does not validate a standalone installed pack command. Package code, fixture content, thresholds, scoring, and repetitions were unchanged. The command selected memory and SQLite stores, baseline and engine retrieval, top-k 1, and two repetitions with production-gate enforcement.

The private evidence directory retains the complete report and per-query latency observations, console log, fixture and package hashes, dependency inventory, helper and runner hashes, exact command, and terminal result. HTTP and non-loopback socket connections were blocked; numeric loopback was allowed for Windows asyncio. The test used disposable configuration and databases. No live configuration or pilot was changed.

Release readiness remains unresolved. The separate six-environment installed checks passed 42 distinct cases in each environment, but those checks do not override this failed gate. The original quality, host latency, and actual two-week pilot requirements remain open.
