# Native Hermes lifecycle v2: wrong job identity

The fresh successor stopped at teach with status `failed`. It used verified
runtime `3730fd9`, the same pinned wheel and models, and runner snapshot
`1418038bc91c28ce5db74e26f2300cab7028a1d0`. The host exited zero after
27.096 seconds, but the structured trace failed acceptance. Host exit status
and a plausible final answer cannot substitute for the required tool results.

The model issued `mnemos_remember` with the exact requested content, including
the final period. Its response acknowledged durable commit and reported a
pending job. All three observed model requests used Qwen3.5:9b, reasoning
disabled, and context size 65,536. The disposable title-generation change
therefore resolved the two setup problems in the
[first attempt](../hermes-native-lifecycle-local-v1/README.md).

The next model call supplied a different job ID from the one just returned.
The service responded with HTTP 500, exposed by the provider as a degraded
result. The evaluator rejected that unsuccessful result. The runner did not
perform post-teach maintenance or any later recall, correction, forgetting,
restart, or source-replay phase. Successful asserted-fact materialization and
the complete lifecycle remain unverified in this attempt.

This is an observed model argument-copy error and an unknown-job error-handling
problem. Do not rewrite the native arguments, accept the unrelated ID, silently
retry the phase, or report the original job as completed. A service error repair
would be separate from proving that the host checks the correct job.

[report.json](report.json), [failure.txt](failure.txt), and
[teach/trace.jsonl](teach/trace.jsonl) preserve the failure. The
[evidence index](evidence-index.json) binds 17 exact retained files. Live host
configuration is unchanged; credentials, runtime descriptors, and databases
are excluded. The three-host native driver remains unrun.
