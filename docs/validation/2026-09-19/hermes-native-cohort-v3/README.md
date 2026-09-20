# Native Hermes fixed cohort v3: three incomplete lifecycles

All three scheduled attempts ended at teach with a failed lifecycle result.
None was retried or replaced. Runner `899490a196983009aea3047061dc7230d9900267`
used installed runtime `68693f83b5894198daea114e018031e381a508a3`, wheel
`2e121c1ae81717d1b9a32349bd27ea7fafc2e5562e6eadbd7d24ddecd16f5cb8`, the pinned
Hermes cwd-only compatibility checkout, and the separate Qwen3.5:9b sampling
profile. Live host configuration and memory stores were unchanged.

| Attempt | Teach seconds | Model requests | Native tool calls | Lifecycle result |
| --- | ---: | ---: | --- | --- |
| 1 | 35.885 | 2 | `mnemos_remember` | Failed: no job-status call |
| 2 | 29.148 | 2 | `mnemos_remember` | Failed: no job-status call |
| 3 | 32.965 | 2 | `mnemos_remember` | Failed: no job-status call |

In each attempt the model saved the exact requested content, including its
final period. The write response reported committed storage and pending
processing. The model then answered from that response without issuing the
explicitly requested `mnemos_job_status` call. The unchanged evaluator rejected
the missing tool observation. Controller maintenance subsequently found one
asserted target in each database. This supports bounded write/materialization
evidence, not a completed native lifecycle.

All three attempts passed the recorded correlation, permitted-tool, scope,
provider, capture, control-preservation, exact-target, and sampling-profile
checks. Sampling evidence means a pinned server-side profile without a
conflicting request override; it does not inspect the sampler's internals.
Each native host process exited zero, and each lifecycle runner exited one.
The cohort verified terminal cleanup before starting the next scheduled attempt.

Restarted recall, correction, forgetting, and source replay were NOT RUN in
this cohort. The three-host driver remains NOT RUN because its Hermes
prerequisite is unmet. The v1 and v2 failures remain separate evidence. V3
changes both runtime and sampling settings, so these observations do not
isolate the effect of temperature, seed, penalty, or the HTTP error repair.

The harness preparation passed 74 focused tests, Black, and strict Python 3.10
mypy across 93 production/benchmark files. These checks do not add native
successes. No new package was built; the existing package verification remains
bound to `68693f8`.

`cohort.json` preserves all three outcomes. The evidence index binds 73 files,
including the executed helpers/protocol, phase traces and logs, selected
synthetic database rows, and sampling snapshots. Databases, runtime access
descriptors, credentials, and live user content are excluded.
