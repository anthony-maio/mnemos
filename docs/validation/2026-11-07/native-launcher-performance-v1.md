# Native comparison fails the resident-route requirement

All 104 native hook commands, including four warmups, failed the requirement
to dispatch through the resident worker. They returned valid output and their
captures survived restart, but no candidate worker dispatch was recorded. These
measurements do not establish native-path performance. The original failures,
raw observations and thresholds are retained.

Source `c83b0d9e27b50fac7d0a71cfe32a1408a7819dc8` adds a hash-pinned native
profile and guarded fallback. Both arms use the unchanged installed product
`3d1866d5`, Python 3.12.11, 96 package files, 57 dependencies, 17 local BGE
assets and the original 10,000-fact seed. The
[prospective protocol](../../plans/2026-11-07-native-launcher-performance-v1.md)
keeps scope resolution, settings checks, capture, maintenance, real embeddings
and full context intact. No native-host session or live profile was involved.

## Recorded results

Each path has 50 timed observations per arm. The p95 uses the original
interpolated convention. Candidate hook times include the failed resident
route and subsequent guarded adapter execution; they must not be presented as
successful native-worker latency.

| Path | Python median / p95 ms | Candidate command median / p95 ms | Failed candidate observations | Gate |
| --- | ---: | ---: | ---: | --- |
| Recall | 110.38 / 118.73 | 109.67 / 114.54 | 0/50 | Both pass 500 ms |
| Stop | 290.27 / 469.95 | 934.67 / 1403.90 | 50/50 | Both fail 100 ms |
| UserPromptSubmit | 427.97 / 648.78 | 1131.82 / 1644.00 | 50/50 | Both fail 100 ms |

Recall uses the unchanged service path. Its small timing difference does not
establish a native launcher improvement. All 160 decoded response/context pairs,
including warmups, match exactly. There are 320 observations, 300 timed, and
208 durable captures. Every capture survives service restart and every associated
job completes. Accepted facts and their retained evidence/vectors remain unchanged in
all four stores; captures add their own event and job records. Integrity and
foreign-key checks pass.

Both baseline blocks record 52 handled, nondegraded worker dispatches with one
capture each. Neither candidate block records a worker dispatch. All candidate
commands exit zero without stderr or a harness exception. The frozen success
criterion correctly rejects them despite valid output and durable capture.

## Reproduced environment defect

A separate, model-free observer compares Python and native effective environment
fingerprints under the same reviewed profile. With implicit subprocess
inheritance, the native process sees Windows' hidden `=C:` drive-directory
entry, which Python omits from `os.environ`. With an explicit Python environment
dictionary, the fingerprints match. The earlier loopback, profile and policy
tests used explicit dictionaries and missed this case.

A synthetic `=Q:` regression then fails in all four combinations of host and
launcher mode. Two ordinary-environment mismatch controls pass. This establishes
a defect that can cause the observed pre-dispatch decline. The reconstructed
Python fingerprint differs from the original run's recorded fingerprint, and
the original native commands did not log their decline reason. The diagnostic
therefore does not establish that the hidden entry was the only difference in
the original run. It records hashes only, without environment values, model
calls, runtime startup or database access.

## Evidence and next check

The guarded profile has 19 new checks within 131 passing subprocess/related
checks. Two Rust deadline checks, strict Clippy, formatting and 12 documentation
checks pass on the original candidate. The exact private guard agrees across
12 paired cases, including valid settings, inherited selectors, prohibited
global/project LLM configuration and malformed configuration. These checks
establish preparation behavior; the full comparison still fails its route gate.

The first audit assumed every block had a dispatch file and stopped at the
missing candidate file. That auditor and failure are retained. The revised audit
verifies the failed route against the original observations, recomputes summaries
and pairing, checks all 208 captures and completed jobs, and verifies source,
artifact, model, seed and process cleanup evidence. It does not regrade failures
as successes. Archive details are in the
[receipt](native-launcher-performance-v1-receipt.json).

Correct the Windows environment conversion, preserve the failing regression,
then freeze a separate complete comparison. Keep the same seed, complete timers,
worker receipts, output equality and latency thresholds. Production defaults and
live configuration remain unchanged. The private pilot is inactive at zero days;
there is no native-host acceptance, release or publication claim.
