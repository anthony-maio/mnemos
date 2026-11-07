# Native routing repaired; complete hook latency still exceeds the gate

The corrected native launcher reaches the resident worker for all 104 hooks,
including warmups. Complete Stop median time falls from 289.46 to 132.30 ms,
and prompt submission from 427.90 to 255.26 ms. All output/context pairs match
and all 208 captures survive restart with completed jobs. Both complete-hook
p95 values still exceed the original 100 ms requirement. The latency gate
remains failed.

## Candidate and retained failure

Source `08a1d61e5cb1564abb06b4dc4efc1d49c4036050` omits Windows hidden
drive-directory entries, such as `=C:`, when constructing the native environment.
Python omits these entries from `os.environ`, while an implicitly inherited
native environment retains them. Ordinary environment mismatches still decline
dispatch. Other platforms retain their prior behavior; no other-platform runtime
acceptance is claimed.

The original executable fails four inherited-environment regressions covering
both hosts and both command modes. Two ordinary-mismatch controls pass before
the fix. The corrected executable passes all six within 137 subprocess/related
checks, plus two Rust deadline tests, strict Clippy, Rustfmt, Black and 12
documentation checks. Twelve exact private guard pairs, or 24 commands, agree
without starting a model or service.

The [failed v1 comparison](native-launcher-performance-v1.md) and its initial
audit failure are preserved separately. Its 104 failed native routes remain
failures. This run uses the separately frozen
[v2 protocol](../../plans/2026-11-07-native-launcher-performance-v2.md), output
directory and executable. No v1 timing or success flag is replaced.

Production Python files remain `3d1866d5`. Both arms use the same installed
Python 3.12.11 environment, 96 package files, 57 dependencies, 17 local BGE
assets and original pristine 10,000-fact seed. Wheel SHA-256 is
`253ca806dba5aa817bd7ddcba1af123535b29853aa211fee61a14f71799a62b8`;
native executable SHA-256 is
`ec52c79a0ae613ae336836b3e82ddf853a601a0de3f7cd3aa0047318d06ee12b`.
Implicit environment inheritance is preserved. Scope resolution, current
settings and membership checks, durable capture, maintenance, real query
embeddings and full context output remain inside the measured paths.

## Complete measurements

The timer includes command startup, environment preparation, transport, worker
checks, capture/service work, output collection and process exit. There are
50 timed observations per path and arm, with no failed observations, omitted
commands or retries. The p95 uses the original interpolated convention.

| Path and arm | Median ms | p95 ms | Maximum ms | Gate |
| --- | ---: | ---: | ---: | --- |
| Recall, Python baseline | 108.82 | 126.78 | 129.76 | PASS, 500 ms |
| Recall, candidate block | 110.00 | 125.16 | 142.65 | PASS, 500 ms |
| Stop, Python baseline | 289.46 | 506.12 | 978.80 | FAIL, 100 ms |
| Stop, native | 132.30 | 252.32 | 838.43 | FAIL, 100 ms |
| UserPromptSubmit, Python baseline | 427.90 | 596.42 | 1081.96 | FAIL, 100 ms |
| UserPromptSubmit, native | 255.26 | 373.46 | 664.85 | FAIL, 100 ms |

Native is faster in 49/50 matched Stop pairs and 48/50 prompt pairs. Median
paired changes are -152.20 and -167.83 ms, respectively. Recall uses the
unchanged service path: its median paired change is +0.83 ms, with the candidate
block faster in 21/50 pairs. It does not establish a native retrieval improvement.

Block results preserve the original baseline/candidate/candidate/baseline order:

| Block | Stop median / p95 ms | Prompt median / p95 ms |
| --- | ---: | ---: |
| baseline-0 | 285.68 / 486.38 | 420.95 / 614.72 |
| candidate-0 | 131.76 / 254.96 | 252.17 / 373.03 |
| candidate-1 | 132.93 / 245.93 | 257.33 / 373.18 |
| baseline-1 | 295.46 / 483.66 | 429.73 / 531.23 |

The comparison takes 474.86 seconds including preparation, restarts and cleanup.
It retains 320 observations, 300 timed, and 160 exact decoded output/context
pairs including warmups. Every hook has one handled, nondegraded worker receipt,
bound to its payload, output, environment, event and host. None uses fallback.
All 208 captures survive service restart and all 208 associated jobs complete.
The accepted facts and their retained evidence/vectors remain unchanged in all
four stores; captures add their own event and job records. Database integrity
and foreign-key checks pass.

## Audit and remaining work

The independent audit recomputes ordering, every summary, paired differences,
route acceptance, capture identity and restart durability from retained records.
It checks source/artifact/model bindings, unchanged accepted projections and
owned-process cleanup without model calls. The private archive includes raw
observations, worker receipts, databases, drivers and their preparation diff,
the candidate executable, checks and audit. It references the preserved v1
failure archive and original seed archive by hash. Details are in the
[receipt](native-launcher-performance-v2-receipt.json).

All recorded process identities have exited. One numeric PID was reused by an
unrelated process created after the run finished; it was left untouched.

This establishes a faster complete path in the synthetic installed comparison,
with the required behavior preserved. It does not meet the complete-hook latency
gate. Further diagnosis needs to separate remaining worker, project-resolution,
capture and retrieval costs while preserving the current checks and complete
timers. Do not infer a component bottleneck from this end-to-end measurement.

The native executable remains a development experiment outside the package and
host configuration. There is no native-host acceptance of this executable,
live activation, release or publication. The private pilot remains inactive
with zero actual days. Existing semantic quality and pilot entry requirements
remain open.
