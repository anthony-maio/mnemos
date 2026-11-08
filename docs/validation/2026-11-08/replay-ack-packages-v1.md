# Installed replay comparison packages

Two private 0.7.0b1 packages are prepared from exact canonical commits:

| Arm | Source | Wheel SHA256 |
| --- | --- | --- |
| Baseline | 7a77e40f4ec8dd5b2fbf47ff0b1e3f0862e0129d | 0eecf076ca4e3cbcd7f5330e099176e90650e69e0badf93dc44a3b0e8e11213d |
| Candidate | ec4a36b262be61c16d807a2b39006bcad7f8367c | ff7e97e1ed24a7122dd4094b9dcdd4ed2c5ece3e805ea3becca58e0e62502d5d |

Each wheel and sdist was built with Hatchling 1.33.0 from an archive of its
commit. Both pass Twine. Wheel RECORD entries and archive/source parity were
checked. Wheel contents differ only in mnemos/client.py and RECORD, isolating the
replay acknowledgement implementation. Each installed target contains 98
verified package/resource files.

Baseline sdist SHA256:
`57085c442fc99d51b0db05b71f059ed3f4b2210bff8b83bccb80d55de60b650f`.
Candidate sdist SHA256:
`183413aed656433b08eae45dc6ee9db21bcc1980b23e865f089120e18ecdbca6`.

## Installed checks

Both arms ran the same six selected replay, queued-capture, SQLite-worker,
registry-freshness and client-thread test files. Baseline: 35 passed. Candidate:
39 passed, including four additional scheduling/cancellation/failure tests.
There were no failures or skips. All 49 imported Mnemos modules came from the
intended installed target, and all 98 installed files remained byte-matched.
Dependency versions matched across the two checks. The pre-existing canonical
runtime package was verified unchanged before and after.

Private artifacts, source archives, build logs, installed targets, module origins,
dependency versions and JUnit are under `.venv/replay-ack-packages-v1`. The
preparation and check drivers are `.venv/prepare-replay-ack-packages-v1.py` and
`.venv/check-replay-ack-installed-v1.py`. No live install or optional pilot was
updated. These checks do not establish complete latency or native-host acceptance.

## Comparison still unrun

The [prospective protocol](../../plans/2026-11-08-replay-ack-comparison-v1.md)
requires timing both consecutive commands, alternating arm order, exact durable
event accounting and an explicit drain barrier. No injected delay is allowed.
Actual acknowledgement overlap must be reported; lack of overlap limits the
mechanism conclusion rather than authorizing a rerun.

The runner and auditor are not yet implemented or frozen. No measured comparison
ran, so the remaining comparison has not been consumed. The earlier attribution
failure, original 100 ms target, 10,000-fact recall requirement, semantic quality
gates and daily-use pilot obligations remain unchanged. The prepared synthetic
contention fixture cannot substitute for real embedding-workload acceptance.
