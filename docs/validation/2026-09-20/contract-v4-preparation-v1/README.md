# V4 contract repair and real-model run preparation

September 20, 2026. Source `3d0965805622efad21d1bb5200a982bd1a956aa8`.
Preparation passed and the new seven-profile matrix started at 13:36:53 UTC.
The [start observation](run-start-observation.json) verifies a live process
and 105 complete baseline observations at 13:39:08 UTC. Execution and quality
acceptance are still open; this is a startup checkpoint, not a terminal result.

## What changed

The [prospective protocol](../../../plans/2026-09-20-contract-v4-evaluation-protocol-v1.md)
and [frozen input](../../../../benchmarks/local_memory/datasets/local-memory-heldout-v4.jsonl)
resolve the documented fixture conflicts without changing the scoring code:

- Forty supported explicit-memory gold bodies now match the exact existing
  API body. Sixty unscored annotations preserve the old text and API body.
- Twenty state-pair targets follow those supported bodies, with their query
  text and state metadata unchanged.
- Twenty foreign queries remain foreign but are labeled no-answer. Twenty
  separately identified local queries preserve the 100-answerable minimum.
- All source events, lifecycle targets, replacement bodies, and historical
  stale-detection text remain unchanged. So do thresholds and scoring.

The dataset has 200 heldout-labeled sessions, 20 development sessions,
100 critical facts, 100 answerable queries, 120 no-answer queries, and 50
state pairs. It is a repair of previously observed synthetic data. It must
not be presented as unseen heldout evidence or used to turn old results
into passes. The [original audit](../retrieval-fixture-audit-v1/README.md)
and all v3 negative results remain intact.

## Verification and limits

The development full suite passed 1,764 tests with four skips. It began
before five final canary-guard tests were added. The final frozen checkout
separately passes all 29 current v3/v4 contract tests, including those
guards. Black passes across 323 files and mypy across 142 source files in
that checkout; strict Python 3.10 checking also passes for the three
changed code/test files. See [verification.json](verification.json) and
the indexed test records.

The canary regression record preserves five failures before the guard was
added. The guarded loader rejects missing foreign identities, localized
foreign identities, and an altered local-pair identity. V3's original
correction and forget target validation still runs for v4.

The product tree is identical to verified package `e6887de`; this checkpoint
does not claim a new installed-package or native-host acceptance result.
Adapter, harness, scoring, and retrieval-comparator Git blobs are unchanged.
The dataset hash is
`0c1f74e24d485a37889a18a65463b85c4486824975de8283dda50754d6e434aa`.

The unchanged adapter includes supplemental controlled-state probes inside
the session timer. Even the modules-off baseline calls the state classifier
for these probes, after computing the primary retrieval metrics. Its
`latency_ms` includes that diagnostic work; it is not ordinary memory-lookup
latency. Likewise, `cognition_observed` and `adapter_kind=real_model` can
reflect those probe calls and do not prove a production module was enabled.
Use the recorded module flags and separate primary/diagnostic fields when
interpreting the run. See `evaluate_session()` and
`_controlled_state_pairs()` in the frozen `ollama_adapter.py`.

## Run binding and observation

[preparation.json](preparation.json) was written before inference and binds
the committed source, complete code provenance, dataset, protocol, runner,
and pinned Qwen/Nomic models. Its `NOT RUN` value records that preparation
moment. [metadata.json](metadata.json) contains the live discovery records
and RTX 4080 hardware output. The later start observation records both
models resident in VRAM; it does not prove residency for every request.
The archived `protocol.md` preserves the frozen bytes for verification;
use the prospective-protocol link above to follow its relative citations.

The isolated source is
`C:/Users/Ant/AppData/Local/Temp/mnemos-source-contract-v4-v1`.
The active status, reports, checkpoints, and attempt databases are under
`C:/Users/Ant/AppData/Local/Temp/mnemos-contract-v4-matrix-v1`.
The recorded process is PID 67836 and the execution tool session is 9892.
Verify that specific process/session before deciding whether work stopped.
Do not launch a replacement merely because a polling call times out.

There are seven profiles and 600 planned observations per profile. Preserve
all partial outcomes and missing-target failures. Complete execution alone
will not establish quality, module utility, native reliability, or the pilot.
The current runner deliberately refuses to overwrite existing execution
directories; any interruption recovery needs the same frozen bindings and
the existing checkpoint contract.
