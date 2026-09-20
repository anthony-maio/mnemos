# Native Hermes lifecycle v8: separate captured evidence from ordinary recall

This prospective successor tests the verified retrieval-view package in the
actual Hermes host with automatic capture enabled. The [v7 cohort](../validation/2026-09-20/hermes-native-cohort-v7/README.md)
remains zero of three complete lifecycles, despite successful readiness.
It omitted correction, inspection/deletion, or the required job-status call.
No earlier outcome is rewritten.

Use runtime `e419c2aa615e037e639a74b14615889d03f4e47f`, wheel SHA-256
`185f3df7720a32e5ed285d9c507e1e45e7f57370942740b53c4f8178bb97a97e`.
The [checkpoint](../validation/2026-09-20/retrieval-views-checkpoint-v1/README.md)
passes 1,500 source tests, 648 installed contract cases, four matching skips
in each suite, all six source/package checks, and seven replay assertions.

The package separates ordinary memory from whole captured conversations.
The same retrieve tool now has an optional evidence view, with separate
response channels and corresponding guidance. Automatic recall excludes
passthrough capture before candidate limits, while inspection and related
forget still expose stored copies. This does not delete those copies or
establish that old context caused v7's model omissions.

## Frozen workflow and criteria

Use `hermes_lifecycle_run_v8.py`, `hermes_cohort_run_v6.py`, unchanged
`hermes_native_bootstrap_v4.py`, unchanged `ollama_readiness_v1.py`, and the
unchanged `hermes_trace_v2.py` evaluator. Commit the runners and this protocol
before inference. Bind them, the earlier v3-v7 protocols, and the model
profile by hash.

Retain every v7 prompt, fixture value, scope, control, eight tool names,
capture setting, restart, source replay, deadline, and evaluator gate.
The model must retrieve and inspect the target, review its forget preview,
inspect selected additional facts, and explicitly select required sources.
The controller supplies no fact IDs, repairs no arguments, and performs no
model-requested deletion itself. Preserve the final-model-request substring
check and the scoped stored-label scan across facts, events, evidence,
revisions, and jobs. Excluding a captured copy from recall cannot satisfy
erasure while any required stored-copy gate still fails. No threshold or
success criterion changes.

Retain local Ollama at `http://127.0.0.1:11434`, the Qwen3.5:9b greedy-v1
alias and its existing digest, Nomic embedding digest, 65,536-token context,
no reasoning, temperature zero, seed 42, and presence penalty zero. Verify
digests and sampling profiles before the run and each phase. All five
cognition modules remain disabled for this host-lifecycle baseline; their
quality experiments remain separate.

Keep the pinned Hermes checkout and isolated cwd-only compatibility patch.
Use a disposable home, project, registry, service, and database. Do not alter
live host configuration, source, credentials, or memory. The import-only
preflight must make no readiness or model call. Readiness retains the v7
single tool-free chat, 180-second timeout, 16-token bound, structural checks,
digest checks, and positive reported GPU residency, without retries.

## Fixed schedule and interpretation

Run exactly three independent fresh attempts. Retain eight host turns,
180 seconds per host model run, the 240-second outer host deadline,
five-second kill grace, and the 1,800-second total attempt deadline.
Stop dependent phases after failure and start the next scheduled attempt
only after established cleanup. Do not repeat selected phases, add attempts,
change inputs during the cohort, or omit startup failures from the denominator.

Report every outcome, unreached phase, readiness result, cleanup result,
and runtime/model/input binding. This tests the newer package as a whole,
including its tool schema and guidance; it is not a controlled estimate of
the retrieval filter's causal effect. A pass establishes only that bounded
attempt. Three-host continuity requires its separate stronger test after a
bound Hermes prerequisite passes. Complete capture, real-model quality,
Desktop, and the pilot remain separate. Update the research handoff after
the cohort reaches a terminal state.
