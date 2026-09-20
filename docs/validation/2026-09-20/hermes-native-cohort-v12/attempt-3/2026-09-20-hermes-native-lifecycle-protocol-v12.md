# Native Hermes lifecycle v12: selection-rejection package

This prospective fixed cohort tests the verified selection-rejection package.
The earlier v11 cohort remains zero of three complete lifecycles. Its invalid
source selection returned HTTP 500; this package returns the fixed,
noncommitting `related_forget_selection_invalid` rejection. The current run
tests whether a complete native lifecycle succeeds on this package, without
assuming the model will encounter or recover from that rejection.

## Frozen inputs

Use source `5155ebddd8e8cf42df1c67f11a1578f5edfb43ec` and wheel SHA-256
`3828edfeab80f11a94a3b5519793af0b8b228b4bb5bef9ced34608186d6468e7`.
Require the existing passing source/package and isolated-installed reports,
verify installed bytes, and use that installed environment outside the repo.
Commit the v12 runner, v10 cohort wrapper, and this protocol before preflight.

Keep the bootstrap, evaluator, request observer, readiness helper, models,
model digests, sampling, thinking setting, host source and compatibility patch,
live interpreter revision, and dependency inventory from
[v11](2026-09-20-hermes-native-lifecycle-protocol-v11.md). Use local Ollama at
`http://127.0.0.1:11434`; do not use the failed remote machine. Require the
pinned Qwen alias, 65,536-token context, and positive finite GPU residency.
Preserve exact provider requests and loaded-model observations after responses.

## Execution and acceptance

First perform import-only host discovery with no inference or service writes.
Then run exactly three fresh attempts. Keep every lifecycle prompt, fixture,
tool schema, capture rule, scope control, restart, source replay, stored-copy
erasure requirement, and evaluator from v11. All five cognition modules stay
disabled for this transport baseline. Do not repair model arguments or perform
missing tool calls on its behalf.

Keep eight host turns, a 180-second model budget, 240-second outer deadline,
five-second kill grace, and 1,800-second attempt deadline. Stop dependent phases
at the first failure; stop the cohort if cleanup is uncertain. Preserve every
scheduled outcome. Do not add attempts or selectively repeat failed phases.

Record whether an invalid-selection rejection occurs and what the model does
next. A correct rejection is not successful deletion. Report full lifecycle
outcomes independently of package tests. This is not a causal comparison or a
general reliability estimate; fresh identifiers and conversations vary.

Live host configuration, source, and memory stay unchanged. Three-host
continuity requires a separately bound prospective protocol; its earlier v6
prerequisite remains unsatisfied. Full capture, Desktop, module quality, and
the operational pilot remain separate gates. Preserve all earlier failures.
