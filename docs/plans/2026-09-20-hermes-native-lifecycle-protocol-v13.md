# Native Hermes lifecycle v13: review-rejection package

This prospective fixed cohort tests the verified review-rejection package.
The preceding v12 cohort finished zero of three complete lifecycles. One
attempt received a valid invalid-selection rejection, then submitted an
incomplete preview twice and received HTTP 500. It ultimately committed
target-only deletion without proving reviewed-copy erasure. An independent
regression also found that invalid-selection rejection could replay valid
pending captures. Preserve those observations and their separate provenance.

## Frozen inputs

Use source `db217509477d92b9a82bd7e11d4a4bffb542fa29` and wheel SHA-256
`8b1f560e98b93376fc2aa073be7c1500997d888d09ae87573d5d70ec8694ad00`.
Require the passing source/package and isolated-installed reports, verify
installed bytes, and use that installed environment outside the repository.
Commit the v13 runner, v11 cohort wrapper, and this protocol before preflight.

Keep the bootstrap, evaluator, request observer, readiness helper, models,
model digests, sampling, thinking setting, host source and compatibility patch,
live interpreter revision, and dependency inventory from
[v12](2026-09-20-hermes-native-lifecycle-protocol-v12.md). Use local Ollama at
`http://127.0.0.1:11434`; do not use the failed remote machine. Require the
pinned Qwen alias, 65,536-token context, and positive finite GPU residency.
Preserve exact provider requests and loaded-model observations after responses.

## Execution and acceptance

First perform import-only host discovery with no inference or service writes.
Then run exactly three fresh attempts. Keep every lifecycle prompt, fixture,
tool schema, capture rule, scope control, restart, source replay, stored-copy
erasure requirement, and evaluator from v12. All five cognition modules stay
disabled for this transport baseline. Do not repair model arguments or perform
missing tool calls on its behalf.

Keep eight host turns, a 180-second model budget, 240-second outer deadline,
five-second kill grace, and 1,800-second attempt deadline. Stop dependent phases
at the first failure; stop the cohort if cleanup is uncertain. Preserve every
scheduled outcome. Do not add attempts or selectively repeat failed phases.

Record whether any typed rejection occurs and what the model does next.
A correct rejection is not successful deletion. The new package makes known
review failures noncommitting and prevents rejected requests from draining
pending captures; it does not establish model recovery or a complete lifecycle.
Report observed full lifecycle outcomes independently of package checks.
Fresh identifiers and conversations vary, so this is not a causal comparison
or a general reliability estimate.

Live host configuration, source, and memory stay unchanged. Three-host
continuity requires a separately bound prospective protocol; its earlier v6
prerequisite remains unsatisfied. Full capture, Desktop, module quality, and
the operational pilot remain separate gates. Preserve all earlier failures.
