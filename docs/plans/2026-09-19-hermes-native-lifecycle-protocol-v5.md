# Native Hermes lifecycle v5: bridge and forgotten-fact repairs

This is a prospective successor to the failed fixed v4 cohort. Both v3 and v4
remain zero completed lifecycles out of three attempts each. V4 attempt 1
retrieved a forgotten value from newly captured assistant narration; attempt 2
failed on non-UTF-8 bridge output; attempt 3 announced a correction without
calling the tool. None of those outcomes is replaced by this successor.

The new runtime writes explicit UTF-8 bridge responses, reports absent or
inaccessible fact targets as a typed 404, and suppresses automatic assistant
captures quoting exact forgotten fact IDs. The last rule is conservative at
the whole-event boundary and does not establish arbitrary paraphrase erasure.
It leaves user captures and explicit writes eligible. The implementation does
not force the model to make a correction call.

Preserve the v4 asynchronous-write guidance in tool descriptions and the system block.
When a committed response includes a job ID, the agent must check that exact
ID once before its final answer and report the observed state. Pending or
leased work is not completed. Suppressed writes without a job ID do not require
a fabricated status lookup. No controller call substitutes for native tool use.

Use runtime source `fd83fb27042c05897f29dba952b0924b15889aad` only after its full
source/package checks and isolated installed checks pass. The wheel digest is
`4958c5ef7a509fefecacc0666d87f578b46e94e1a33a896831468c0f971bf895`. The package
also contains the independent affective numeric-parser repair; modules remain
disabled for this baseline native lifecycle, so this probe does not evaluate
affective fidelity or module utility.

All phases, synthetic content, exact identity/version rules, scope/capture and
replay requirements, model profile, context size, turn/time budgets, and
failure behavior follow the [v3 protocol](2026-09-19-hermes-native-lifecycle-protocol-v3.md)
at `899490a196983009aea3047061dc7230d9900267`. The parent protocol SHA-256 is
`18b8a01ba62510f714e6e3db6aa24707b72d8028f96ae68dabb2e939a40b36b5`.
Its v1/v2 historical setup notes remain historical, not v5 instructions.
The v5 source/wheel binding supersedes v4's runtime binding only for this run.

Use `hermes_lifecycle_run_v5.py`, the unchanged `hermes_native_bootstrap_v4.py`, and the
unchanged `hermes_trace.py` evaluator. Bootstrap additionally records the
Mnemos tool descriptions in every native model request. Require them to match
the schemas in the verified installed package. This verifies supplied guidance;
the original native tool-call and result gates still determine whether the
host followed it. Do not alter native arguments or insert a status call.

Run exactly three independent fresh attempts through `hermes_cohort_run_v3.py`.
Each starts with an empty home, database, and occurrence registry. Stop the
dependent phases after a failure. Continue to the next scheduled fresh attempt
only after established cleanup. Retain every outcome and do not add attempts
or selectively repeat a phase. The 1,800-second total attempt deadline and
uncertain-cleanup stop behavior are unchanged. A successful attempt establishes
only its bounded lifecycle; all three outcomes must accompany it.

Keep live host source, configuration, memory stores, and credentials unchanged.
The pinned isolated Hermes checkout retains only its recorded cwd propagation
patch. The earlier cohort's model profile and negative results remain intact.
The three-host workflow is separate and remains unrun until its bound Hermes
prerequisite passes. The writer packet must retain every cohort's outcomes.

The [v4 parent protocol](2026-09-19-hermes-native-lifecycle-protocol-v4.md) is
also a committed input. Preserve its supplied-guidance check and every native
tool gate. This successor changes the installed source and wheel binding;
it keeps the model profile, prompts, phases, budgets, and exact three-attempt
schedule unchanged. No controller action substitutes for a native call.

