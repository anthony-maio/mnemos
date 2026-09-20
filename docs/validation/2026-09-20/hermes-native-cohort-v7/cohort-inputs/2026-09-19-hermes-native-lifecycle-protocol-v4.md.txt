# Native Hermes lifecycle v4: explicit write-completion guidance

This is a prospective successor to the failed fixed v3 cohort, not a retry of
its observations. V3 remains zero completed lifecycles out of three attempts.
Every v3 model request exposed the status tool, but each host saved the fact
and ended without making the required status call.

The implementation change places the asynchronous-write contract directly in
the remember/correct tool descriptions as well as the provider system block.
When a committed response includes a job ID, the agent must check that exact
ID once before its final answer and report the observed state. Pending or
leased work is not completed. Suppressed writes without a job ID do not require
a fabricated status lookup. No controller call substitutes for native tool use.

Use runtime source `8f8f63f3250858667dfe460bdd22295c279f2625` only after its full
source/package checks and isolated installed checks pass. The wheel digest is
`53444f7f87141e7107fe9f4daf4a6095e04c9d69d007cd218634fd2fff50f99f`. The package
also contains the independent affective numeric-parser repair; modules remain
disabled for this baseline native lifecycle, so this probe does not evaluate
affective fidelity or module utility.

All phases, synthetic content, exact identity/version rules, scope/capture and
replay requirements, model profile, context size, turn/time budgets, and
failure behavior follow the [v3 protocol](2026-09-19-hermes-native-lifecycle-protocol-v3.md)
at `899490a196983009aea3047061dc7230d9900267`. The parent protocol SHA-256 is
`18b8a01ba62510f714e6e3db6aa24707b72d8028f96ae68dabb2e939a40b36b5`.
Its v1/v2 historical setup notes remain historical, not v4 instructions.
The v4 source/wheel binding supersedes v3's runtime binding only for this run.

Use `hermes_lifecycle_run_v4.py`, `hermes_native_bootstrap_v4.py`, and the
unchanged `hermes_trace.py` evaluator. Bootstrap additionally records the
Mnemos tool descriptions in every native model request. Require them to match
the schemas in the verified installed package. This verifies supplied guidance;
the original native tool-call and result gates still determine whether the
host followed it. Do not alter native arguments or insert a status call.

Run exactly three independent fresh attempts through `hermes_cohort_run_v2.py`.
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
prerequisite passes. The writer packet must retain both cohorts' outcomes.
