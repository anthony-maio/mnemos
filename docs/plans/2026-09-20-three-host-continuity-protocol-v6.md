# Three-host native continuity v6: bind the repaired Hermes prerequisite

Status: prospective; NOT RUN. The completed Hermes v11 cohort has zero full
passes, so its prerequisite is not satisfied. Do not execute this protocol
on the basis of that cohort. The Hermes v11 prerequisite must first finish
all three fresh attempts with cleanup established and at least one full pass.
Retain every attempt's outcome. Commit this protocol and
all runner inputs before a separate import preflight and one fresh native
attempt.

The [v5 attempt](../validation/2026-09-20/three-host-native-v5/README.md)
passed Codex and Claude teaching, then stopped during Hermes recall when the
generated project guidance supplied unsupported scope arguments. Query-only
retries recovered the facts, but the original rejected calls remain a failure.
Correction and later phases did not run. Preserve that result.

## Confirmed runner mismatch

The custom dispatcher factory in v4 bypasses `MnemosService`'s default-owner
maintenance startup. The correction phase has no consolidation tool, yet its
contract requires a native job-status response showing completion. Explicit
controller maintenance only runs after the native phase. With no concurrent
worker, that completion condition cannot be met through the allowed tools.

The offline regression first failed with a persisted correction remaining
pending under repeated status reads. The repair uses a benchmark subclass
of `MnemosService` to construct the same isolated dispatcher through the
default-owner path, retaining the installed scheduler start, stop, and task
cancellation behavior. The service remains confined to the disposable root.
Archive its `service.log` with the phase records. The subclass relies on a
private factory method in the exact bound installed version; it does not
claim compatibility with arbitrary later service versions.

The benchmark scheduler checks every 0.25 seconds with zero idle delay.
Record those settings and `scheduler_running` health on every service start.
Reject startup if the scheduler is not running. This accelerated schedule
is for the bounded lifecycle experiment; it provides no evidence about
production idle behavior, processing latency, or end-user performance.
Durable job claiming, write validation, batch bounds, retry rules, and
correction semantics remain the installed implementation.

## Unchanged criteria

Keep all v5 native prompts, tool allowlists, call correlation, inspection,
exact-version correction, completed-job observation, scope controls, reviewed
forgetting, source replay, and final recall requirements. In particular,
pending receipts cannot satisfy completed-job checks. Do not suppress a
rejected tool call, repair model arguments, or inject a controller tool result.
The repaired source package changes generated guidance only; it does not alter
the eight tool schemas, the scope firewall, or native acceptance criteria. A
fresh attempt must satisfy every requirement independently.

Keep [v5's model and controller setup](2026-09-20-three-host-continuity-protocol-v5.md),
native phase order, budgets, deadlines, real Nomic embeddings, local Ollama
Qwen profile, and all five cognition modules disabled. Bind product source
`697b6720b201d578c03ce33d98dff3dda6d2def8` and wheel SHA-256
`49c7d791cd6b5d7645125cec72371b453607543b371e6834c6a2e122774f42b0`.
No live host settings, data, or service are modified.

Use profile `native-three-host-v6`, fresh outputs
`mnemos-three-host-preflight-v6` and `mnemos-three-host-native-v6`, one
no-inference import preflight, and exactly one native attempt. Bind v1-v6
protocols and the existing catalog diagnostic. Require the completed, cleaned Hermes v11 cohort at
`C:/Users/Ant/AppData/Local/Temp/mnemos-hermes-native-cohort-v11` and its
exact v11 runner inputs before starting. Stop at the first failure;
no selective retries. Success remains `bounded_lifecycle_observed`, with
broader readiness and real-model quality gates separate. The v5 failure remains
archived; this run is a new package-bound attempt, not a causal comparison
against v5 or v10.


## Required binding diff before execution

The existing three-host runner and runtime still point at v5 and the prior
package. The minimal successor binding updates the runtime source and wheel
constants, profile to `native-three-host-v6`, and protocol path to this file.
The runtime Hermes input list must point to `hermes_lifecycle_run_v11.py` and
the v11 protocol, with its list matching the v11 report's recorded input
hashes. The runner input list must add the v5 protocol while retaining v1-v4;
its existing `PROTOCOL` entry supplies v6. The acceptance report argument
must point to the passed v11 cohort attempt report, and the cohort directory
must have exactly three attempts, a passing report, and cleanup evidence.

The worker profile remains 0.25 seconds with zero idle delay. No prompt, tool
allowlist, phase order, model, sampling, deadline, or acceptance gate changes
are part of this binding.
