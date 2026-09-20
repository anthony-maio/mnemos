# Three-host native continuity v5: run the correction worker

Status: prospective; NOT RUN. Commit this protocol and all runner inputs
before a separate import preflight and one fresh native attempt.

The [v4 attempt](../validation/2026-09-20/three-host-native-v4/README.md)
passed teaching through Codex and Claude and restarted recall through Hermes.
Correction failed on a rejected scope argument, missing inspection, and a
pending correction job. Preserve those failures. No later phase ran.

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

Keep all v4 native prompts, tool allowlists, call correlation, inspection,
exact-version correction, completed-job observation, scope controls, reviewed
forgetting, source replay, and final recall requirements. In particular,
pending receipts cannot satisfy completed-job checks. Do not suppress a
rejected tool call, repair model arguments, or inject a controller tool result.
The worker repair does not resolve v4's incorrect argument or omitted
inspection. A fresh attempt must satisfy those requirements independently.

Keep [v4's model and controller setup](2026-09-20-three-host-continuity-protocol-v4.md),
native phase order, budgets, deadlines, real Nomic embeddings, local Ollama
Qwen profile, and all five cognition modules disabled. Product source
`f2f741443e60f838b392ae8ea1d1a0b4d8c6c1f2` and wheel SHA-256
`8552345d2c17c4c037347961eb3043d13b20f790f54a9ccc86a096add6197b0f`
remain unchanged. No live host settings, data, or service are modified.

Use profile `native-three-host-v5`, fresh outputs
`mnemos-three-host-preflight-v5` and `mnemos-three-host-native-v5`, one
no-inference import preflight, and exactly one native attempt. Bind v1-v5
protocols and the existing catalog diagnostic. Stop at the first failure;
no selective retries. Success remains `bounded_lifecycle_observed`, with
broader readiness and real-model quality gates separate.
