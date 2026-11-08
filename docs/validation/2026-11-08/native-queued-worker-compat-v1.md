# Native launcher compatibility with queued workers

The retained v3 experimental launcher matched SHA-256 `ec52c79a0ae613ae336836b3e82ddf853a601a0de3f7cd3aa0047318d06ee12b`. The current Rust source tree had no differences from its recorded environment-v3 commit `08a1d61e5cb1564abb06b4dc4efc1d49c4036050`. This is provenance and artifact verification, not a new reproducible-build claim.

An actual native Stop invocation reached the installed queued worker. A thin worker-side observer delegated to the unchanged installed dispatcher and retained its exact response before HTTP return. That response proved `dispatched=true`, queued count one, and the original source-event identity. Native stdout remained `{"continue":true}`. The test did not infer dispatch or capture success from that stdout alone.

The event persisted while memory delivery was unavailable. After forced termination of the owned worker, a different worker process replayed the event to the real memory service without another hook call. SQLite contained one pending job before explicit maintenance and one completed job/fact afterward. Both worker trees and ports were verified closed; memory-service cleanup and installed package/native binary identity checks passed.

The native command emitted its existing `degraded` diagnostic for the queued response. Queued capture is deliberately distinguished from synchronous service admission; that diagnostic does not negate the verified local receipt or establish completed processing. Native Codex integration still needs to evaluate this behavior in context.

Evidence is retained under `.venv/native-queued-worker-compat-v1`, including the helper, observer hash, owner proofs, exact worker receipt, native stdout/stderr, persisted-event checks, and cleanup. Deterministic embeddings and disabled experimental modules were used. No command latency, native Codex host dispatch, or real-model quality was measured. No live configuration or default changed.
