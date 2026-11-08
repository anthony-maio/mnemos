# Retained Qwen3.5 support screen failed

The single frozen configuration completed every request and failed semantic
quality and latency. Close it without a prompt, threshold or model sweep.

| Arm | Expected retained | Selected precision | Correct abstentions | Support-only p95 |
| --- | ---: | ---: | ---: | ---: |
| Vector | 27/27 | 27/41, 65.85% | 9/17, 52.94% | 2,615.85 ms |
| Lexical | 27/27 | 27/42, 64.29% | 7/17, 41.18% | 2,321.14 ms |
| Hybrid | 27/27 | 27/41, 65.85% | 9/17, 52.94% | 2,416.00 ms |
| Full cohort | 27/27 | 27/41, 65.85% | 9/17, 52.94% | 2,344.44 ms |

These are 44 known development questions across four retrieval arms, not 176
independent questions. Each arm has 27 answerable and 17 no-answer questions.
There were zero unavailable or unattempted operations. Full expected-fact
retention did not prevent unsupported selections. Critical retention remains
unmeasured because this replay's labels have no critical-fact registry. This
was neither fresh acceptance nor complete retrieval latency at 10,000 facts.

The saved Qwen3-14B contrast baseline retained the same 27 expected hybrid facts
with five unsupported selections and 12/17 correct abstentions. The new
configuration selected fourteen unsupported facts and abstained correctly on
nine. It does not improve that development result. Sampling controls and model
changed together; this is not an isolated architecture-effect estimate.

Execution was frozen at source `20f658c73a5b6655c715bb7f3650c49f2198aef6`,
with freeze SHA-256
`bf5b951a930c6e3c343f8ef26044bf106a9bda9a0f2d5e2267c7a24eebf0847b`.
The unchanged contrast-v5 requests, source excerpts, choices, reference dates
and 0.8 selection rule were replayed through the typed judge. Labels remained
outside inference. An unrelated warmup preceded the 176 requests. No old model
was rerun, no accepted store was reopened and no extraction occurred.

The retained model was Qwen3.5-9B Q4_K_M, digest
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
The runtime recorded all 5,862,210,928 model bytes resident in VRAM, context
8,192 and one native slot. Exact sampling controls, deadlines and admission
bounds are in the [frozen protocol](../../plans/2026-11-08-qwen35-support-screen.md).
All 176 prompts fit the 5,120-byte bound; the maximum was 4,861 bytes. Exact
outgoing request bodies and raw response bodies were recorded using direct
loopback transport. Thirty-six targeted replay/support tests passed before
execution, including label rejection and fail-closed evidence recording.

The enclosing owned job completed in 397.40 seconds, with zero processes before
and after its final cleanup. The model unloaded; server/native ports closed;
all 105 frozen file bindings, including model blobs, and retained manifests
remained unchanged. The runtime receipt reports no cleanup errors. Process
completion proves execution, not a passing semantic result.

Private evidence is `.venv/qwen35-support-screen-v1`. Its `evidence.zip` contains
112 files plus a manifest, with large model blobs retained separately by hash.
Archive SHA-256:
`68d10fbb6d9245b22a432507a44477a6f5b644779582a8a5d1f3bbd1d8e5c26c`.
No live memory, host configuration, optional pilot or published artifact changed.

The existing hosted TypeSafe/Jev backend has no configured key. The user has
been asked whether the next support path may use that backend or must remain
strictly local. Do not silently fall back to a hosted provider or rescue this
configuration while that direction is pending. The original quality, latency,
native workflow, pilot and delivery obligations remain open.
