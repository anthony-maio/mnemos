# Reconciliation decoder development v4

Status: prospective. Commit source, data and this protocol before inference.

The [reasoning comparison](../validation/2026-11-07/reasoning-v3.md) produced
wrong revisions whose final explanations acknowledged a mismatched attribute
or environment. This experiment tests whether decoder constraints contribute
to that behavior. The explanation alone does not establish the cause.

Use the unchanged 20-case preclassified development corpus at
`benchmarks/local_memory/fixtures/reconciliation-development-v1.json`, SHA-256
`366ae35e01f517b88a0826dc2f109d51775065d552fcd0eb992033196776fef0`.
Run four arms once per case, 80 observations, shuffled with seed 2026110701:

| Arm | Outbound format |
| --- | --- |
| bounded | Existing complete decision schema |
| reversed | Same schema, decision oneOf branches reversed |
| json | JSON object mode without the bounded schema |
| none | No format argument |

All arms use identical direct/full prompts, events, preclassified candidates,
target IDs and versions. Keep the existing duplicate-identity, source role,
scope, provenance and current-version runtime guards. The benchmark adapter
changes only the outbound format argument. It neither repairs raw responses
nor replaces the runtime validator. The shared prompt still refers to a
supplied schema; preserve this wording for the controlled comparison and
acknowledge that the two schema-free arms do not receive that schema.

Before inference, verify exact prompt and input equality with v3. Prove that
reversal only changes the order of the complete decision branches, preserving
all allowed decisions and forbidden combinations. Check every expected
alternative under both bounded schemas. Malformed text, runtime rejection,
timeout and interruption remain failures with no abstention credit.

Use local Qwen3:14b Q4_K_M, digest
`bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`,
Ollama 0.40.9, generate route, thinking false, temperature zero, context 8192
and one parallel request. Every arm gets 4096 output tokens, a 3584 native
input-token cap, a 512-token template reserve, a 90-second provider deadline
and a 120-second observation deadline. Count the actual prompt and outbound
format before each request. Format sizes differ by design; record those
differences. The isolated runtime has a three-hour bound including cleanup.

Freeze all source and artifact hashes. Preserve every call, raw response,
failure, stop reason, provider token count and elapsed time. Report exact
decisions, correct abstentions, wrong targets, wrong additions and operational
failures separately. Independently reconstruct every score from original final
responses and expected alternatives. Verify the actual outbound formats,
equal prompts and source inputs, model identity, routes and all accounting.
Do not publish raw reasoning or private conversation material.

A changed decoder qualifies for a separately frozen three-repetition
confirmation only if all 20 cases pass with zero wrong targets, wrong additions
or operational failures. Even a clean screen cannot establish fresh quality,
full extraction quality or safe production defaults. Preserve a failed screen
before selecting further work; do not rerun it to replace inconvenient results.

Unload the model, stop only owned processes and verify their absence and closed
ports. Preserve cleanup warnings. No database, embedding, retrieval, native
host, live configuration, installed pilot artifact or publication is involved.
