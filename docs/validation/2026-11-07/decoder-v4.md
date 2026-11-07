# Reconciliation decoder screen v4

Reversing the bounded schema's decision branches left all 20 outcomes unchanged.
Both bounded arms scored 17/20 with three wrong-target corrections. JSON-only
and unconstrained output each scored 10/20, retained those same three wrong
targets and introduced seven runtime rejections. No changed decoder qualifies
for the planned confirmation. Production defaults remain unchanged.

The [protocol](../../plans/2026-11-07-reconciliation-decoder-v4.md) and source
were frozen at `05d12e9b3d49888bf56d3c7a4c9fa3aa353fca79`. Each arm received
the same direct/full prompt, supplied classification, source events and target
versions for 20 authored development cases across six clusters. This was one
repetition. Only the outbound format argument changed. Preflight verified
all 84 expected alternatives and proved that reversal preserved the complete
schema except for branch order.

| Measure | Bounded | Reversed | JSON only | No format |
| --- | ---: | ---: | ---: | ---: |
| Exact operation and target outcomes | 17/20 | 17/20 | 10/20 | 10/20 |
| Correct unresolved controls | 6/9 | 6/9 | 0/9 | 0/9 |
| Accepted wrong support or correction targets | 3 | 3 | 3 | 3 |
| Accepted unsupported additions | 0 | 0 | 0 | 0 |
| Operational failures | 0 | 0 | 7 | 7 |
| Median observation latency | 1,395 ms | 1,355 ms | 1,208 ms | 1,195 ms |
| p95 latency, nearest rank | 1,699 ms | 1,613 ms | 1,740 ms | 1,743 ms |

Every arm made wrong corrections on the same three cases: an unspecified
environment with multiple plausible targets, a production correction with
only a staging target, and an unspecified count with multiple plausible
numeric settings. The branch-order change did not resolve these errors.
Removing the bounded schema did not resolve them either.

The seven failures in each schema-free arm were runtime rejections. Four
attempted to add corrections whose referents were missing or unrelated. One
attempted to revise a user fact from assistant history. Two tried to choose
among duplicate target identities. The runtime guards rejected all fourteen
responses. These failures receive no abstention credit and are separate from
the accepted wrong targets. Zero accepted wrong additions does not mean that
the model never proposed one.

| Provider accounting | Bounded | Reversed | JSON only | No format |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens | 14,747 | 14,747 | 14,747 | 14,747 |
| Cached prompt tokens | 12,013 | 11,948 | 11,408 | 12,023 |
| Generated tokens | 1,121 | 1,109 | 993 | 1,008 |
| Final text characters | 5,087 | 5,031 | 4,410 | 4,441 |

All 80 calls returned one response with a normal stop reason and no thinking
text. They used the same Qwen3:14b Q4_K_M digest, Ollama 0.40.9, generate route,
thinking false, temperature zero, context 8192 and one request slot. Every arm
had 4096 output tokens, a 3584 counted-input cap, a 512-token template reserve,
a 90-second provider deadline and a 120-second observation deadline. The
largest native prompt-plus-format count was 1,578; maximum reported prompt
and generated counts were 801 and 81. No response exhausted its output limit.

The independent audit reconstructed all 80 scores from original final
responses and frozen expectations. All 80 responses reproduced their recorded
result or rejection when replayed through runtime validation, with zero model
calls during the audit. It checked 135 source hashes, every actual outbound
format, all 20 identical prompt/input groups, token counts and native context
settings. The 89 focused tests, including 21 new decoder checks, and 12
documentation checks passed. Black and strict mypy passed for changed source.
The earlier full-suite result retains its own source binding.

The [receipt](decoder-v4-receipt.json) binds 165 verified private archive
members under SHA-256
`9f04eac7fdb39fde41281c017b9415177374176a9bc27bda7abee24fe1b53d02`.
The model unloaded, every listed owned process was absent and both endpoints
closed. The original tree-stop warning remains in the archive. No extra kill
or study rerun was needed.

This screen provides no evidence that reversing branch order or removing
the schema fixes the observed target errors. It does not rule out every
decoder interaction. The unchanged prompt still asks the model to follow a
supplied schema, which the two schema-free arms do not receive. These are
known development cases; the result does not establish fresh quality or full
extraction behavior. Live settings, installed pilot artifacts, production
defaults and publication state were unchanged. Fresh evaluation, persisted
extraction and the actual daily-use pilot remain open.
